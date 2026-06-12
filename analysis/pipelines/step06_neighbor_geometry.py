"""Step 06: inspect neighbor geometry for relative kNN design.

This checkpoint-backed step compares global kNN distance, predicted-class kNN
distance, and relative neighbor scores across the full-spectrum sample groups.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.core.model_cache import ModelCache, iter_dataset_model_caches
from analysis.core.table_store import write_artifacts
from analysis.core.benchmarks import ACTIVE_DATASETS, CACHE_DIR, DATASET_NETWORKS, DATASET_SAMPLE_GROUPS, REBUILD_ANALYSIS_OUTPUTS, RESULTS_DIR
from analysis.core.scoring import group_summary_records, pairwise_metric_records, summarize_group_table, summarize_metric_table

OUTPUT_ROOT = RESULTS_DIR / 'neighbor_geometry'
EPS = 1e-6
K = 50
RKNN_TIE_BREAK_WEIGHT = 0.001
SEARCH_ROWS = 32768


def require_faiss():
    """Import FAISS with a clear error for this optional analysis."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError('Neighbor-geometry analysis requires faiss to be installed.') from exc
    return faiss


def prepare_feature_rows(values: np.ndarray) -> np.ndarray:
    """Convert feature rows to float32 and L2-normalize them in place when possible."""
    features = np.asarray(values, dtype=np.float32)
    if not features.flags.writeable:
        features = features.copy()
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    np.maximum(norms, EPS, out=norms)
    features /= norms
    return features


def split_cache_path(model_cache: ModelCache, group: str, split_name: str) -> Path:
    """Return the on-disk cache path for one model-cache split."""
    return model_cache.cache_root / f'{group}__{split_name}'


def load_split_array(model_cache: ModelCache, group: str, split_name: str, array_name: str, *, mmap_mode: str | None = None) -> np.ndarray:
    """Load one cached split array without forcing the full ModelOutputs tuple into RAM."""
    return np.load(split_cache_path(model_cache, group, split_name) / f'{array_name}.npy', mmap_mode=mmap_mode, allow_pickle=False)


def build_global_index(train_features: np.ndarray, faiss):
    """Build the global FAISS index once."""
    global_index = faiss.IndexFlatL2(train_features.shape[1])
    global_index.add(train_features.astype(np.float32, copy=False))
    return global_index


def class_rows_by_label(train_labels: np.ndarray) -> dict[int, np.ndarray]:
    """Store row ids per class without duplicating the training feature matrix."""
    return {
        int(class_index): np.flatnonzero(train_labels == class_index)
        for class_index in np.unique(train_labels)
    }


def kth_distance(index, queries: np.ndarray, k: int) -> np.ndarray:
    """Return sqrt-L2 distance to the kth nearest neighbor."""
    k = max(1, min(int(k), int(index.ntotal)))
    distances, _ = index.search(queries.astype(np.float32, copy=False), k)
    return np.sqrt(distances[:, k - 1])


def global_distances(feature_array: np.ndarray, global_index, k: int) -> np.ndarray:
    """Compute global kth-neighbor distances while streaming a split from disk."""
    distances = np.empty(feature_array.shape[0], dtype=np.float32)
    for start in range(0, feature_array.shape[0], SEARCH_ROWS):
        end = min(start + SEARCH_ROWS, feature_array.shape[0])
        queries = prepare_feature_rows(feature_array[start:end])
        distances[start:end] = kth_distance(global_index, queries, k)
    return distances


def predicted_class_distance(train_features: np.ndarray, train_class_rows: dict[int, np.ndarray], feature_array: np.ndarray, preds: np.ndarray, k: int, faiss) -> np.ndarray:
    """Return predicted-class kth-neighbor distances without keeping all class indexes."""
    distances = np.empty(len(preds), dtype=np.float32)
    for class_index in np.unique(preds):
        query_rows = np.flatnonzero(preds == class_index)
        train_rows = train_class_rows.get(int(class_index))
        if train_rows is None or train_rows.size == 0:
            distances[query_rows] = np.inf
            continue

        class_features = train_features[train_rows].astype(np.float32, copy=False)
        index = faiss.IndexFlatL2(train_features.shape[1])
        index.add(class_features)
        class_k = max(1, min(int(k), int(index.ntotal)))
        for start in range(0, query_rows.size, SEARCH_ROWS):
            rows = query_rows[start:start + SEARCH_ROWS]
            queries = prepare_feature_rows(feature_array[rows])
            distances[rows] = kth_distance(index, queries, class_k)
        del index, class_features
    return distances


def compute_neighbor_scores(model_cache: ModelCache, split_group: str, split_name: str, global_index, train_features: np.ndarray, train_class_rows: dict[int, np.ndarray], k: int, faiss) -> dict[str, np.ndarray]:
    """Compute global, class-local, and relative neighbor scores."""
    features = load_split_array(model_cache, split_group, split_name, 'features', mmap_mode='r')
    preds = load_split_array(model_cache, split_group, split_name, 'preds').astype(np.int64, copy=False)
    d_global = global_distances(features, global_index, k)
    d_class = predicted_class_distance(train_features, train_class_rows, features, preds, k, faiss)
    return {
        'global_knn': -d_global,
        'class_knn': -d_class,
        'rknn': (d_global - d_class) - RKNN_TIE_BREAK_WEIGHT * d_global,
        'class_distance_ratio': d_global / np.maximum(d_class, EPS),
    }


def gather_group_scores(model_cache: ModelCache, dataset_name: str, global_index, train_features: np.ndarray, train_class_rows: dict[int, np.ndarray], k: int, faiss) -> dict[str, dict[str, np.ndarray]]:
    """Compute neighbor score arrays for every analysis sample group."""
    grouped_scores: dict[str, dict[str, list[np.ndarray]]] = {}
    for sample_group, splits in DATASET_SAMPLE_GROUPS[dataset_name].items():
        grouped_scores[sample_group] = {}
        for split_group, split_name in splits:
            score_map = compute_neighbor_scores(model_cache, split_group, split_name, global_index, train_features, train_class_rows, k, faiss)
            for score_name, values in score_map.items():
                grouped_scores[sample_group].setdefault(score_name, []).append(values)
    return {
        sample_group: {
            score_name: np.concatenate(parts, axis=0)
            for score_name, parts in score_parts.items()
        }
        for sample_group, score_parts in grouped_scores.items()
    }


def write_outputs(group_summary: pd.DataFrame, group_aggregates: pd.DataFrame, metric_frame: pd.DataFrame, metric_summary: pd.DataFrame, output_dir: Path, k: int) -> None:
    """Write all step 06 tables and k metadata."""
    write_artifacts(output_dir, {
        'neighbor_group_summary': group_summary,
        'neighbor_group_aggregates': group_aggregates,
        'neighbor_pairwise_metrics': metric_frame,
        'neighbor_pairwise_summary': metric_summary,
    }, summary={'k': int(k)})


def run_dataset(dataset: str, output_dir: Path) -> None:
    network = DATASET_NETWORKS[dataset]
    output_dir = output_dir.resolve()

    group_records = []
    metric_records = []
    for checkpoint_name, checkpoint_path, model_cache in iter_dataset_model_caches(CACHE_DIR, dataset):
        faiss = require_faiss()
        train_features, train_labels = model_cache.get_arrays('id', 'train', 'features', 'labels')
        train_features = prepare_feature_rows(train_features)
        train_class_rows = class_rows_by_label(train_labels.astype(np.int64, copy=False))
        del train_labels

        global_index = build_global_index(train_features, faiss)
        grouped_scores = gather_group_scores(model_cache, dataset, global_index, train_features, train_class_rows, K, faiss)
        extra = {'k': K}
        group_records.extend(group_summary_records(checkpoint_name, checkpoint_path, dataset, network, grouped_scores, extra=extra))
        metric_records.extend(pairwise_metric_records(checkpoint_name, checkpoint_path, dataset, network, grouped_scores, extra=extra))
        del global_index, train_features, train_class_rows, grouped_scores
        gc.collect()

    group_summary = pd.DataFrame.from_records(group_records)
    metric_frame = pd.DataFrame.from_records(metric_records)
    group_aggregates = summarize_group_table(group_summary)
    metric_summary = summarize_metric_table(metric_frame)
    write_outputs(group_summary, group_aggregates, metric_frame, metric_summary, output_dir, K)


def main() -> None:
    """CLI entrypoint for step 06."""
    for dataset in ACTIVE_DATASETS:
        output_dir = OUTPUT_ROOT / dataset
        sentinel_path = output_dir / 'summary.json'
        if sentinel_path.exists() and not REBUILD_ANALYSIS_OUTPUTS:
            print(f'Skipping because {sentinel_path} already exists.')
            continue
        run_dataset(dataset, output_dir)
        print(f'Wrote neighbor-geometry analysis to {output_dir}')


if __name__ == '__main__':
    main()
