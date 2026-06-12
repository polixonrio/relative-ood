"""Step 09: characterize near-OOD hardness via feature-covariance alignment, centroid distance, and kNN overlap."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.core.benchmarks import ACTIVE_DATASETS, CACHE_DIR, DATASET_NETWORKS, DATASET_SAMPLE_GROUPS, REBUILD_ANALYSIS_OUTPUTS, RESULTS_DIR
from analysis.core.model_cache import iter_dataset_model_caches
from analysis.core.scoring import l2_normalize, prepare_activation
from analysis.core.table_store import write_artifacts

OUTPUT_ROOT = RESULTS_DIR / 'near_ood_hardness'
K = 50


def feature_covariance_alignment(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Compare unpaired feature distributions by normalized covariance alignment."""
    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    if a.shape[0] < 2 or b.shape[0] < 2:
        return 0.0
    cov_a = (a.T @ a) / float(a.shape[0] - 1)
    cov_b = (b.T @ b) / float(b.shape[0] - 1)
    numerator = float(np.trace(cov_a @ cov_b))
    denominator = float(np.linalg.norm(cov_a, ord='fro') * np.linalg.norm(cov_b, ord='fro'))
    return 0.0 if denominator < 1e-12 else float(numerator / denominator)


def class_centroids(features: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    return {int(class_id): values[label_array == class_id].mean(axis=0) for class_id in np.unique(label_array)}


def nearest_centroid_distances(features: np.ndarray, centroids: dict[int, np.ndarray]) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    centroid_matrix = np.stack(list(centroids.values()), axis=0)
    value_sq = (values ** 2).sum(axis=1, keepdims=True)
    centroid_sq = (centroid_matrix ** 2).sum(axis=1, keepdims=True).T
    return np.sqrt(np.maximum(value_sq + centroid_sq - 2.0 * (values @ centroid_matrix.T), 0.0)).min(axis=1)


def knn_overlap(train_gpu: torch.Tensor, label_gpu: torch.Tensor, ood_features: np.ndarray, k: int, chunk_size: int = 128) -> float:
    query_normed = l2_normalize(np.asarray(ood_features, dtype=np.float32))
    k = max(1, min(int(k), int(train_gpu.shape[0])))
    matches = []
    for start in range(0, len(query_normed), chunk_size):
        stop = min(start + chunk_size, len(query_normed))
        query_gpu = torch.from_numpy(query_normed[start:stop].astype(np.float16, copy=False)).cuda()
        dots = query_gpu @ train_gpu.T
        distances = (2.0 - 2.0 * dots).clamp(min=0.0)
        indices = torch.topk(distances, k, dim=1, largest=False).indices
        neighbor_labels = label_gpu[indices]
        same_class = (neighbor_labels == neighbor_labels[:, 0:1]).all(dim=1)
        matches.append(same_class.float().cpu().numpy())
        del query_gpu, dots, distances, indices, neighbor_labels, same_class
    return float(np.concatenate(matches).mean())


def gather_ood_features(model_cache, dataset: str, sample_group: str) -> list[tuple[str, np.ndarray]]:
    features = []
    for split_group, split_name in DATASET_SAMPLE_GROUPS[dataset][sample_group]:
        outputs = model_cache.get_outputs(split_group, split_name)
        features.append((split_name, prepare_activation(outputs.features, 'features')))
    return features


def hardness_records(id_train_features: np.ndarray, id_train_labels: np.ndarray, id_test_features: np.ndarray, ood_groups: list[tuple[str, list[tuple[str, np.ndarray]]]]) -> list[dict]:
    if not torch.cuda.is_available():
        raise RuntimeError('Near-OOD hardness KNN overlap requires CUDA for exact cached feature search.')

    centroids = class_centroids(id_train_features, id_train_labels)
    id_normed = l2_normalize(np.asarray(id_train_features, dtype=np.float32))
    train_gpu = torch.from_numpy(id_normed.astype(np.float16, copy=False)).cuda()
    label_gpu = torch.from_numpy(np.asarray(id_train_labels, dtype=np.int64)).cuda()
    records = []
    try:
        for ood_type, feature_list in ood_groups:
            for ood_dataset, ood_features in feature_list:
                distances = nearest_centroid_distances(ood_features, centroids)
                records.append({
                    'ood_dataset': ood_dataset,
                    'ood_type': ood_type,
                    'feature_covariance_alignment_id_ood': feature_covariance_alignment(id_test_features, ood_features),
                    'mean_centroid_dist': float(distances.mean()),
                    'std_centroid_dist': float(distances.std()),
                    'knn_overlap_ratio': knn_overlap(train_gpu, label_gpu, ood_features, K),
                })
    finally:
        del train_gpu, label_gpu
    return records


def summary_payload(hardness_metrics: pd.DataFrame) -> dict:
    near = hardness_metrics[hardness_metrics['ood_type'] == 'near']
    far = hardness_metrics[hardness_metrics['ood_type'] == 'far']
    return {
        'near_ood_mean_feature_covariance_alignment': near['feature_covariance_alignment_id_ood'].mean(),
        'near_ood_mean_centroid_dist': near['mean_centroid_dist'].mean(),
        'near_ood_mean_knn_overlap': near['knn_overlap_ratio'].mean(),
        'far_ood_mean_feature_covariance_alignment': far['feature_covariance_alignment_id_ood'].mean(),
        'far_ood_mean_centroid_dist': far['mean_centroid_dist'].mean(),
        'far_ood_mean_knn_overlap': far['knn_overlap_ratio'].mean(),
        'feature_covariance_alignment_gap_near_minus_far': near['feature_covariance_alignment_id_ood'].mean() - far['feature_covariance_alignment_id_ood'].mean(),
        'centroid_dist_gap_far_minus_near': far['mean_centroid_dist'].mean() - near['mean_centroid_dist'].mean(),
        'knn_overlap_gap_near_minus_far': near['knn_overlap_ratio'].mean() - far['knn_overlap_ratio'].mean(),
    }


def run_dataset(dataset: str, output_dir: Path) -> None:
    network = DATASET_NETWORKS[dataset]

    rows = []
    for checkpoint_name, checkpoint_path, model_cache in iter_dataset_model_caches(CACHE_DIR, dataset):
        train_outputs = model_cache.get_outputs('id', 'train')
        test_outputs = model_cache.get_outputs('id', 'test')
        id_train_features = prepare_activation(train_outputs.features, 'features')
        id_test_features = prepare_activation(test_outputs.features, 'features')
        ood_groups = [('near', gather_ood_features(model_cache, dataset, 'near_ood')), ('far', gather_ood_features(model_cache, dataset, 'far_ood'))]
        for record in hardness_records(id_train_features, train_outputs.labels, id_test_features, ood_groups):
            record.update({
                'checkpoint_name': checkpoint_name,
                'checkpoint_path': str(checkpoint_path),
                'dataset': dataset,
                'network': network,
                'k': K,
            })
            rows.append(record)

    hardness_metrics = pd.DataFrame.from_records(rows)
    write_artifacts(
        output_dir,
        {'dataset_hardness_metrics': hardness_metrics},
        summary=summary_payload(hardness_metrics),
        summary_name='nearness_summary.json',
    )


def main() -> None:
    for dataset in ACTIVE_DATASETS:
        output_dir = OUTPUT_ROOT / dataset
        sentinel_path = output_dir / 'nearness_summary.json'
        if sentinel_path.exists() and not REBUILD_ANALYSIS_OUTPUTS:
            print(f'Skipping because {sentinel_path} already exists.')
            continue
        run_dataset(dataset, output_dir)
        print(f'Wrote near-OOD hardness analysis to {output_dir}')


if __name__ == '__main__':
    main()
