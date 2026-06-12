"""Step 05: inspect logit-space score geometry.

This checkpoint-backed step compares simple logit scores, raw TSS, and rTSS
across sample groups, and records classwise TSS baseline spread on train data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.core.model_cache import ModelCache, iter_dataset_model_caches
from analysis.core.table_store import write_artifacts
from analysis.core.benchmarks import ACTIVE_DATASETS, CACHE_DIR, DATASET_NETWORKS, DATASET_SAMPLE_GROUPS, REBUILD_ANALYSIS_OUTPUTS, RESULTS_DIR
from analysis.core.methods.rtss import RTSSState, fit_state, score_outputs, tss_from_logits
from analysis.core.scoring import group_summary_records, pairwise_metric_records, summarize_group_table, summarize_metric_table

OUTPUT_ROOT = RESULTS_DIR / 'logit_score_geometry'
T1 = 1.0
T2 = 0.5


def energy_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute energy confidence from logits."""
    scaled = logits / float(temperature)
    return float(temperature) * np.logaddexp.reduce(scaled, axis=1)


def neg_entropy_from_logits(logits: np.ndarray) -> np.ndarray:
    """Compute negative predictive entropy from logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.logaddexp.reduce(shifted, axis=1, keepdims=True)
    probs = np.exp(log_probs)
    return (probs * log_probs).sum(axis=1)


def top2_margin_from_logits(logits: np.ndarray) -> np.ndarray:
    """Compute the top-1 minus top-2 logit margin."""
    partition = np.partition(logits, kth=logits.shape[1] - 2, axis=1)
    top2 = partition[:, -2:]
    return top2[:, -1] - top2[:, -2]


def compute_logit_scores(logits: np.ndarray, preds: np.ndarray, state: RTSSState, t1: float, t2: float) -> dict[str, np.ndarray]:
    """Return all logit-derived score arrays used by step 05."""
    return {
        'max_logit': logits.max(axis=1),
        'energy': energy_from_logits(logits),
        'neg_entropy': neg_entropy_from_logits(logits),
        'logit_margin': top2_margin_from_logits(logits),
        'neg_tss': -tss_from_logits(logits, t1, t2),
        'rtss': score_outputs(state, logits, preds),
    }


def gather_group_scores(model_cache: ModelCache, dataset_name: str, state: RTSSState, t1: float, t2: float) -> dict[str, dict[str, np.ndarray]]:
    """Compute logit-score arrays for every analysis sample group."""
    grouped_scores: dict[str, dict[str, list[np.ndarray]]] = {}
    for sample_group, splits in DATASET_SAMPLE_GROUPS[dataset_name].items():
        grouped_scores[sample_group] = {}
        for split_group, split_name in splits:
            outputs = model_cache.get_outputs(split_group, split_name)
            logits = np.asarray(outputs.logits, dtype=np.float32)
            preds = outputs.preds
            score_map = compute_logit_scores(logits, preds, state, t1, t2)
            for score_name, values in score_map.items():
                grouped_scores[sample_group].setdefault(score_name, []).append(values)
    return {
        sample_group: {
            score_name: np.concatenate(parts, axis=0)
            for score_name, parts in score_parts.items()
        }
        for sample_group, score_parts in grouped_scores.items()
    }


def build_class_baseline_records(checkpoint_name: str, checkpoint_path: Path, dataset_name: str, network_name: str, logits: np.ndarray, preds: np.ndarray, labels: np.ndarray, t1: float, t2: float) -> list[dict]:
    """Record predicted-class TSS baseline statistics on correct train samples."""
    correct = preds == labels
    logits_fit = logits[correct] if correct.any() else logits
    preds_fit = preds[correct] if correct.any() else preds
    tss = tss_from_logits(logits_fit, t1, t2)
    records = []
    for class_index in range(int(preds_fit.max()) + 1):
        class_values = tss[preds_fit == class_index]
        if class_values.size == 0:
            continue
        records.append({
            'checkpoint_name': checkpoint_name,
            'checkpoint_path': str(checkpoint_path),
            'dataset': dataset_name,
            'network': network_name,
            'class_index': int(class_index),
            'n_correct_train': int(class_values.size),
            'tss_mean': float(class_values.mean()),
            'tss_std': float(class_values.std(ddof=1)) if class_values.size > 1 else 0.0,
        })
    return records


def build_baseline_summary(class_baselines: pd.DataFrame) -> pd.DataFrame:
    """Summarize classwise TSS baseline variation per checkpoint."""
    if class_baselines.empty:
        return class_baselines
    grouped = class_baselines.groupby(['checkpoint_name', 'checkpoint_path', 'dataset', 'network'], observed=True)
    summary = grouped.agg(
        n_classes=('class_index', 'nunique'),
        total_correct_train=('n_correct_train', 'sum'),
        class_mean_std=('tss_mean', 'std'),
        class_mean_min=('tss_mean', 'min'),
        class_mean_max=('tss_mean', 'max'),
        class_std_mean=('tss_std', 'mean'),
    ).reset_index()
    summary['class_mean_std'] = summary['class_mean_std'].fillna(0.0)
    summary['class_mean_range'] = summary['class_mean_max'] - summary['class_mean_min']
    return summary


def write_outputs(group_summary: pd.DataFrame, group_aggregates: pd.DataFrame, metric_frame: pd.DataFrame, metric_summary: pd.DataFrame, class_baselines: pd.DataFrame, baseline_summary: pd.DataFrame, output_dir: Path, t1: float, t2: float) -> None:
    """Write all step 05 tables and temperature metadata."""
    write_artifacts(output_dir, {
        'logit_group_summary': group_summary,
        'logit_group_aggregates': group_aggregates,
        'logit_pairwise_metrics': metric_frame,
        'logit_pairwise_summary': metric_summary,
        'tss_class_baselines': class_baselines,
        'tss_class_baseline_summary': baseline_summary,
    }, summary={'t1': t1, 't2': t2})


def run_dataset(dataset: str, output_dir: Path) -> None:
    network = DATASET_NETWORKS[dataset]
    output_dir = output_dir.resolve()

    group_records = []
    metric_records = []
    class_baseline_records = []
    for checkpoint_name, checkpoint_path, model_cache in iter_dataset_model_caches(CACHE_DIR, dataset):
        train_outputs = model_cache.get_outputs('id', 'train')
        logits = np.asarray(train_outputs.logits, dtype=np.float32)
        preds = train_outputs.preds
        labels = train_outputs.labels
        fit_mask = preds == labels
        fit_logits = logits[fit_mask] if fit_mask.any() else logits
        fit_preds = preds[fit_mask] if fit_mask.any() else preds
        state = fit_state(fit_logits, fit_preds, t1=T1, t2=T2)

        grouped_scores = gather_group_scores(model_cache, dataset, state, T1, T2)
        group_records.extend(group_summary_records(checkpoint_name, checkpoint_path, dataset, network, grouped_scores))
        metric_records.extend(pairwise_metric_records(checkpoint_name, checkpoint_path, dataset, network, grouped_scores))
        class_baseline_records.extend(build_class_baseline_records(checkpoint_name, checkpoint_path, dataset, network, logits, preds, labels, T1, T2))

    group_summary = pd.DataFrame.from_records(group_records)
    metric_frame = pd.DataFrame.from_records(metric_records)
    group_aggregates = summarize_group_table(group_summary)
    metric_summary = summarize_metric_table(metric_frame)
    class_baselines = pd.DataFrame.from_records(class_baseline_records)
    baseline_summary = build_baseline_summary(class_baselines)
    write_outputs(
        group_summary, group_aggregates, metric_frame, metric_summary, class_baselines, baseline_summary, output_dir, T1, T2
    )


def main() -> None:
    """CLI entrypoint for step 05."""
    for dataset in ACTIVE_DATASETS:
        output_dir = OUTPUT_ROOT / dataset
        sentinel_path = output_dir / 'summary.json'
        if sentinel_path.exists() and not REBUILD_ANALYSIS_OUTPUTS:
            print(f'Skipping because {sentinel_path} already exists.')
            continue
        run_dataset(dataset, output_dir)
        print(f'Wrote logit-score geometry analysis to {output_dir}')


if __name__ == '__main__':
    main()
