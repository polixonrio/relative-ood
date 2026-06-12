"""Step 04: compare global and class-conditioned activation scores.

This checkpoint-backed step asks whether class-conditional Gaussian structure
improves the hard CS-ID versus near-OOD separation compared with global
activation distance scores.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.covariance

from analysis.core.model_cache import ModelCache, iter_dataset_model_caches
from analysis.core.table_store import figures_dir, write_artifacts
from analysis.core.benchmarks import ACTIVE_DATASETS, CACHE_DIR, DATASET_NETWORKS, DATASET_SAMPLE_GROUPS, REBUILD_ANALYSIS_OUTPUTS, RESULTS_DIR
from analysis.core.scoring import pairwise_metric_records, prepare_activation

OUTPUT_ROOT = RESULTS_DIR / 'class_conditioning_gap'
ACTIVATIONS = ['features', 'logits']
ANALYSIS_DATASETS = [dataset for dataset in ACTIVE_DATASETS if dataset != 'imagenet']


def empirical_precision(values: np.ndarray) -> np.ndarray:
    estimator = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    try:
        estimator.fit(values)
        return estimator.precision_.astype(np.float32, copy=False)
    except np.linalg.LinAlgError:
        values64 = np.asarray(values, dtype=np.float64)
        covariance = np.cov(values64, rowvar=False, bias=True)
        scale = float(np.trace(covariance) / max(covariance.shape[0], 1))
        ridge = max(scale * 1e-6, 1e-6)
        precision = np.linalg.pinv(covariance + np.eye(covariance.shape[0]) * ridge, hermitian=True)
        return precision.astype(np.float32, copy=False)


def fit_global_gaussian(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), empirical_precision(values)


def fit_class_gaussian(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    class_means = []
    centered = []
    for class_index in range(int(labels.max()) + 1):
        class_values = values[labels == class_index]
        mean = class_values.mean(axis=0)
        class_means.append(mean)
        centered.append(class_values - mean[None, :])
    return np.stack(class_means), empirical_precision(np.concatenate(centered, axis=0))


def global_scores(values: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    centered = values - mean[None, :]
    return -((centered @ precision) * centered).sum(axis=1)


def classwise_scores(values: np.ndarray, class_means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    value_proj = values @ precision
    class_proj = class_means @ precision
    value_quadratic = (value_proj * values).sum(axis=1, keepdims=True)
    class_quadratic = (class_proj * class_means).sum(axis=1)[None, :]
    cross = value_proj @ class_means.T
    return -(value_quadratic - 2.0 * cross + class_quadratic)


def top2_margin(class_scores: np.ndarray) -> np.ndarray:
    partition = np.partition(class_scores, kth=class_scores.shape[1] - 2, axis=1)
    top2 = partition[:, -2:]
    return top2[:, -1] - top2[:, -2]


def gather_scored_groups(model_cache: ModelCache, dataset_name: str, activation_name: str, global_mean: np.ndarray, global_precision: np.ndarray, class_means: np.ndarray, class_precision: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Score all sample groups with global and class-conditional distances."""
    grouped_scores: dict[str, dict[str, list[np.ndarray]]] = {}
    for sample_group, splits in DATASET_SAMPLE_GROUPS[dataset_name].items():
        grouped_scores[sample_group] = {'global_distance': [], 'predicted_class_distance': [], 'max_class_distance': [], 'top2_margin': []}
        for split_group, split_name in splits:
            outputs = model_cache.get_outputs(split_group, split_name)
            values = prepare_activation(getattr(outputs, activation_name), activation_name)
            global_conf = global_scores(values, global_mean, global_precision)
            class_conf = classwise_scores(values, class_means, class_precision)
            indices = np.arange(class_conf.shape[0])
            grouped_scores[sample_group]['global_distance'].append(global_conf)
            grouped_scores[sample_group]['predicted_class_distance'].append(class_conf[indices, outputs.preds])
            grouped_scores[sample_group]['max_class_distance'].append(class_conf.max(axis=1))
            grouped_scores[sample_group]['top2_margin'].append(top2_margin(class_conf))

    return {
        sample_group: {
            scorer: np.concatenate(parts, axis=0)
            for scorer, parts in scorer_parts.items()
        }
        for sample_group, scorer_parts in grouped_scores.items()
    }


def build_summary_table(metric_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pairwise metrics and add the hard-case balance score."""
    if metric_frame.empty:
        return metric_frame
    summary = (metric_frame.groupby(['activation', 'scorer', 'comparison'], observed=True).mean(numeric_only=True).reset_index())
    if {'clean_id__vs__cs_id', 'cs_id__vs__near_ood'}.issubset(set(summary['comparison'])):
        overlap = summary[summary['comparison'] == 'clean_id__vs__cs_id'][['activation', 'scorer',
                                                                           'overlap_coeff']].rename(columns={'overlap_coeff': 'clean_csid_overlap'})
        hard = summary[summary['comparison'] == 'cs_id__vs__near_ood'][['activation', 'scorer', 'auroc']].rename(columns={'auroc': 'csid_near_auroc'})
        joined = overlap.merge(hard, on=['activation', 'scorer'], how='inner')
        joined['hard_case_balance'] = joined['csid_near_auroc'] - (joined['clean_csid_overlap'] * 100.0)
        summary = summary.merge(joined, on=['activation', 'scorer'], how='left')
    return summary.sort_values(['activation', 'scorer', 'comparison']).reset_index(drop=True)


def plot_hard_case_summary(summary: pd.DataFrame, figures_dir: Path) -> None:
    """Plot the strongest class-structure scorers for CS-ID vs near-OOD."""
    hard = summary[summary['comparison'] == 'cs_id__vs__near_ood'].copy()
    if hard.empty:
        return
    hard['label'] = hard['activation'] + ' / ' + hard['scorer']
    ranked = hard.sort_values(['auroc', 'label'], ascending=[False, True]).head(10)
    positions = np.arange(len(ranked))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(positions, ranked['auroc'].to_numpy(dtype=float))
    ax.set_ylabel('AUROC')
    ax.set_title('Hard-case class-structure performance')
    ax.set_xticks(positions)
    ax.set_xticklabels(ranked['label'].tolist(), rotation=35, ha='right')
    fig.tight_layout()
    fig.savefig(figures_dir / 'hard_case_class_structure.png', dpi=180)
    plt.close(fig)


def write_outputs(metrics: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    """Write step 04 metric tables, summary JSON, and figure outputs."""
    plot_dir = figures_dir(output_dir)
    write_artifacts(output_dir, {
        'class_conditioning_metrics': metrics,
        'class_conditioning_summary': summary,
    }, summary={'activations': ACTIVATIONS})
    plot_hard_case_summary(summary, plot_dir)


def run_dataset(dataset: str, output_dir: Path) -> None:
    network = DATASET_NETWORKS[dataset]
    output_dir = output_dir.resolve()

    metric_records = []
    for checkpoint_name, checkpoint_path, model_cache in iter_dataset_model_caches(CACHE_DIR, dataset):
        train_outputs = model_cache.get_outputs('id', 'train')

        for activation_name in ACTIVATIONS:
            train_values = prepare_activation(getattr(train_outputs, activation_name), activation_name)
            global_mean, global_precision = fit_global_gaussian(train_values)
            class_means, class_precision = fit_class_gaussian(train_values, train_outputs.labels)
            grouped_scores = gather_scored_groups(
                model_cache, dataset, activation_name, global_mean, global_precision, class_means, class_precision
            )
            metric_records.extend(pairwise_metric_records(
                checkpoint_name, checkpoint_path, dataset, network, grouped_scores,
                score_column='scorer', extra={'activation': activation_name},
            ))

    metrics = pd.DataFrame.from_records(metric_records)
    summary = build_summary_table(metrics)
    write_outputs(metrics, summary, output_dir)


def main() -> None:
    """CLI entrypoint for step 04."""
    for dataset in ANALYSIS_DATASETS:
        output_dir = OUTPUT_ROOT / dataset
        sentinel_path = output_dir / 'summary.json'
        if sentinel_path.exists() and not REBUILD_ANALYSIS_OUTPUTS:
            print(f'Skipping because {sentinel_path} already exists.')
            continue
        run_dataset(dataset, output_dir)
        print(f'Wrote class-structure analysis to {output_dir}')


if __name__ == '__main__':
    main()
