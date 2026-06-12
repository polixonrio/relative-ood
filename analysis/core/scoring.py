"""Shared scoring, activation preprocessing, and score-summary helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from openood.evaluators.metrics import compute_all_metrics

DEFAULT_SCORE_COMPARISONS = [('clean_id', 'cs_id', 'distribution_shift'), ('cs_id', 'near_ood', 'ood_separation'), ('clean_id', 'near_ood', 'ood_separation'), ('clean_id', 'far_ood', 'ood_separation')]


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize rows to unit length with a small zero-norm guard."""
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return values / norms


def prepare_activation(values: np.ndarray, activation_name: str) -> np.ndarray:
    """Apply the standard preprocessing for logits or feature activations."""
    array = np.asarray(values, dtype=np.float32)
    return array if activation_name == 'logits' else l2_normalize(array)


def openood_metric_array(id_tuple: tuple[np.ndarray, np.ndarray, np.ndarray], split_group: str, dataset_names: list[str], score_split, include_mean: bool = False) -> np.ndarray:
    """Compute OpenOOD metrics for one ID tuple against several OOD splits."""
    id_pred, id_conf, id_label = id_tuple
    rows = []
    for dataset_name in dataset_names:
        ood_pred, ood_conf, ood_label = score_split(split_group, dataset_name)
        rows.append(compute_all_metrics(
            np.concatenate([id_conf, ood_conf]),
            np.concatenate([id_label, -np.ones_like(ood_label)]),
            np.concatenate([id_pred, ood_pred]),
        ))
    metrics = np.array(rows, dtype=np.float32) * 100.0
    return np.concatenate([metrics, metrics.mean(axis=0, keepdims=True)], axis=0) if include_mean else metrics


def summarize_scores(array: np.ndarray) -> dict[str, float]:
    """Return count, mean, std, median, min, max, and 5/25/75/95 percentiles for one score array."""
    return {
        'n': int(array.size),
        'mean': float(array.mean()),
        'std': float(array.std(ddof=0)),
        'median': float(np.median(array)),
        'p05': float(np.percentile(array, 5)),
        'p25': float(np.percentile(array, 25)),
        'p75': float(np.percentile(array, 75)),
        'p95': float(np.percentile(array, 95)),
        'min': float(array.min()),
        'max': float(array.max()),
    }


def compute_overlap_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Compute distribution-overlap statistics for two score arrays."""
    mean_diff = abs(float(a.mean() - b.mean()))
    pooled_var = 0.5 * (float(a.var(ddof=0)) + float(b.var(ddof=0)))
    std_mean_diff = mean_diff / (pooled_var ** 0.5) if pooled_var > 0.0 else 0.0

    low = float(min(a.min(), b.min()))
    high = float(max(a.max(), b.max()))
    if low == high:
        overlap_coeff = 1.0
    else:
        hist_a, edges = np.histogram(a, bins=128, range=(low, high), density=True)
        hist_b, _ = np.histogram(b, bins=edges, density=True)
        overlap = np.minimum(hist_a, hist_b) * np.diff(edges)
        overlap_coeff = float(np.clip(overlap.sum(), 0.0, 1.0))

    return {
        'mean_diff': mean_diff,
        'std_mean_diff': float(std_mean_diff),
        'wasserstein': float(wasserstein_distance(a, b)),
        'overlap_coeff': overlap_coeff,
        'ks_stat': float(ks_2samp(a, b).statistic),
    }


def compare_scores(a: np.ndarray, b: np.ndarray, include_openood_metrics: bool = False) -> dict[str, float]:
    """Compare two score arrays with overlap stats and optional OpenOOD metrics."""
    metrics = compute_overlap_metrics(a, b)
    metrics.update({
        'mean_delta': float(b.mean() - a.mean()),
        'median_delta': float(np.median(b) - np.median(a)),
    })
    if not include_openood_metrics:
        return metrics

    conf = np.concatenate([a, b], axis=0)
    label = np.concatenate([np.zeros(a.shape[0], dtype=np.int64), -np.ones(b.shape[0], dtype=np.int64)])
    pred = np.zeros_like(label)
    fpr95, auroc, aupr_in, aupr_out, _ = compute_all_metrics(conf, label, pred)
    metrics.update({
        'auroc': float(auroc * 100.0),
        'aupr_in': float(aupr_in * 100.0),
        'aupr_out': float(aupr_out * 100.0),
        'fpr95': float(fpr95 * 100.0),
    })
    return metrics


def compare_grouped_scores(grouped_scores: dict[str, dict[str, np.ndarray]], score_name: str):
    """Build default pairwise comparisons for one named score."""
    records = []
    for group_a, group_b, kind in DEFAULT_SCORE_COMPARISONS:
        scores_a = grouped_scores[group_a][score_name]
        scores_b = grouped_scores[group_b][score_name]
        metrics = compare_scores(scores_a, scores_b, include_openood_metrics=kind == 'ood_separation')
        records.append({
            'comparison': f'{group_a}__vs__{group_b}',
            'group_a': group_a,
            'group_b': group_b,
            **metrics,
        })
    return records


def group_summary_records(checkpoint_name: str, checkpoint_path, dataset_name: str, network_name: str, grouped_scores: dict[str, dict[str, np.ndarray]], score_column: str = 'score', extra: dict | None = None) -> list[dict]:
    """Convert grouped score arrays into long-form summary-stat records."""
    records = []
    extra = extra or {}
    for sample_group, score_map in grouped_scores.items():
        for score_name, values in score_map.items():
            records.append({
                'checkpoint_name': checkpoint_name,
                'checkpoint_path': str(checkpoint_path),
                'dataset': dataset_name,
                'network': network_name,
                'sample_group': sample_group,
                score_column: score_name,
                **extra,
                **summarize_scores(values),
            })
    return records


def pairwise_metric_records(checkpoint_name: str, checkpoint_path, dataset_name: str, network_name: str, grouped_scores: dict[str, dict[str, np.ndarray]], score_column: str = 'score', extra: dict | None = None) -> list[dict]:
    """Convert grouped score arrays into long-form pairwise metric records."""
    records = []
    extra = extra or {}
    score_names = list(next(iter(grouped_scores.values())).keys())
    for score_name in score_names:
        for metrics in compare_grouped_scores(grouped_scores, score_name):
            records.append({
                'checkpoint_name': checkpoint_name,
                'checkpoint_path': str(checkpoint_path),
                'dataset': dataset_name,
                'network': network_name,
                score_column: score_name,
                **extra,
                **metrics,
            })
    return records


def summarize_group_table(group_summary: pd.DataFrame, score_column: str='score') -> pd.DataFrame:
    """Average group summary records across checkpoints."""
    return (
        group_summary.groupby(['sample_group', score_column], observed=True).mean(numeric_only=True).reset_index()
        .sort_values(['sample_group', score_column]).reset_index(drop=True)
    )


def summarize_metric_table(metric_frame: pd.DataFrame, score_column: str='score') -> pd.DataFrame:
    """Average pairwise metric records across checkpoints."""
    return (
        metric_frame.groupby([score_column, 'comparison', 'group_a', 'group_b'], observed=True).mean(numeric_only=True).reset_index()
        .sort_values(['comparison', score_column]).reset_index(drop=True)
    )
