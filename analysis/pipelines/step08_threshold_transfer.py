"""Step 08: test whether clean-ID score thresholds transfer to cs-ID, near-OOD, and far-OOD groups."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.core.benchmarks import DATASET_NETWORKS, FULL_SPECTRUM_DATASETS, REBUILD_ANALYSIS_OUTPUTS, RESULTS_DIR
from analysis.core.methods.registry import FAMILY_COLORS, FAMILY_ORDER, METHOD_FAMILY, METHOD_ORDER
from analysis.core.table_store import figures_dir, iter_dataframes, write_artifacts

SCORE_OVERLAP_ROOT = RESULTS_DIR / 'score_overlap'
OUTPUT_ROOT = RESULTS_DIR / 'threshold_transfer'
TARGET_TPRS = [0.95]
PROTOCOLS = ['full_spectrum']


def collect_grouped_scores(path: Path, protocols: list[str]) -> dict[tuple[str, str, str, str], dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f'Input table not found: {path}')
    required = {'checkpoint_name', 'protocol', 'method', 'family', 'sample_group', 'score'}
    columns = list(required)
    grouped_scores: dict[tuple[str, str, str, str], dict[str, np.ndarray]] = {}
    saw_rows = False
    group_keys = ['checkpoint_name', 'protocol', 'method', 'family', 'sample_group']

    for frame in iter_dataframes(path, columns=columns):
        if frame.empty:
            continue
        saw_rows = True
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f'Missing required columns in {path}: {sorted(missing)}')
        frame = frame[frame['protocol'].isin(protocols)]
        frame = frame[frame['method'].isin(METHOD_ORDER)].copy()
        frame['family'] = frame['method'].map(METHOD_FAMILY)
        if frame.empty:
            continue
        for keys, group in frame.groupby(group_keys, sort=False, observed=True):
            (checkpoint_name, protocol, method, family, sample_group) = keys
            # Step 02 stores an OOD-oriented plotting score: score = -confidence.
            # Threshold transfer is defined in OpenOOD confidence orientation,
            # where larger values are more ID-like, so convert back here.
            scores = -group['score'].to_numpy(dtype=float, copy=False)
            key = (str(checkpoint_name), str(protocol), str(method), str(family))
            group_dict = grouped_scores.setdefault(key, {})
            existing = group_dict.get(str(sample_group))
            if existing is None:
                group_dict[str(sample_group)] = scores
            else:
                group_dict[str(sample_group)] = np.concatenate([existing, scores])

    if not saw_rows:
        return {}
    return grouped_scores


def compute_threshold_metrics(clean_id_scores: np.ndarray, cs_id_scores: np.ndarray | None, near_ood_scores: np.ndarray | None, far_ood_scores: np.ndarray | None, target_tpr: float) -> dict:
    # OpenOOD postprocessors use larger confidence scores for more ID-like
    # samples, so a 95% clean-ID operating point is the 5th percentile and
    # accepted samples are those above the threshold.
    threshold = float(np.percentile(clean_id_scores, (1.0 - target_tpr) * 100.0))
    tpr_clean_id = float(np.mean(clean_id_scores >= threshold))
    fpr_cs_id = float(np.mean(cs_id_scores < threshold)) if cs_id_scores is not None and cs_id_scores.size > 0 else np.nan
    tnr_near_ood = float(np.mean(near_ood_scores < threshold)) if near_ood_scores is not None and near_ood_scores.size > 0 else np.nan
    tnr_far_ood = float(np.mean(far_ood_scores < threshold)) if far_ood_scores is not None and far_ood_scores.size > 0 else np.nan
    tpr_cs_id = float(np.mean(cs_id_scores >= threshold)) if cs_id_scores is not None and cs_id_scores.size > 0 else np.nan

    transfer_gap_clean_to_cs = tpr_clean_id - tpr_cs_id if not np.isnan(tpr_cs_id) else np.nan
    transfer_gap_clean_to_near = tpr_clean_id - tnr_near_ood if not np.isnan(tnr_near_ood) else np.nan
    transfer_gap_clean_to_far = tpr_clean_id - tnr_far_ood if not np.isnan(tnr_far_ood) else np.nan

    return {
        'target_tpr': target_tpr,
        'threshold': threshold,
        'tpr_clean_id': tpr_clean_id,
        'fpr_cs_id': fpr_cs_id,
        'tnr_near_ood': tnr_near_ood,
        'tnr_far_ood': tnr_far_ood,
        'transfer_gap_clean_to_cs': transfer_gap_clean_to_cs,
        'transfer_gap_clean_to_near': transfer_gap_clean_to_near,
        'transfer_gap_clean_to_far': transfer_gap_clean_to_far,
    }


def build_metric_records(grouped_scores: dict, target_tprs: list[float]) -> list[dict]:
    records = []
    for (checkpoint_name, protocol, method, family), group_dict in grouped_scores.items():
        clean_id_scores = group_dict.get('clean_id')
        if clean_id_scores is None or clean_id_scores.size == 0:
            continue
        cs_id_scores = group_dict.get('cs_id')
        near_ood_scores = group_dict.get('near_ood')
        far_ood_scores = group_dict.get('far_ood')
        for target_tpr in target_tprs:
            metrics = compute_threshold_metrics(clean_id_scores, cs_id_scores, near_ood_scores, far_ood_scores, target_tpr)
            records.append(
                {
                    'checkpoint_name': checkpoint_name,
                    'protocol': protocol,
                    'method': method,
                    'family': family,
                    **metrics,
                }
            )
    return records


def build_method_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    keys = ['protocol', 'method', 'family', 'target_tpr']
    for column in ['network', 'dataset']:
        if column in metrics.columns:
            keys.insert(0, column)
    summary = (
        metrics.groupby(keys, observed=True)
        .agg(
            n_checkpoints=('checkpoint_name', 'nunique'),
            mean_tpr_clean_id=('tpr_clean_id', 'mean'),
            mean_fpr_cs_id=('fpr_cs_id', 'mean'),
            mean_tnr_near_ood=('tnr_near_ood', 'mean'),
            mean_tnr_far_ood=('tnr_far_ood', 'mean'),
            mean_transfer_gap_clean_to_cs=('transfer_gap_clean_to_cs', 'mean'),
            mean_transfer_gap_clean_to_near=('transfer_gap_clean_to_near', 'mean'),
            mean_transfer_gap_clean_to_far=('transfer_gap_clean_to_far', 'mean'),
            std_transfer_gap_clean_to_cs=('transfer_gap_clean_to_cs', 'std'),
            std_transfer_gap_clean_to_near=('transfer_gap_clean_to_near', 'std'),
            std_transfer_gap_clean_to_far=('transfer_gap_clean_to_far', 'std'),
        )
        .reset_index()
    )
    summary['method'] = pd.Categorical(summary['method'], categories=METHOD_ORDER, ordered=True)
    summary['family'] = pd.Categorical(summary['family'], categories=FAMILY_ORDER, ordered=True)
    sort_cols = [column for column in ['dataset', 'network', 'protocol', 'target_tpr', 'family', 'method'] if column in summary.columns]
    return summary.sort_values(sort_cols).reset_index(drop=True)


def build_family_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    keys = ['protocol', 'family', 'target_tpr']
    for column in ['network', 'dataset']:
        if column in metrics.columns:
            keys.insert(0, column)
    summary = (
        metrics.groupby(keys, observed=True)
        .agg(
            n_checkpoints=('checkpoint_name', 'nunique'),
            n_methods=('method', 'nunique'),
            mean_tpr_clean_id=('tpr_clean_id', 'mean'),
            mean_fpr_cs_id=('fpr_cs_id', 'mean'),
            mean_tnr_near_ood=('tnr_near_ood', 'mean'),
            mean_tnr_far_ood=('tnr_far_ood', 'mean'),
            mean_transfer_gap_clean_to_cs=('transfer_gap_clean_to_cs', 'mean'),
            mean_transfer_gap_clean_to_near=('transfer_gap_clean_to_near', 'mean'),
            mean_transfer_gap_clean_to_far=('transfer_gap_clean_to_far', 'mean'),
            std_transfer_gap_clean_to_cs=('transfer_gap_clean_to_cs', 'std'),
            std_transfer_gap_clean_to_near=('transfer_gap_clean_to_near', 'std'),
            std_transfer_gap_clean_to_far=('transfer_gap_clean_to_far', 'std'),
        )
        .reset_index()
    )
    summary['family'] = pd.Categorical(summary['family'], categories=FAMILY_ORDER, ordered=True)
    sort_cols = [column for column in ['dataset', 'network', 'protocol', 'target_tpr', 'family'] if column in summary.columns]
    return summary.sort_values(sort_cols).reset_index(drop=True)


def plot_transfer_gap_bars(metrics: pd.DataFrame, figures_dir: Path) -> None:
    if metrics.empty or 'protocol' not in metrics.columns:
        return
    subset = metrics[metrics['protocol'] == 'full_spectrum'].copy()
    if subset.empty:
        return
    target_tprs = sorted(subset['target_tpr'].dropna().unique().tolist())
    for tpr in target_tprs:
        tpr_subset = subset[subset['target_tpr'] == tpr].copy()
        if tpr_subset.empty:
            continue
        avg = (
            tpr_subset.groupby(['method', 'family'], observed=True)
            .agg(
                transfer_gap_clean_to_cs=('transfer_gap_clean_to_cs', 'mean'),
                transfer_gap_clean_to_near=('transfer_gap_clean_to_near', 'mean'),
                transfer_gap_clean_to_far=('transfer_gap_clean_to_far', 'mean'),
            )
            .reset_index()
        )
        avg['method'] = pd.Categorical(avg['method'], categories=METHOD_ORDER, ordered=True)
        avg = avg.sort_values('method').reset_index(drop=True)
        x = np.arange(len(avg))
        width = 0.25
        fig, ax = plt.subplots(figsize=(16, 6))
        colors = [FAMILY_COLORS[str(f)] for f in avg['family']]
        ax.bar(x - width, avg['transfer_gap_clean_to_cs'], width, label='clean -> cs', color=colors, alpha=0.9)
        ax.bar(x, avg['transfer_gap_clean_to_near'], width, label='clean -> near', color=colors, alpha=0.6)
        ax.bar(x + width, avg['transfer_gap_clean_to_far'], width, label='clean -> far', color=colors, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(avg['method'].tolist(), rotation=45, ha='right')
        ax.set_ylabel('Transfer gap')
        ax.set_title(f'Threshold transfer gaps by method (TPR={tpr:.2f}, full_spectrum)')
        ax.legend()
        ax.axhline(0, color='black', linewidth=0.5)
        fig.tight_layout()
        fig.savefig(figures_dir / f'transfer_gap_bar_tpr{int(tpr * 100):02d}_full_spectrum.png', dpi=180)
        plt.close(fig)


def plot_threshold_vs_fpr_cs_id(metrics: pd.DataFrame, figures_dir: Path) -> None:
    if metrics.empty or 'protocol' not in metrics.columns:
        return
    subset = metrics[metrics['protocol'] == 'full_spectrum'].copy()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    for family in FAMILY_ORDER:
        family_data = subset[subset['family'] == family]
        if family_data.empty:
            continue
        ax.scatter(
            family_data['threshold'],
            family_data['fpr_cs_id'],
            color=FAMILY_COLORS[family],
            label=family,
            alpha=0.6,
            s=40,
        )
    ax.set_xlabel('Clean-ID threshold (accept if score >= threshold)')
    ax.set_ylabel('CS-ID rejection rate')
    ax.set_title('Clean-ID threshold vs CS-ID rejection rate (full_spectrum)')
    ax.legend(title='Family')
    fig.tight_layout()
    fig.savefig(figures_dir / 'threshold_vs_fpr_cs_id.png', dpi=180)
    plt.close(fig)


def build_summary_json(metrics: pd.DataFrame, method_summary: pd.DataFrame) -> dict:
    if metrics.empty:
        return {'notes': 'No threshold transfer metrics were available.'}
    summary = {
        'notes': 'Scores follow OpenOOD confidence orientation: larger values are more ID-like, so accepted samples are score >= threshold.',
    }
    for (protocol, target_tpr), group in metrics.groupby(['protocol', 'target_tpr'], sort=False):
        key = f'{protocol}_tpr{int(target_tpr * 100):02d}'
        worst_cs = group.sort_values('fpr_cs_id', ascending=False).iloc[0]
        worst_near = group.sort_values('transfer_gap_clean_to_near', ascending=False).iloc[0]
        summary[key] = {
            'worst_cs_id_rejection': {
                'method': str(worst_cs['method']),
                'family': str(worst_cs['family']),
                'rejection_rate': worst_cs['fpr_cs_id'],
            },
            'worst_clean_to_near': {
                'method': str(worst_near['method']),
                'family': str(worst_near['family']),
                'gap': worst_near['transfer_gap_clean_to_near'],
            },
        }
    if not method_summary.empty:
        fs = method_summary[method_summary['protocol'] == 'full_spectrum']
        if not fs.empty:
            best_family = fs.groupby('family', observed=True)['mean_fpr_cs_id'].mean().reset_index().sort_values('mean_fpr_cs_id')
            summary['best_family_cs_id_rejection'] = best_family.iloc[0].to_dict() if not best_family.empty else {}
    return summary


def write_outputs(metrics: pd.DataFrame, method_summary: pd.DataFrame, family_summary: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = figures_dir(output_dir)
    write_artifacts(output_dir, {
        'threshold_transfer_metrics': metrics,
        'method_threshold_summary': method_summary,
        'family_threshold_summary': family_summary,
    }, summary=build_summary_json(metrics, method_summary))
    plot_transfer_gap_bars(metrics, plot_dir)
    plot_threshold_vs_fpr_cs_id(metrics, plot_dir)


def run_dataset(dataset: str, output_dir: Path) -> None:
    input_table = SCORE_OVERLAP_ROOT / dataset / 'per_sample_scores_parts'

    grouped_scores = collect_grouped_scores(input_table, PROTOCOLS)
    records = build_metric_records(grouped_scores, TARGET_TPRS)
    metrics = pd.DataFrame.from_records(records)
    if not metrics.empty:
        metrics['dataset'] = dataset
        metrics['network'] = DATASET_NETWORKS[dataset]
        metrics['family'] = pd.Categorical(metrics['family'], categories=FAMILY_ORDER, ordered=True)
        metrics['method'] = pd.Categorical(metrics['method'], categories=METHOD_ORDER, ordered=True)
        metrics = metrics.sort_values(['dataset', 'network', 'checkpoint_name', 'protocol', 'target_tpr', 'family', 'method']).reset_index(drop=True)
    method_summary = build_method_summary(metrics)
    family_summary = build_family_summary(metrics)
    write_outputs(metrics, method_summary, family_summary, output_dir)

    print(f'Wrote outputs to {output_dir}')
    print('Generated:')
    print(' - threshold_transfer_metrics.parquet')
    print(' - method_threshold_summary.parquet')
    print(' - family_threshold_summary.parquet')
    print(' - summary.json')
    print(' - figures/')


def main() -> None:
    for dataset in FULL_SPECTRUM_DATASETS:
        output_dir = OUTPUT_ROOT / dataset
        sentinel_path = output_dir / 'threshold_transfer_metrics.parquet'
        if sentinel_path.exists() and not REBUILD_ANALYSIS_OUTPUTS:
            print(f'Skipping because {sentinel_path} already exists.')
            continue
        run_dataset(dataset, output_dir)


if __name__ == '__main__':
    main()
