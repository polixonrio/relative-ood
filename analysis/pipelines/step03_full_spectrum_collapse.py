"""Step 03: summarize standard-to-full-spectrum performance collapse.

This cached step consumes step 01 raw metrics and compares each method's
standard AUROC against its full-spectrum AUROC.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.core.table_store import figures_dir, write_artifacts
from analysis.core.benchmarks import FULL_SPECTRUM_DATASETS, REBUILD_ANALYSIS_OUTPUTS, RESULTS_DIR
from analysis.core.methods.registry import FAMILY_COLORS, FAMILY_ORDER, METHOD_FAMILY, METHOD_ORDER

INPUT_TABLE = RESULTS_DIR / 'ranking_stability' / 'raw_metrics.parquet'
OUTPUT_ROOT = RESULTS_DIR / 'fs_collapse'
METRICS = ['near_auroc', 'far_auroc', 'overall_auroc']


# Methods retired from the evaluated panel (superseded by GEN and SCALE); excluded
# from the family-level collapse averages so the family rows match the final panel.
PANEL_EXCLUDED_METHODS = {'MaxLogit', 'DICE'}


def build_per_method_collapse(frame: pd.DataFrame) -> pd.DataFrame:
    successful = frame[
        (frame['status'] == 'success')
        & frame['protocol'].isin(['standard', 'full_spectrum'])
        & ~frame['method'].isin(PANEL_EXCLUDED_METHODS)
    ].copy()
    successful['family'] = successful['method'].map(METHOD_FAMILY)

    index_cols = ['checkpoint_name', 'checkpoint_path', 'dataset', 'network', 'method', 'postprocessor', 'family']
    averaged = successful.groupby(index_cols + ['protocol'], as_index=False, observed=True)[METRICS].mean()
    pivot = averaged.pivot(index=index_cols, columns='protocol', values=METRICS)

    collapse = pivot.reset_index()
    collapse.columns = [
        column[0] if column[1] == '' else f'{"std" if column[1] == "standard" else "fs"}_{column[0]}'
        for column in collapse.columns.to_flat_index()
    ]
    for metric in METRICS:
        prefix = metric.split('_')[0]
        collapse[f'{prefix}_drop'] = collapse[f'fs_{metric}'] - collapse[f'std_{metric}']

    collapse['family'] = pd.Categorical(collapse['family'], categories=FAMILY_ORDER, ordered=True)
    collapse['method'] = pd.Categorical(collapse['method'], categories=METHOD_ORDER, ordered=True)
    return collapse.sort_values(['checkpoint_name', 'family', 'method']).reset_index(drop=True)


def aggregate_collapse(per_method: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary = per_method.groupby(group_cols, as_index=False, observed=True).agg(
        n_checkpoints=('checkpoint_path', 'nunique'),
        mean_std_near_auroc=('std_near_auroc', 'mean'),
        mean_fs_near_auroc=('fs_near_auroc', 'mean'),
        mean_near_drop=('near_drop', 'mean'),
        std_near_drop=('near_drop', 'std'),
        mean_std_far_auroc=('std_far_auroc', 'mean'),
        mean_fs_far_auroc=('fs_far_auroc', 'mean'),
        mean_far_drop=('far_drop', 'mean'),
        std_far_drop=('far_drop', 'std'),
        mean_std_overall_auroc=('std_overall_auroc', 'mean'),
        mean_fs_overall_auroc=('fs_overall_auroc', 'mean'),
        mean_overall_drop=('overall_drop', 'mean'),
        std_overall_drop=('overall_drop', 'std'),
    )
    for column in ['std_near_drop', 'std_far_drop', 'std_overall_drop']:
        summary[column] = summary[column].fillna(0.0)
    for column, order in [('family', FAMILY_ORDER), ('method', METHOD_ORDER)]:
        if column in group_cols:
            summary[column] = pd.Categorical(summary[column], categories=order, ordered=True)
    return summary.sort_values(group_cols).reset_index(drop=True)


def summary_payload(method_summary: pd.DataFrame, family_summary: pd.DataFrame) -> dict[str, object]:
    worst = method_summary.sort_values('mean_overall_drop').iloc[0]
    best = method_summary.sort_values('mean_fs_overall_auroc', ascending=False).iloc[0]
    return {
        'n_methods': method_summary['method'].nunique(),
        'n_families': family_summary['family'].nunique(),
        'largest_overall_drop_method': str(worst['method']),
        'largest_overall_drop': worst['mean_overall_drop'],
        'best_full_spectrum_method': str(best['method']),
        'best_full_spectrum_overall_auroc': best['mean_fs_overall_auroc'],
    }


def plot_standard_vs_full_spectrum(per_method: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = [FAMILY_COLORS.get(str(family), '#777777') for family in per_method['family']]
    ax.scatter(per_method['std_overall_auroc'], per_method['fs_overall_auroc'], c=colors, alpha=0.8)
    low = float(min(per_method['std_overall_auroc'].min(), per_method['fs_overall_auroc'].min()))
    high = float(max(per_method['std_overall_auroc'].max(), per_method['fs_overall_auroc'].max()))
    ax.plot([low, high], [low, high], color='#555555', linestyle='--', linewidth=1)
    ax.set_xlabel('Standard overall AUROC')
    ax.set_ylabel('Full-spectrum overall AUROC')
    ax.set_title('Standard vs full-spectrum performance')
    fig.tight_layout()
    fig.savefig(figures_dir / 'standard_vs_fs_scatter.png', dpi=180)
    plt.close(fig)


def plot_drop_bars(summary: pd.DataFrame, label_col: str, output_path: Path) -> None:
    ordered = summary.sort_values('mean_overall_drop')
    labels = ordered[label_col].tolist()
    colors = [FAMILY_COLORS.get(str(family), '#777777') for family in ordered['family']] if 'family' in ordered.columns else '#777777'
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 5))
    ax.bar(np.arange(len(labels)), ordered['mean_overall_drop'], color=colors)
    ax.axhline(0, color='#444444', linewidth=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Full-spectrum minus standard overall AUROC')
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_outputs(result: dict[str, pd.DataFrame], output_dir: Path) -> None:
    plot_dir = figures_dir(output_dir)
    write_artifacts(output_dir, result, summary=summary_payload(result['method_summary'], result['family_summary']), csv=True)
    plot_standard_vs_full_spectrum(result['per_method_collapse'], plot_dir)
    plot_drop_bars(result['method_summary'], 'method', plot_dir / 'per_method_grouped_bars.png')
    plot_drop_bars(result['family_summary'], 'family', plot_dir / 'family_grouped_bars.png')


def build_dataset(dataset: str) -> dict[str, pd.DataFrame]:
    raw = pd.read_parquet(INPUT_TABLE)
    per_method = build_per_method_collapse(raw[raw['dataset'] == dataset].copy())
    method_summary = aggregate_collapse(per_method, ['family', 'method'])
    family_summary = aggregate_collapse(per_method, ['family'])
    return {
        'per_method_collapse': per_method,
        'method_summary': method_summary,
        'family_summary': family_summary,
    }


def run_dataset(dataset: str, output_dir: Path) -> None:
    write_outputs(build_dataset(dataset), output_dir)


def main() -> None:
    for dataset in FULL_SPECTRUM_DATASETS:
        output_dir = OUTPUT_ROOT / dataset
        sentinel_path = output_dir / 'summary.json'
        if sentinel_path.exists() and not REBUILD_ANALYSIS_OUTPUTS:
            print(f'Skipping because {sentinel_path} already exists.')
            continue
        run_dataset(dataset, output_dir)
        print(f'Wrote full-spectrum collapse analysis to {output_dir}')


if __name__ == '__main__':
    main()
