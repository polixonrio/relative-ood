"""Step 10: explain ranking changes by joining overlap, threshold-transfer, and near-OOD-hardness drivers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.core.table_store import write_artifacts
from analysis.core.benchmarks import DATASET_NETWORKS, REBUILD_ANALYSIS_OUTPUTS, RESULTS_DIR
from analysis.core.methods.registry import METHOD_FAMILY, METHOD_ORDER

RAW_TABLE = RESULTS_DIR / 'ranking_stability' / 'raw_metrics.parquet'
SCORE_OVERLAP_ROOT = RESULTS_DIR / 'score_overlap'
FS_COLLAPSE_ROOT = RESULTS_DIR / 'fs_collapse'
THRESHOLD_TRANSFER_ROOT = RESULTS_DIR / 'threshold_transfer'
NEAR_OOD_HARDNESS_ROOT = RESULTS_DIR / 'near_ood_hardness'
OUTPUT_DIR = RESULTS_DIR / 'ranking_change_explanation'

RAW_REQUIRED = {'checkpoint_name', 'dataset', 'network', 'protocol', 'method', 'family', 'status', 'near_auroc', 'far_auroc', 'overall_auroc'}
DRIVER_COLUMNS = ['score_overlap_shift', 'csid_near_overlap', 'threshold_transfer_fpr_csid', 'threshold_transfer_gap_near', 'near_ood_feature_covariance_alignment', 'near_ood_knn_overlap']
EXPLANATION_LABELS = {
    'score_overlap_shift': 'score-distribution overlap increased',
    'csid_near_overlap': 'CSID and near-OOD scores overlap',
    'threshold_transfer_fpr_csid': 'clean-ID threshold rejects shifted ID',
    'threshold_transfer_gap_near': 'clean-ID threshold accepts near-OOD',
    'near_ood_feature_covariance_alignment': 'near-OOD feature covariance is ID-like',
    'near_ood_knn_overlap': 'near-OOD overlaps ID neighborhoods',
}


def dataset_dirs(root: Path) -> list[Path]:
    return [root / dataset for dataset in sorted(DATASET_NETWORKS)]


def existing_table(path: Path, table_name: str | None = None) -> Path | None:
    path = path / table_name if table_name and path.is_dir() else path
    return path if path.exists() else None


def load_tables(paths: list[Path], table_name: str | None = None) -> pd.DataFrame:
    frames = []
    for path in paths:
        table_path = existing_table(path, table_name)
        if table_path is None:
            continue
        frame = pd.read_parquet(table_path)
        frame['source_path'] = str(table_path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def method_sort_key(values: pd.Series) -> pd.Series:
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    return values.map(lambda method: order.get(method, len(order)))


def build_rank_change(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    missing = RAW_REQUIRED.difference(raw_metrics.columns)
    if missing:
        raise ValueError(f'Raw metrics missing required columns: {sorted(missing)}')

    successful = raw_metrics[
        (raw_metrics['status'] == 'success')
        & raw_metrics['protocol'].isin(['standard', 'full_spectrum'])
    ].copy()
    if successful.empty:
        raise RuntimeError('No successful standard/full_spectrum rows found.')

    group_cols = ['dataset', 'network', 'protocol', 'method', 'family']
    metric_cols = ['near_auroc', 'far_auroc', 'overall_auroc']
    if 'postprocessor' in successful.columns:
        group_cols.insert(4, 'postprocessor')
    else:
        successful['postprocessor'] = successful['method']
        group_cols.insert(4, 'postprocessor')

    summary = successful.groupby(group_cols, observed=True)[metric_cols].mean().reset_index()
    summary['rank'] = (
        summary.groupby(['dataset', 'network', 'protocol'], observed=True)['overall_auroc']
        .rank(method='min', ascending=False)
        .astype(int)
    )

    index_cols = ['dataset', 'network', 'method', 'postprocessor', 'family']
    pivot = summary.pivot_table(
        index=index_cols,
        columns='protocol',
        values=metric_cols + ['rank'],
        observed=True,
    )
    required_pairs = [('overall_auroc', 'standard'), ('overall_auroc', 'full_spectrum')]
    missing_pairs = [pair for pair in required_pairs if pair not in pivot.columns]
    if missing_pairs:
        available = sorted(summary['protocol'].unique().tolist())
        raise RuntimeError(f'Raw metrics must contain both standard and full_spectrum protocols; found {available}.')
    pivot = pivot.dropna(subset=required_pairs)
    if pivot.empty:
        raise RuntimeError('No methods had both standard and full_spectrum rows.')

    result = pivot.reset_index()
    result.columns = [
        col[0] if col[1] == '' else f'{col[1]}_{col[0]}'
        for col in result.columns.to_flat_index()
    ]
    for metric in metric_cols:
        result[f'{metric}_delta'] = result[f'full_spectrum_{metric}'] - result[f'standard_{metric}']
    result['rank_delta'] = result['full_spectrum_rank'] - result['standard_rank']
    result['rank_direction'] = np.select(
        [result['rank_delta'] > 0, result['rank_delta'] < 0],
        ['worse', 'better'],
        default='unchanged',
    )
    return result.sort_values(['dataset', 'network', 'full_spectrum_rank', 'method']).reset_index(drop=True)


def build_collapse_drivers(collapse: pd.DataFrame) -> pd.DataFrame:
    if collapse.empty:
        return pd.DataFrame()
    required = {'dataset', 'network', 'method', 'near_drop', 'far_drop', 'overall_drop'}
    if not required.issubset(collapse.columns):
        return pd.DataFrame()
    return (
        collapse.groupby(['dataset', 'network', 'method'], observed=True)
        .agg(
            fs_near_drop=('near_drop', 'mean'),
            fs_far_drop=('far_drop', 'mean'),
            fs_overall_drop=('overall_drop', 'mean'),
        )
        .reset_index()
    )


def build_overlap_drivers(overlap: pd.DataFrame) -> pd.DataFrame:
    if overlap.empty:
        return pd.DataFrame()
    required = {'dataset', 'network', 'protocol', 'method', 'comparison', 'overlap_coeff'}
    if not required.issubset(overlap.columns):
        return pd.DataFrame()

    standard_pairs = {'clean_id__vs__near_ood', 'clean_id__vs__far_ood'}
    fs_pairs = {'clean_id__vs__cs_id', 'cs_id__vs__near_ood', 'cs_id__vs__far_ood'}
    rows = []
    for keys, group in overlap.groupby(['dataset', 'network', 'method'], observed=True):
        standard = group[
            (group['protocol'] == 'standard')
            & group['comparison'].isin(standard_pairs)
        ]['overlap_coeff'].mean()
        fs = group[
            (group['protocol'] == 'full_spectrum')
            & group['comparison'].isin(fs_pairs)
        ]['overlap_coeff'].mean()
        csid_near = group[
            (group['protocol'] == 'full_spectrum')
            & (group['comparison'] == 'cs_id__vs__near_ood')
        ]['overlap_coeff'].mean()
        rows.append({
            'dataset': keys[0],
            'network': keys[1],
            'method': keys[2],
            'standard_overlap': standard,
            'full_spectrum_overlap': fs,
            'score_overlap_shift': fs - standard,
            'csid_near_overlap': csid_near,
        })
    return pd.DataFrame.from_records(rows)


def build_threshold_drivers(threshold: pd.DataFrame) -> pd.DataFrame:
    if threshold.empty:
        return pd.DataFrame()
    required = {'protocol', 'method', 'target_tpr', 'mean_fpr_cs_id', 'mean_transfer_gap_clean_to_near'}
    if not required.issubset(threshold.columns):
        return pd.DataFrame()
    subset = threshold[
        (threshold['protocol'] == 'full_spectrum')
        & np.isclose(threshold['target_tpr'], 0.95)
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    group_cols = ['method']
    if 'dataset' in subset.columns:
        group_cols.insert(0, 'dataset')
    if 'network' in subset.columns:
        group_cols.insert(1 if 'dataset' in group_cols else 0, 'network')
    out = (
        subset.groupby(group_cols, observed=True)
        .agg(
            threshold_transfer_fpr_csid=('mean_fpr_cs_id', 'mean'),
            threshold_transfer_gap_near=('mean_transfer_gap_clean_to_near', 'mean'),
        )
        .reset_index()
    )
    return out


def build_hardness_drivers(hardness: pd.DataFrame) -> pd.DataFrame:
    if hardness.empty:
        return pd.DataFrame()
    alignment_column = 'feature_covariance_alignment_id_ood'
    if alignment_column not in hardness.columns and 'cka_id_ood' in hardness.columns:
        alignment_column = 'cka_id_ood'
    required = {'dataset', 'network', 'ood_type', alignment_column, 'knn_overlap_ratio'}
    if not required.issubset(hardness.columns):
        return pd.DataFrame()
    near = hardness[hardness['ood_type'] == 'near']
    if near.empty:
        return pd.DataFrame()
    return (
        near.groupby(['dataset', 'network'], observed=True)
        .agg(
            near_ood_feature_covariance_alignment=(alignment_column, 'mean'),
            near_ood_knn_overlap=('knn_overlap_ratio', 'mean'),
        )
        .reset_index()
    )


def merge_driver(base: pd.DataFrame, driver: pd.DataFrame) -> pd.DataFrame:
    if driver.empty:
        return base
    join_cols = ['method']
    if {'dataset', 'network'}.issubset(driver.columns):
        join_cols = ['dataset', 'network', 'method'] if 'method' in driver.columns else ['dataset', 'network']
    elif 'dataset' in driver.columns:
        join_cols = ['dataset', 'method'] if 'method' in driver.columns else ['dataset']
    elif 'network' in driver.columns:
        join_cols = ['network', 'method'] if 'method' in driver.columns else ['network']
    return base.merge(driver, on=join_cols, how='left')


def normalized_driver_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame(index=frame.index)
    for column in DRIVER_COLUMNS:
        if column not in frame.columns:
            scores[column] = np.nan
            continue
        values = pd.to_numeric(frame[column], errors='coerce')
        raw = values.clip(lower=0)
        max_value = raw.max(skipna=True)
        scores[column] = raw / max_value if pd.notna(max_value) and max_value > 0 else np.nan
    return scores


def choose_top_explanation(frame: pd.DataFrame) -> pd.DataFrame:
    driver_scores = normalized_driver_scores(frame)
    enriched = frame.copy()
    enriched['top_driver_score'] = driver_scores.max(axis=1, skipna=True)
    top_driver = driver_scores.fillna(-np.inf).idxmax(axis=1)
    top_driver = top_driver.where(enriched['top_driver_score'].notna(), 'none')
    enriched['top_driver'] = top_driver
    enriched['top_explanation'] = enriched['top_driver'].map(EXPLANATION_LABELS).fillna('no driver table available')
    improved = enriched['rank_delta'] < 0
    worsened = enriched['rank_delta'] > 0
    unchanged = enriched['rank_delta'] == 0
    enriched.loc[improved, 'explanation_notes'] = 'rank improved; driver indicates the strongest remaining full-spectrum stressor'
    enriched.loc[worsened, 'explanation_notes'] = 'rank worsened; driver is a heuristic explanation, not a causal estimate'
    enriched.loc[unchanged, 'explanation_notes'] = 'rank unchanged; driver indicates exposure to full-spectrum stress'
    for column in DRIVER_COLUMNS:
        enriched[f'{column}_driver_score'] = driver_scores[column]
    return enriched


def build_driver_summary(drivers: pd.DataFrame) -> pd.DataFrame:
    if drivers.empty:
        return pd.DataFrame()
    return (
        drivers.groupby(['dataset', 'network', 'top_driver'], observed=True)
        .agg(
            n_methods=('method', 'nunique'),
            mean_rank_delta=('rank_delta', 'mean'),
            mean_auroc_delta=('overall_auroc_delta', 'mean'),
            mean_driver_score=('top_driver_score', 'mean'),
        )
        .reset_index()
        .sort_values(['dataset', 'network', 'n_methods', 'top_driver'], ascending=[True, True, False, True])
    )


def build_summary_json(drivers: pd.DataFrame, driver_summary: pd.DataFrame) -> dict:
    payload: dict = {
        'n_rows': len(drivers),
        'n_datasets': drivers['dataset'].nunique() if 'dataset' in drivers else 0,
        'available_driver_columns': [column for column in DRIVER_COLUMNS if column in drivers.columns and drivers[column].notna().any()],
        'notes': 'Top explanations are heuristic joins over existing analyses; they are not causal estimates.',
    }
    if not drivers.empty:
        largest_worse = drivers.sort_values('rank_delta', ascending=False).head(5)
        largest_better = drivers.sort_values('rank_delta', ascending=True).head(5)
        payload['largest_rank_worsening'] = largest_worse[
            ['dataset', 'network', 'method', 'rank_delta', 'overall_auroc_delta', 'top_driver', 'top_explanation']
        ].to_dict(orient='records')
        payload['largest_rank_improvement'] = largest_better[
            ['dataset', 'network', 'method', 'rank_delta', 'overall_auroc_delta', 'top_driver', 'top_explanation']
        ].to_dict(orient='records')
    if not driver_summary.empty:
        payload['driver_counts'] = driver_summary.to_dict(orient='records')
    return payload


def write_outputs(drivers: pd.DataFrame, driver_summary: pd.DataFrame, output_dir: Path) -> None:
    write_artifacts(output_dir, {
        'ranking_change_drivers': drivers,
        'ranking_change_driver_summary': driver_summary,
    }, summary=build_summary_json(drivers, driver_summary), csv={'ranking_change_drivers'})


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_tables([RAW_TABLE])
    if raw.empty:
        raise FileNotFoundError(f'No raw metric table found at {RAW_TABLE}. Run step 01 first.')

    drivers = build_rank_change(raw)
    drivers = merge_driver(drivers, build_collapse_drivers(load_tables(dataset_dirs(FS_COLLAPSE_ROOT), 'per_method_collapse.parquet')))
    drivers = merge_driver(drivers, build_overlap_drivers(load_tables(dataset_dirs(SCORE_OVERLAP_ROOT), 'pairwise_overlap_metrics.parquet')))
    drivers = merge_driver(drivers, build_threshold_drivers(load_tables(dataset_dirs(THRESHOLD_TRANSFER_ROOT), 'method_threshold_summary.parquet')))
    drivers = merge_driver(drivers, build_hardness_drivers(load_tables(dataset_dirs(NEAR_OOD_HARDNESS_ROOT), 'dataset_hardness_metrics.parquet')))
    drivers = choose_top_explanation(drivers)
    drivers['family'] = drivers['family'].fillna(drivers['method'].map(METHOD_FAMILY))
    drivers['method_order'] = method_sort_key(drivers['method'])
    drivers = drivers.sort_values(['dataset', 'network', 'full_spectrum_rank', 'method_order']).drop(columns=['method_order']).reset_index(drop=True)
    return drivers, build_driver_summary(drivers)


def main() -> None:
    primary_output = OUTPUT_DIR / 'ranking_change_drivers.parquet'
    if primary_output.exists() and not REBUILD_ANALYSIS_OUTPUTS:
        print(f'Skipping because {primary_output} already exists.')
        return

    drivers, driver_summary = run_analysis()
    write_outputs(drivers, driver_summary, OUTPUT_DIR)
    print(f'Wrote ranking-change explanation to {OUTPUT_DIR}')
    print('Generated:')
    print(' - ranking_change_drivers.parquet')
    print(' - ranking_change_drivers.csv')
    print(' - ranking_change_driver_summary.parquet')
    print(' - summary.json')


if __name__ == '__main__':
    main()
