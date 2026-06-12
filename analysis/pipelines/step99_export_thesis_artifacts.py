"""Step 99: export the thesis-facing artifact bundle."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from analysis.core.benchmarks import RESULTS_DIR
from analysis.core.methods.registry import METHODS, method_label
from analysis.core.scoring import compare_scores
from analysis.core.table_store import write_json

OUTPUT_DIR = RESULTS_DIR / 'thesis_artifacts'
RANKING_ROOT = RESULTS_DIR / 'ranking_stability'
FS_COLLAPSE_ROOT = RESULTS_DIR / 'fs_collapse'
SCORE_OVERLAP_ROOT = RESULTS_DIR / 'score_overlap'
TSS_ROOT = RESULTS_DIR / 'tss_temperature_sensitivity'
LOGIT_GEOMETRY_ROOT = RESULTS_DIR / 'logit_score_geometry'

PROTOCOLS = {'cifar100': ['standard'], 'imagenet200': ['standard', 'full_spectrum'], 'imagenet': ['standard', 'full_spectrum']}
DATASET_LABELS = {'cifar100': 'CIFAR-100', 'imagenet200': 'ImageNet-200', 'imagenet': 'ImageNet-1K'}
PROTOCOL_LABELS = {'standard': 'Standard', 'full_spectrum': 'Full-spectrum'}

FINAL_METHODS = ['tss', 'rtss', 'lsm', 'lsmpp', 'rlsm', 'rlsmpp', 'rknnpp']
CORE_BASELINES = ['msp', 'ebo', 'gen', 'gradnorm', 'knn', 'mds', 'mdspp', 'rmdspp', 'ash', 'react', 'scale', 'vim', 'rmds']
BOUNDARY_DATASETS = [('imagenet200', 'ImageNet-200', 'in200'), ('imagenet', 'ImageNet-1K', 'in1K')]
FINALIST_STAGE_PAIRS = [('knn_to_rknnpp', 'knn', 'rknnpp'), ('lsm_to_lsmpp', 'lsm', 'lsmpp'), ('lsmpp_to_rlsm', 'lsmpp', 'rlsm'), ('rlsm_to_rlsmpp', 'rlsm', 'rlsmpp'), ('tss_to_rtss', 'tss', 'rtss')]
METHOD_ALIASES = {method: method for method in METHODS}
METHOD_ALIASES.update({spec['label']: method for method, spec in METHODS.items()})
METHOD_ALIASES.update({method_label(method): method for method in METHODS})
METHOD_ALIASES.update({
    'rTSS': 'rtss',
    'RTSS': 'rtss',
    'RkNN++': 'rknnpp',
    'RKNN++': 'rknnpp',
    'rLSM': 'rlsm',
    'RLSM': 'rlsm',
    'rLSM++': 'rlsmpp',
    'RLSM++': 'rlsmpp',
})
BOUNDARY_LABELS = {'rTSS': 'RTSS', 'RkNN++': 'RKNN++', 'rLSM': 'RLSM', 'rLSM++': 'RLSM++'}


def boundary_label(value: object) -> str:
    label = str(value)
    return BOUNDARY_LABELS.get(label, label)


def dataset_label(dataset: object) -> str:
    return DATASET_LABELS.get(str(dataset), str(dataset))


def protocol_label(protocol: object) -> str:
    return PROTOCOL_LABELS.get(str(protocol), str(protocol))


def metric_mean(group: pd.DataFrame, column: str) -> float:
    return float(group[column].mean()) if column in group.columns else float('nan')


def performance_row(dataset: object, protocol: object, method: str, family: object, group: pd.DataFrame | None, source_priority: int, t2: float = float('nan')) -> dict:
    return {
        'dataset': dataset,
        'Dataset': dataset_label(dataset),
        'protocol': protocol,
        'Protocol': protocol_label(protocol),
        'canonical_method': method,
        'Method': method_label(method),
        'Family': family,
        'N': int(group['checkpoint_path'].nunique()) if group is not None and 'checkpoint_path' in group.columns else int(group['n'].iloc[0]),
        'Near AUROC': metric_mean(group, 'near_auroc'),
        'Far AUROC': metric_mean(group, 'far_auroc'),
        'Overall AUROC': metric_mean(group, 'overall_auroc'),
        'FPR95': metric_mean(group, 'overall_fpr95'),
        'ID/CSID Acc': metric_mean(group, 'id_or_csid_acc'),
        'T2': float(t2),
        'source_priority': int(source_priority),
    }


def raw_performance_rows() -> pd.DataFrame:
    raw = pd.read_parquet(RANKING_ROOT / 'raw_metrics.parquet')
    raw = raw[(raw['status'] == 'success') & raw['dataset'].isin(PROTOCOLS)].copy()
    raw = raw[raw.apply(lambda row: row['protocol'] in PROTOCOLS[row['dataset']], axis=1)]
    raw['canonical_method'] = raw['method'].map(METHOD_ALIASES)
    raw = raw[raw['canonical_method'].notna()].copy()
    rows = [
        performance_row(dataset, protocol, canonical_method, family, group, source_priority=0)
        for (dataset, protocol, canonical_method, family), group in raw.groupby(['dataset', 'protocol', 'canonical_method', 'family'], sort=False, dropna=False)
    ]
    return pd.DataFrame.from_records(rows)


def selected_tss_temperatures() -> pd.DataFrame:
    validation = pd.read_parquet(TSS_ROOT / 'id_validation_summary.parquet')
    selected = validation.loc[validation.groupby(['dataset', 'method'])['id_stability_score'].idxmin()].copy()
    return selected[['dataset', 'method', 't2', 'id_stability_score', 'val_z_abs_mean', 'val_z_abs95', 'n']]


def selected_tss_performance_rows() -> pd.DataFrame:
    selected = selected_tss_temperatures()
    performance = pd.read_parquet(TSS_ROOT / 'summary.parquet').merge(selected[['dataset', 'method', 't2']], on=['dataset', 'method', 't2'])
    performance = performance[performance['dataset'].isin(PROTOCOLS)].copy()
    performance = performance[performance.apply(lambda row: row['protocol'] in PROTOCOLS[row['dataset']], axis=1)]
    accuracy = pd.read_parquet(RANKING_ROOT / 'raw_metrics.parquet')
    accuracy = accuracy[(accuracy['status'] == 'success') & accuracy['dataset'].isin(PROTOCOLS)].copy()
    accuracy = accuracy[accuracy.apply(lambda row: row['protocol'] in PROTOCOLS[row['dataset']], axis=1)]
    accuracy = accuracy.groupby(['dataset', 'protocol'], as_index=False)['id_or_csid_acc'].mean()
    rows = []
    for group_keys, group in performance.groupby(['dataset', 'protocol', 'method', 't2'], sort=False):
        dataset, protocol, method_label_value, t2 = group_keys
        rows.append(performance_row(dataset, protocol, METHOD_ALIASES[method_label_value], 'output', group, source_priority=1, t2=t2))
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return frame
    frame = frame.merge(accuracy, on=['dataset', 'protocol'], how='left')
    frame['ID/CSID Acc'] = frame['ID/CSID Acc'].fillna(frame['id_or_csid_acc'])
    return frame.drop(columns=['id_or_csid_acc'])


def master_table() -> pd.DataFrame:
    frame = pd.concat([raw_performance_rows(), selected_tss_performance_rows()], ignore_index=True)
    frame = frame.sort_values(['Dataset', 'Protocol', 'canonical_method', 'source_priority'])
    frame = frame.drop_duplicates(['Dataset', 'Protocol', 'canonical_method'], keep='last')
    frame['Near Rank'] = frame.groupby(['Dataset', 'Protocol'])['Near AUROC'].rank(method='min', ascending=False).astype(int)
    frame['Overall Rank'] = frame.groupby(['Dataset', 'Protocol'])['Overall AUROC'].rank(method='min', ascending=False).astype(int)
    columns = ['canonical_method', 'Dataset', 'Protocol', 'Method', 'Family', 'N', 'Near AUROC', 'Far AUROC', 'Overall AUROC', 'FPR95', 'ID/CSID Acc', 'Near Rank', 'Overall Rank', 'T2']
    return frame[columns].sort_values(['Dataset', 'Protocol', 'Near Rank', 'Method']).reset_index(drop=True)


def finalist_stage_summary(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, baseline, variant in FINALIST_STAGE_PAIRS:
        paired = master[master['canonical_method'] == baseline].merge(master[master['canonical_method'] == variant], on=['Dataset', 'Protocol'], suffixes=('_baseline', '_variant'))
        paired['delta_near_auroc'] = paired['Near AUROC_variant'] - paired['Near AUROC_baseline']
        paired['delta_far_auroc'] = paired['Far AUROC_variant'] - paired['Far AUROC_baseline']
        paired['delta_overall_auroc'] = paired['Overall AUROC_variant'] - paired['Overall AUROC_baseline']
        for protocol, group in paired.groupby('Protocol', sort=False):
            rows.append({
                'stage': stage,
                'baseline': method_label(baseline),
                'variant': method_label(variant),
                'protocol': {'Standard': 'standard', 'Full-spectrum': 'full_spectrum'}[protocol],
                'n_settings': int(len(group)),
                'mean_delta_near_auroc': float(group['delta_near_auroc'].mean()),
                'mean_delta_far_auroc': float(group['delta_far_auroc'].mean()),
                'mean_delta_overall_auroc': float(group['delta_overall_auroc'].mean()),
                'mean_baseline_overall_auroc': float(group['Overall AUROC_baseline'].mean()),
                'mean_variant_overall_auroc': float(group['Overall AUROC_variant'].mean()),
                'mean_baseline_near_auroc': float(group['Near AUROC_baseline'].mean()),
                'mean_variant_near_auroc': float(group['Near AUROC_variant'].mean()),
                'mean_baseline_far_auroc': float(group['Far AUROC_baseline'].mean()),
                'mean_variant_far_auroc': float(group['Far AUROC_variant'].mean()),
            })
    return pd.DataFrame.from_records(rows)


def round_numbers(frame: pd.DataFrame) -> pd.DataFrame:
    rounded = frame.copy()
    for column in rounded.select_dtypes(include='number').columns:
        rounded[column] = rounded[column].round(2)
    return rounded


def write_table(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    rounded = round_numbers(frame.drop(columns=['canonical_method'], errors='ignore'))
    rounded.to_csv(output_dir / f'{name}.csv', index=False)
    rounded.to_parquet(output_dir / f'{name}.parquet', index=False)


def write_tss_selection(output_dir: Path) -> None:
    selected = selected_tss_temperatures().rename(columns={
        'dataset': 'Dataset Key',
        'method': 'Method',
        't2': 'Selected T2',
        'id_stability_score': 'ID Stability Score',
        'val_z_abs_mean': 'ID Val |z| Mean',
        'val_z_abs95': 'ID Val |z| 95th',
        'n': 'N',
    })
    selected['Method'] = selected['Method'].map(lambda value: method_label(METHOD_ALIASES[value]))
    selected['Dataset'] = selected['Dataset Key'].map(dataset_label)
    write_table(selected[['Dataset', 'Method', 'Selected T2', 'ID Stability Score', 'ID Val |z| Mean', 'ID Val |z| 95th', 'N']], output_dir, 'tss_id_temperature_selection')


def tss_class_baseline_summary() -> pd.DataFrame:
    rows = []
    for dataset in PROTOCOLS:
        path = LOGIT_GEOMETRY_ROOT / dataset / 'tss_class_baseline_summary.parquet'
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        rows.append({
            'Dataset': dataset_label(dataset),
            'N': int(len(frame)),
            'Classes': int(frame['n_classes'].max()),
            'Correct Train': float(frame['total_correct_train'].mean()),
            'Across-Class Mean SD': float(frame['class_mean_std'].mean()),
            'Across-Class Mean Range': float(frame['class_mean_range'].mean()),
            'Within-Class SD': float(frame['class_std_mean'].mean()),
        })
    return pd.DataFrame.from_records(rows)


def copy_png(source: Path, destination: Path) -> None:
    image = Image.open(source)
    if image.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', image.size, 'white')
        background.paste(image, mask=image.getchannel('A'))
        image = background
    image.convert('RGB').save(destination, optimize=True)


def copy_thesis_figures(output_dir: Path) -> int:
    figure_dir = output_dir / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_pairs = [(FS_COLLAPSE_ROOT / 'imagenet200' / 'figures' / 'standard_vs_fs_scatter.png', figure_dir / 'step03_imagenet200_standard_vs_fs_scatter.png'), (SCORE_OVERLAP_ROOT / 'imagenet200' / 'figures' / 'csid_vs_nearood_overlap.png', figure_dir / 'step02_imagenet200_csid_vs_nearood_overlap.png')]
    copied = 0
    for source, destination in figure_pairs:
        if source.exists():
            copy_png(source, destination)
            copied += 1
    return copied


def copy_supporting_tables(output_dir: Path, master: pd.DataFrame) -> None:
    support_dir = output_dir / 'supporting_tables'
    support_dir.mkdir(parents=True, exist_ok=True)
    finalist_stage_summary(master).to_csv(support_dir / 'finalist_stage_summary.csv', index=False)
    tss_baseline = tss_class_baseline_summary()
    if not tss_baseline.empty:
        tss_baseline.to_csv(support_dir / 'tss_class_baseline_summary.csv', index=False)
        tss_baseline.to_parquet(support_dir / 'tss_class_baseline_summary.parquet', index=False)
    family_summary = pd.read_parquet(FS_COLLAPSE_ROOT / 'imagenet200' / 'family_summary.parquet')
    family_summary.to_csv(support_dir / 'fs_collapse_imagenet200_family_summary.csv', index=False)
    family_summary.to_parquet(support_dir / 'fs_collapse_imagenet200_family_summary.parquet', index=False)


def cs_near_auroc_by_method(dataset_key: str) -> dict[str, float]:
    """Per-method cs-ID versus near-OOD AUROC under the full-spectrum protocol, averaged over
    checkpoints, computed from the step 02 per-sample detector scores. cs-ID is the accepted
    side, so the value is the complement of the OpenOOD OOD-positive AUROC; below 50 marks the
    inversion where the detector scores cs-ID as more OOD than near-OOD. This metric is not in
    the master performance table (which only carries near/far/overall AUROC against the accepted
    set), so it must be recomputed from the grouped per-sample scores."""
    parts_dir = SCORE_OVERLAP_ROOT / dataset_key / 'per_sample_scores_parts'
    per_method_scores: dict[str, list[float]] = {}
    for path in sorted(parts_dir.glob('*.parquet')):
        frame = pd.read_parquet(path, columns=['score', 'protocol', 'method', 'checkpoint_name', 'sample_group'])
        frame = frame[(frame['protocol'] == 'full_spectrum') & frame['sample_group'].isin(['cs_id', 'near_ood'])]
        if frame.empty:
            continue
        for (method, _checkpoint), group in frame.groupby(['method', 'checkpoint_name'], sort=False):
            cs_id = group.loc[group['sample_group'] == 'cs_id', 'score'].to_numpy()
            near_ood = group.loc[group['sample_group'] == 'near_ood', 'score'].to_numpy()
            if cs_id.size and near_ood.size:
                auroc = 100.0 - compare_scores(cs_id, near_ood, include_openood_metrics=True)['auroc']
                per_method_scores.setdefault(boundary_label(method), []).append(auroc)
    return {method: sum(values) / len(values) for method, values in per_method_scores.items()}


def write_boundary_inversion_panel(output_dir: Path) -> None:
    panel_methods = [
        boundary_label(method_label(method))
        for method in METHODS
        if method in set(FINAL_METHODS + CORE_BASELINES)
    ]
    rows = pd.DataFrame({'method': panel_methods})
    for dataset_key, _dataset_name, prefix in BOUNDARY_DATASETS:
        auroc_by_method = cs_near_auroc_by_method(dataset_key)
        performance = pd.DataFrame({
            'method': list(auroc_by_method),
            f'{prefix}_cs_near_auroc': list(auroc_by_method.values()),
        })

        threshold = pd.read_parquet(RESULTS_DIR / 'threshold_transfer' / dataset_key / 'method_threshold_summary.parquet')
        threshold = threshold[
            (threshold['protocol'] == 'full_spectrum')
            & (threshold['target_tpr'].round(2) == 0.95)
        ].copy()
        threshold['method'] = threshold['method'].map(boundary_label)
        threshold[f'{prefix}_nearood_accepted'] = 1.0 - threshold['mean_tnr_near_ood']
        threshold = threshold.rename(columns={'mean_fpr_cs_id': f'{prefix}_csid_rejected'})
        threshold = threshold[['method', f'{prefix}_csid_rejected', f'{prefix}_nearood_accepted']]

        rows = rows.merge(performance, on='method', how='left').merge(threshold, on='method', how='left')

    for column in rows.columns:
        if column == 'method':
            continue
        decimals = 2 if column.endswith('auroc') else 3
        rows[column] = rows[column].round(decimals)
    rows.to_csv(output_dir / 'boundary_inversion_panel.csv', index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master = master_table()
    write_table(master, OUTPUT_DIR, 'master_performance_table')
    write_table(master[master['canonical_method'].isin(FINAL_METHODS)], OUTPUT_DIR, 'final_method_table')
    write_table(master[master['canonical_method'].isin(CORE_BASELINES)], OUTPUT_DIR, 'core_baseline_table')
    write_tss_selection(OUTPUT_DIR)
    copy_supporting_tables(OUTPUT_DIR, master)
    write_boundary_inversion_panel(OUTPUT_DIR)
    figures_copied = copy_thesis_figures(OUTPUT_DIR)
    write_json({'output_dir': str(OUTPUT_DIR), 'master_rows': len(master), 'figures_copied': figures_copied}, OUTPUT_DIR / 'summary.json')
    print(f'Wrote thesis artifacts to {OUTPUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
