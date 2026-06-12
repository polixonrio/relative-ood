"""Regenerate the thesis's generated LaTeX tables and figures.

Reads the analysis artifact CSVs under results/thesis_artifacts/ and the per-step
parquet tables, and writes the generated tables into
draft/aauReportTemplate/sections/generated/ and figures into
draft/aauReportTemplate/AAUgraphics/thesis_artifacts/.
Run: python -m analysis.tools.update_thesis_artifacts
"""
from pathlib import Path
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image

TEMPLATE_ROOT = Path(__file__).resolve().parents[2] / 'draft' / 'aauReportTemplate'
REPO_ROOT = TEMPLATE_ROOT.parents[1]
RESULTS_DIR = REPO_ROOT / 'results'
ARTIFACT_DIR = RESULTS_DIR / 'thesis_artifacts'
SCORE_OVERLAP_ROOT = RESULTS_DIR / 'score_overlap'
CLASS_CONDITIONING_ROOT = RESULTS_DIR / 'class_conditioning_gap'
LOGIT_GEOMETRY_ROOT = RESULTS_DIR / 'logit_score_geometry'
THRESHOLD_TRANSFER_ROOT = RESULTS_DIR / 'threshold_transfer'
NEAR_OOD_HARDNESS_ROOT = RESULTS_DIR / 'near_ood_hardness'
RANKING_CHANGE_ROOT = RESULTS_DIR / 'ranking_change_explanation'
GENERATED_DIR = TEMPLATE_ROOT / 'sections' / 'generated'
FIGURE_DIR = TEMPLATE_ROOT / 'AAUgraphics' / 'thesis_artifacts'
DATASET_LABELS = {'cifar100': 'CIFAR-100', 'imagenet200': 'ImageNet-200', 'imagenet': 'ImageNet-1K'}
FAMILY_LABELS = {
    'output': 'Output',
    'distance': 'Distance',
    'feature_activation': 'Feature activation',
    'gradient': 'Gradient',
}
FAMILY_COLORS = {
    'output': '#4C78A8',
    'distance': '#F58518',
    'feature_activation': '#54A24B',
    'gradient': '#B279A2',
}
METHOD_ORDER = [
    'MSP', 'Energy', 'GEN', 'TSS', 'rTSS',
    'KNN', 'RkNN++', 'Mahalanobis', 'MDS++', 'RMDS', 'RMDS++',
    'LSM', 'LSM++', 'rLSM', 'rLSM++',
    'ReAct', 'ASH', 'SCALE', 'VIM', 'GradNorm',
]
CUSTOM_METHODS = {'TSS', 'rTSS', 'LSM', 'LSM++', 'rLSM', 'rLSM++', 'RkNN++'}
# Stale artifact labels carry lowercase-r forms; display them with the canonical capital-R names.
DISPLAY_LABELS = {'rTSS': 'RTSS', 'RkNN++': 'RKNN++', 'rLSM': 'RLSM', 'rLSM++': 'RLSM++',
                  'Mahalanobis': 'MDS', 'VIM': 'ViM'}


def display_label(method):
    return DISPLAY_LABELS.get(str(method), str(method))


def format_number(value):
    if pd.isna(value):
        return '--'
    return f'{float(value):.2f}'


def format_small(value):
    if pd.isna(value):
        return '--'
    return f'{float(value):.3f}'


def latex_escape(value):
    text = str(value)
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def table_header(caption, label, columns, small=True, full_width=False, placement='htbp'):
    size = '\\scriptsize' if small else '\\small'
    spec = 'l' * len(columns)
    resize = (
        '\\resizebox{\\textwidth}{!}{%'
        if full_width
        else '\\resizebox{\\ifdim\\width>\\textwidth \\textwidth\\else\\width\\fi}{!}{%'
    )
    return [
        f'\\begin{{table}}[{placement}]',
        '\\centering',
        size,
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        resize,
        f'\\begin{{tabular}}{{{spec}}}',
        '\\toprule',
        ' & '.join(columns) + r' \\',
        '\\midrule',
    ]


def table_footer():
    return [
        '\\bottomrule',
        '\\end{tabular}%',
        '}',
        '\\end{table}',
        '',
    ]


def write_lines(path, lines):
    path.write_text('\n'.join(lines), encoding='utf-8')


def method_sort_key(method):
    try:
        return METHOD_ORDER.index(str(method))
    except ValueError:
        return len(METHOD_ORDER)


def setting_sort_key(row):
    dataset_order = {'CIFAR-100': 0, 'ImageNet-200': 1, 'ImageNet-1K': 2}
    protocol_order = {'Standard': 0, 'Full-spectrum': 1}
    return dataset_order.get(row['Dataset'], 99), protocol_order.get(row['Protocol'], 99)


def artifact_dir():
    return ARTIFACT_DIR


# Colour-blind-safe palette (blue = better, orange = worse) instead of red/green.
TOP1_COLOR = 'blue!38'   # best value in the column
TOP_COLOR = 'blue!16'    # remaining top-three values
BOTTOM_COLOR = 'orange!30'  # bottom-three values


def auroc_cell(value, color=None, bold=False):
    if pd.isna(value):
        return '--'
    text = format_number(value)
    if bold:
        text = f'\\textbf{{{text}}}'
    return f'\\cellcolor{{{color}}}{text}' if color else text


def rank_color_map(frame, dataset, protocol, metric, top_n=3, bottom_n=3):
    """Map each method to (cell colour, is_best). The single best value in the
    column is shaded darker and bold so the top method stands out; the rest of
    the top three are light blue and the bottom three are orange."""
    subset = frame[(frame['Dataset'] == dataset) & (frame['Protocol'] == protocol)]
    subset = subset.dropna(subset=[metric]).sort_values(metric, ascending=False)
    colors = {}
    for rank, method in enumerate(subset.head(top_n)['Method']):
        colors[method] = (TOP1_COLOR, True) if rank == 0 else (TOP_COLOR, False)
    for method in subset.tail(bottom_n)['Method']:
        colors.setdefault(method, (BOTTOM_COLOR, False))
    return colors


def read_combined_leaderboard():
    baselines = pd.read_csv(artifact_dir() / 'core_baseline_table.csv')
    baselines['Source'] = 'Baseline'
    finals = pd.read_csv(artifact_dir() / 'final_method_table.csv')
    finals['Source'] = 'Custom'
    frame = pd.concat([baselines, finals], ignore_index=True)
    frame = frame.drop_duplicates(subset=['Dataset', 'Protocol', 'Method'], keep='last')
    frame['setting_order'] = frame.apply(setting_sort_key, axis=1)
    frame['source_order'] = frame['Source'].map({'Custom': 0, 'Baseline': 1}).fillna(2)
    frame['method_order'] = frame['Method'].map(method_sort_key)
    return frame.sort_values(['setting_order', 'Near AUROC', 'source_order', 'method_order'], ascending=[True, False, True, True])


def write_openood_style_leaderboard():
    frame = read_combined_leaderboard()
    settings = [
        ('CIFAR-100', 'Standard', 'R-18'),
        ('ImageNet-200', 'Standard', 'R-18'),
        ('ImageNet-200', 'Full-spectrum', 'R-18'),
        ('ImageNet-1K', 'Standard', 'R-50'),
        ('ImageNet-1K', 'Full-spectrum', 'R-50'),
    ]
    metrics = [
        ('Near AUROC', r'\shortstack{near\\OOD}'),
        ('Far AUROC', r'\shortstack{far\\OOD}'),
        ('Overall AUROC', r'\shortstack{avg.\\OOD}'),
    ]
    rank_colors = {}
    for dataset, protocol, _ in settings:
        for metric, _ in metrics[:3]:
            rank_colors[(dataset, protocol, metric)] = rank_color_map(frame, dataset, protocol, metric)

    method_rows = []
    for method, subset in frame.groupby('Method', sort=False):
        by_setting = {
            (row['Dataset'], row['Protocol']): row
            for _, row in subset.iterrows()
        }
        fs_near = [
            by_setting[(dataset, protocol)]['Near AUROC']
            for dataset, protocol, _ in settings
            if protocol == 'Full-spectrum' and (dataset, protocol) in by_setting
        ]
        sort_score = float(np.nanmean(fs_near)) if fs_near else -np.inf
        source = 'Custom' if (subset['Source'] == 'Custom').any() else 'Baseline'
        family = FAMILY_LABELS.get(subset.iloc[0]['Family'], subset.iloc[0]['Family'])
        method_rows.append({
            'method': method,
            'source': source,
            'family': family,
            'sort_score': sort_score,
            'by_setting': by_setting,
        })
    method_rows.sort(key=lambda row: (row['sort_score'], row['source'] == 'Custom', -method_sort_key(row['method'])), reverse=True)
    # Proposed methods follow the order they are introduced in the methods chapter (by family);
    # baselines stay in near-OOD order.
    custom_order = {name: i for i, name in enumerate(['TSS', 'RTSS', 'LSM', 'LSM++', 'RLSM', 'RLSM++', 'RKNN++'])}
    custom_rows = sorted((row for row in method_rows if row['source'] == 'Custom'),
                         key=lambda row: custom_order.get(row['method'], len(custom_order)))
    baseline_rows = [row for row in method_rows if row['source'] == 'Baseline']
    method_rows = custom_rows + baseline_rows

    lines = [
        '% Generated by analysis/tools/update_thesis_artifacts.py',
        '\\begin{table}[htbp]',
        '\\centering',
        '\\tiny',
        '\\setlength{\\tabcolsep}{2.3pt}',
        '\\renewcommand{\\arraystretch}{0.96}',
        '\\caption{Detector leaderboard. The proposed methods are listed in the order Chapter~\\ref{ch:methods} introduces them, and the baseline panel is ordered by full-spectrum near-OOD AUROC averaged over the two ImageNet settings, following OpenOOD, which ranks its leaderboard by near-OOD AUROC \\cite{openood_leaderboard}. Within the panel, blue marks the top three AUROC values and orange the bottom three in each dataset, protocol, and metric column, and the single best value in each column is shown in bold. All classifiers use plain cross-entropy training. Classifier accuracy on the accepted ID pool is the same for every detector in a setting and is omitted (CIFAR-100 77.2, ImageNet-200 86.4 standard and 43.9 full-spectrum, ImageNet-1K 76.2 standard and 54.4 full-spectrum).}',
        '\\label{tab:openood-style-leaderboard}',
        '\\resizebox{\\textwidth}{!}{%',
        '\\begin{tabular}{l' + 'r' * (len(settings) * len(metrics)) + '}',
        '\\toprule',
        'ID dataset $\\rightarrow$ & '
        + ' & '.join([f'\\multicolumn{{{len(metrics)}}}{{c}}{{{latex_escape(dataset)}}}' for dataset, _, _ in settings])
        + r' \\',
        'classifier $\\rightarrow$ & '
        + ' & '.join([f'\\multicolumn{{{len(metrics)}}}{{c}}{{{latex_escape(classifier)}}}' for _, _, classifier in settings])
        + r' \\',
        'eval setting $\\rightarrow$ & '
        + ' & '.join([f'\\multicolumn{{{len(metrics)}}}{{c}}{{{latex_escape(protocol.lower())}}}' for _, protocol, _ in settings])
        + r' \\',
        ''.join(f'\\cmidrule(lr){{{2 + i * len(metrics)}-{1 + (i + 1) * len(metrics)}}}' for i in range(len(settings))),
        'OOD detector & ' + ' & '.join([label for _ in settings for _, label in metrics]) + r' \\',
        '\\midrule',
    ]
    total_columns = 1 + len(settings) * len(metrics)

    def emit_row(row):
        method = latex_escape(display_label(row['method']))
        values = []
        for dataset, protocol, _ in settings:
            setting_row = row['by_setting'].get((dataset, protocol))
            for metric, _ in metrics:
                if setting_row is None or pd.isna(setting_row[metric]):
                    values.append('--')
                elif metric == 'ID/CSID Acc':
                    values.append(format_number(setting_row[metric]))
                else:
                    entry = rank_colors[(dataset, protocol, metric)].get(row['method'])
                    color, is_best = entry if entry else (None, False)
                    values.append(auroc_cell(setting_row[metric], color, bold=is_best))
        expected_values = len(settings) * len(metrics)
        if len(values) != expected_values:
            raise RuntimeError(f'leaderboard row for {row["method"]} has {len(values)} cells, expected {expected_values}')
        lines.append(f'{method} & ' + ' & '.join(values) + r' \\')

    for group_index, (group_label, group_source) in enumerate([('Proposed methods', 'Custom'), ('Baselines', 'Baseline')]):
        if group_index > 0:
            lines.append('\\midrule')
        lines.append(f'\\multicolumn{{{total_columns}}}{{l}}{{\\textit{{{group_label}}}}} \\\\')
        lines.append('\\midrule')
        for row in method_rows:
            if row['source'] == group_source:
                emit_row(row)
    lines.extend([
        '\\bottomrule',
        '\\end{tabular}%',
        '}',
        '\\end{table}',
        '',
    ])
    write_lines(GENERATED_DIR / 'tab_openood_style_leaderboard.tex', lines)


def grouped_table_rows(rows, n_group):
    """Collapse the first n_group columns so a grouping value (e.g. Dataset)
    is shown once per block via multirow instead of repeating on every row.
    rows is a list of lists of formatted cell strings; a block is a maximal
    run of rows whose first n_group cells match. Blocks are separated by a
    midrule."""
    out = []
    i = 0
    total = len(rows)
    while i < total:
        j = i
        key = rows[i][:n_group]
        while j < total and rows[j][:n_group] == key:
            j += 1
        block_len = j - i
        if i > 0:
            out.append('\\midrule')
        for r in range(i, j):
            cells = list(rows[r])
            for c in range(n_group):
                if r == i:
                    cells[c] = (
                        f'\\multirow{{{block_len}}}{{*}}{{{rows[r][c]}}}'
                        if block_len > 1 else rows[r][c]
                    )
                else:
                    cells[c] = ''
            out.append(' & '.join(cells) + ' \\\\')
        i = j
    return out


def write_final_methods():
    frame = pd.read_csv(artifact_dir() / 'final_method_table.csv')
    frame = frame[['Dataset', 'Protocol', 'Method', 'Near AUROC', 'Far AUROC', 'Overall AUROC', 'FPR95']]
    frame['Method'] = frame['Method'].map(display_label)
    lines = table_header(
        'Proposed-method performance across the thesis evaluation settings. Values are AUROC and FPR95 in percent. CIFAR-100 is reported only under the standard OpenOOD protocol.',
        'tab:final-methods',
        ['Dataset', 'Protocol', 'Method', 'Near', 'Far', 'Overall', 'FPR95'],
        full_width=False,
        placement='H',
    )
    body = [
        [
            latex_escape(row.Dataset), latex_escape(row.Protocol), latex_escape(row.Method),
            format_number(row._3), format_number(row._4), format_number(row._5), format_number(row.FPR95),
        ]
        for row in frame.itertuples(index=False)
    ]
    lines.extend(grouped_table_rows(body, 2))
    lines.extend(table_footer())
    write_lines(GENERATED_DIR / 'tab_final_methods.tex', lines)


def write_core_baselines():
    frame = pd.read_csv(artifact_dir() / 'core_baseline_table.csv')
    frame = frame.sort_values(['Dataset', 'Protocol', 'Near Rank']).groupby(['Dataset', 'Protocol'], sort=False).head(5)
    frame = frame[['Dataset', 'Protocol', 'Method', 'Near AUROC', 'Far AUROC', 'Overall AUROC', 'FPR95']]
    frame['Method'] = frame['Method'].map(display_label)
    lines = table_header(
        'Top-five baselines by near-OOD AUROC for each dataset and protocol. Values are AUROC and FPR95 in percent.',
        'tab:core-baselines',
        ['Dataset', 'Protocol', 'Method', 'Near', 'Far', 'Overall', 'FPR95'],
        full_width=False,
        placement='H',
    )
    body = [
        [
            latex_escape(row.Dataset), latex_escape(row.Protocol), latex_escape(row.Method),
            format_number(row._3), format_number(row._4), format_number(row._5), format_number(row.FPR95),
        ]
        for row in frame.itertuples(index=False)
    ]
    lines.extend(grouped_table_rows(body, 2))
    lines.extend(table_footer())
    write_lines(GENERATED_DIR / 'tab_core_baselines.tex', lines)


def write_tss_temperature():
    frame = pd.read_csv(artifact_dir() / 'tss_id_temperature_selection.csv')
    frame['Method'] = frame['Method'].map(display_label)
    lines = table_header(
        'ID-only temperature selection for TSS and RTSS. A lower stability score means the held-out ID scores stay closer to their per-class training reference.',
        'tab:tss-temperature',
        ['Dataset', 'Method', '$t_2$', 'Stability'],
        small=False,
    )
    body = [
        [latex_escape(row.Dataset), latex_escape(row.Method), format_number(row._2), format_number(row._3)]
        for row in frame.itertuples(index=False)
    ]
    lines.extend(grouped_table_rows(body, 1))
    lines.extend(table_footer())
    write_lines(GENERATED_DIR / 'tab_tss_temperature.tex', lines)


def read_tss_class_baseline_summary():
    for base_dir in dict.fromkeys([artifact_dir(), ARTIFACT_DIR]):
        artifact_path = base_dir / 'supporting_tables' / 'tss_class_baseline_summary.csv'
        if artifact_path.exists():
            return pd.read_csv(artifact_path)

    roots = [LOGIT_GEOMETRY_ROOT]
    rows = []
    for dataset_key, dataset in DATASET_LABELS.items():
        source = next(
            (root / dataset_key / 'tss_class_baseline_summary.parquet' for root in roots
             if (root / dataset_key / 'tss_class_baseline_summary.parquet').exists()),
            None,
        )
        if source is None:
            continue
        frame = pd.read_parquet(source)
        if frame.empty:
            continue
        rows.append({
            'Dataset': dataset,
            'N': len(frame),
            'Classes': frame['n_classes'].max(),
            'Correct Train': frame['total_correct_train'].mean(),
            'Across-Class Mean SD': frame['class_mean_std'].mean(),
            'Across-Class Mean Range': frame['class_mean_range'].mean(),
            'Within-Class SD': frame['class_std_mean'].mean(),
        })
    return pd.DataFrame.from_records(rows)


def write_tss_class_baseline():
    frame = read_tss_class_baseline_summary()
    lines = table_header(
        'Classwise TSS training baselines. Range($\\mu_c$) is the difference between the largest and smallest per-predicted-class training means of the TSS score, and Mean $\\sigma_c$ is the average within-class standard deviation. Both are computed using only correct ID training samples.',
        'tab:tss-class-baseline',
        ['Dataset', 'Range($\\mu_c$)', 'Mean $\\sigma_c$'],
        small=False,
        placement='H',
    )
    for row in frame.itertuples(index=False):
        lines.append(
            f'{latex_escape(row.Dataset)} & {format_small(row._5)} & {format_small(row._6)} \\\\'
        )
    lines.extend(table_footer())
    write_lines(GENERATED_DIR / 'tab_tss_class_baseline.tex', lines)


def write_ablation_summary():
    frame = pd.read_csv(artifact_dir() / 'supporting_tables' / 'finalist_stage_summary.csv')
    frame = frame.rename(columns={
        'baseline': 'Baseline',
        'variant': 'Variant',
        'protocol': 'Protocol',
        'mean_variant_near_auroc': 'Near',
        'mean_variant_far_auroc': 'Far',
        'mean_variant_overall_auroc': 'Overall',
        'mean_delta_near_auroc': 'Delta Near',
        'mean_delta_far_auroc': 'Delta Far',
        'mean_delta_overall_auroc': 'Delta Overall',
    })
    frame['Protocol'] = frame['Protocol'].map({'standard': 'Standard', 'full_spectrum': 'Full-spectrum'})
    frame = frame[['Protocol', 'Baseline', 'Variant', 'Near', 'Far', 'Overall', 'Delta Near', 'Delta Far', 'Delta Overall']]
    frame['Baseline'] = frame['Baseline'].map(display_label)
    frame['Variant'] = frame['Variant'].map(display_label)
    lines = table_header(
        "Branch ablations. Near, Far, and Overall are the variant's absolute AUROC in percent, averaged over the evaluated settings, and the delta columns give its change over the baseline. Positive deltas mean the variant improves over the baseline.",
        'tab:finalist-ablation',
        ['Protocol', 'Ablation', 'Near', 'Far', 'Overall', '$\\Delta$ Near', '$\\Delta$ Far', '$\\Delta$ Overall'],
        small=False,
    )
    for row in frame.itertuples(index=False):
        pair = f'{latex_escape(row.Baseline)} $\\rightarrow$ {latex_escape(row.Variant)}'
        lines.append(
            f'{latex_escape(row.Protocol)} & {pair} & '
            f'{format_number(row.Near)} & {format_number(row.Far)} & {format_number(row.Overall)} & '
            f'{format_number(row._6)} & {format_number(row._7)} & {format_number(row._8)} \\\\'
        )
    lines.extend(table_footer())
    write_lines(GENERATED_DIR / 'tab_finalist_ablation.tex', lines)


def copy_png(source, destination):
    image = Image.open(source)
    if image.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', image.size, 'white')
        background.paste(image, mask=image.getchannel('A'))
        image = background
    else:
        image = image.convert('RGB')
    image.save(destination, optimize=True)


def copy_figures():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_names = {
        'step03_imagenet200_standard_vs_fs_scatter.png': 'standard_vs_fs_scatter.png',
        'step02_imagenet200_csid_vs_nearood_overlap.png': 'csid_vs_nearood_overlap.png',
    }
    for stale_name in [
        'family_overlap_heatmap.png',
        'near_auroc_trends.png',
        'near_auroc_trends.pdf',
        'overall_auroc_trends.png',
        'overall_auroc_trends.pdf',
        'family_grouped_bars.png',
        'per_method_grouped_bars.png',
        'selected_checkpoint_distributions.png',
    ]:
        stale_path = FIGURE_DIR / stale_name
        if stale_path.exists():
            stale_path.unlink()
    for stale_path in FIGURE_DIR.glob('step*.png'):
        stale_path.unlink()

    for source_name, destination_name in figure_names.items():
        source = artifact_dir() / 'figures' / source_name
        if not source.exists():
            continue
        destination = FIGURE_DIR / destination_name
        if source.suffix.lower() == '.png':
            copy_png(source, destination)
        elif source.suffix.lower() == '.pdf':
            shutil.copy2(source, destination)


FINALIST_OVERLAP_GROUPS = [
    ('Output', ['MSP', 'Energy', 'GEN', 'TSS', 'rTSS']),
    ('Distance', ['KNN', 'RkNN++', 'Mahalanobis', 'MDS++', 'RMDS', 'RMDS++', 'LSM', 'LSM++', 'rLSM', 'rLSM++']),
    ('Feature activation', ['ReAct', 'ASH', 'SCALE', 'VIM']),
    ('Gradient', ['GradNorm']),
]
FINALIST_OVERLAP_PROPOSED = {'TSS', 'rTSS', 'LSM', 'LSM++', 'rLSM', 'rLSM++', 'RkNN++'}


def write_finalist_overlap():
    methods = [method for _, group in FINALIST_OVERLAP_GROUPS for method in group]
    columns = [
        ('imagenet200', 'clean_id__vs__cs_id', 'up'),
        ('imagenet200', 'cs_id__vs__near_ood', 'down'),
        ('imagenet', 'clean_id__vs__cs_id', 'up'),
        ('imagenet', 'cs_id__vs__near_ood', 'down'),
    ]
    overlap = {}
    for dataset_key in ['imagenet200', 'imagenet']:
        frame = pd.read_parquet(SCORE_OVERLAP_ROOT / dataset_key / 'pairwise_overlap_metrics.parquet')
        frame = frame[
            (frame['protocol'] == 'full_spectrum')
            & frame['comparison'].isin(['clean_id__vs__cs_id', 'cs_id__vs__near_ood'])
        ]
        overlap[dataset_key] = frame.groupby(['method', 'comparison'])['overlap_coeff'].mean().unstack('comparison')

    values = {}
    for method in methods:
        for dataset_key, comparison, _ in columns:
            table = overlap[dataset_key]
            values[(method, dataset_key, comparison)] = (
                float(table.loc[method, comparison]) if method in table.index else None
            )

    # Per-column highlighting: best is bold + dark blue, the rest of the top three are light
    # blue, and the three weakest are orange. clean-ID to cs-ID is better high; cs-ID to
    # near-OOD is better low.
    style = {}
    for dataset_key, comparison, direction in columns:
        ranked = sorted(
            [m for m in methods if values[(m, dataset_key, comparison)] is not None],
            key=lambda m: values[(m, dataset_key, comparison)],
            reverse=(direction == 'up'),
        )
        for index, method in enumerate(ranked):
            if index == 0:
                style[(method, dataset_key, comparison)] = ('blue!38', True)
            elif index < 3:
                style[(method, dataset_key, comparison)] = ('blue!16', False)
        for method in ranked[-3:]:
            style.setdefault((method, dataset_key, comparison), ('orange!30', False))

    lines = [
        '\\begin{table}[htbp]',
        '\\centering',
        '\\small',
        '\\caption{Boundary score overlap under full-spectrum evaluation. Clean-ID$\\rightarrow$cs-ID overlap measures score invariance under covariate shift, while lower cs-ID$\\rightarrow$near-OOD overlap indicates better separation at the cs-ID versus near-OOD boundary. The best value in each column is bold. Blue marks the three best values and orange marks the three weakest, within each metric column. Proposed-method rows have bold names.}',
        '\\label{tab:finalist-overlap}',
        '\\begin{tabular}{llrrrr}',
        '\\toprule',
        ' &  & \\multicolumn{2}{c}{ImageNet-200} & \\multicolumn{2}{c}{ImageNet-1K} \\\\',
        '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}',
        'Family & Method & \\shortstack{clean-ID\\\\$\\rightarrow$ cs-ID\\\\($\\uparrow$)} & \\shortstack{cs-ID\\\\$\\rightarrow$ near-OOD\\\\($\\downarrow$)} & \\shortstack{clean-ID\\\\$\\rightarrow$ cs-ID\\\\($\\uparrow$)} & \\shortstack{cs-ID\\\\$\\rightarrow$ near-OOD\\\\($\\downarrow$)} \\\\',
        '\\midrule',
    ]
    for group_index, (family, group) in enumerate(FINALIST_OVERLAP_GROUPS):
        if group_index > 0:
            lines.append('\\midrule')
        for row_index, method in enumerate(group):
            display = display_label(method)
            name = f'\\textbf{{{display}}}' if method in FINALIST_OVERLAP_PROPOSED else display
            cells = []
            for dataset_key, comparison, _ in columns:
                value = values[(method, dataset_key, comparison)]
                cell = '--' if value is None else f'{value:.3f}'
                entry = style.get((method, dataset_key, comparison))
                if entry and value is not None:
                    color, bold = entry
                    cell = f'\\cellcolor{{{color}}}\\textbf{{{cell}}}' if bold else f'\\cellcolor{{{color}}}{cell}'
                cells.append(cell)
            family_cell = family if row_index == 0 else ''
            lines.append(f'{family_cell} & {name} & ' + ' & '.join(cells) + ' \\\\')
    lines.extend(['\\bottomrule', '\\end{tabular}', '\\end{table}', ''])
    write_lines(GENERATED_DIR / 'tab_finalist_overlap.tex', lines)


def write_boundary_inversion():
    frame = pd.read_csv(artifact_dir() / 'boundary_inversion_panel.csv')
    lines = [
        '\\begin{table}[H]',
        '\\centering',
        '\\scriptsize',
        '\\caption{Per-method boundary diagnostics under full-spectrum evaluation, all values in percent. cs/near AUROC is the cs-ID versus near-OOD AUROC, where below $50$ means the detector ranks covariate-shifted ID as more OOD than near-OOD. cs-ID rej.\\ and near-OOD acc.\\ are the wrongly rejected cs-ID percentage and the wrongly accepted near-OOD percentage at the clean-ID $95\\%$ TPR threshold, where lower is better. Values are averaged over checkpoints.}',
        '\\label{tab:boundary-inversion}',
        '\\resizebox{\\ifdim\\width>\\textwidth \\textwidth\\else\\width\\fi}{!}{%',
        '\\begin{tabular}{l rrr rrr}',
        '\\toprule',
        ' & \\multicolumn{3}{c}{ImageNet-200} & \\multicolumn{3}{c}{ImageNet-1K} \\\\',
        '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}',
        'Method & cs/near AUROC & cs-ID rej. & near-OOD acc. & cs/near AUROC & cs-ID rej. & near-OOD acc. \\\\',
        '\\midrule',
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f'{latex_escape(display_label(row.method))} & '
            f'{row.in200_cs_near_auroc:.2f} & {row.in200_csid_rejected * 100:.1f} & {row.in200_nearood_accepted * 100:.1f} & '
            f'{row.in1K_cs_near_auroc:.2f} & {row.in1K_csid_rejected * 100:.1f} & {row.in1K_nearood_accepted * 100:.1f} \\\\'
        )
    lines.extend(['\\bottomrule', '\\end{tabular}%', '}', '\\end{table}', ''])
    write_lines(GENERATED_DIR / 'tab_boundary_inversion.tex', lines)


def write_family_collapse():
    labels = {
        'output': 'Output',
        'distance': 'Distance',
        'feature_activation': 'Feature activation',
        'gradient': 'Gradient',
    }
    family_order = {'output': 0, 'distance': 1, 'feature_activation': 2, 'gradient': 3}
    frames = {
        'ImageNet-200': pd.read_parquet(
            artifact_dir() / 'supporting_tables' / 'fs_collapse_imagenet200_family_summary.parquet'
        ),
        'ImageNet-1K': pd.read_parquet(RESULTS_DIR / 'fs_collapse' / 'imagenet' / 'family_summary.parquet'),
    }
    for frame in frames.values():
        frame['family_order'] = frame['family'].map(family_order)
        frame.sort_values('family_order', inplace=True)
    families = frames['ImageNet-200']['family'].tolist()
    lines = [
        '\\begin{table}[htbp]',
        '\\centering',
        '\\scriptsize',
        '\\setlength{\\tabcolsep}{3.2pt}',
        '\\caption{Family-level collapse from standard to full-spectrum evaluation on the ImageNet settings. Negative drops indicate degradation under the full-spectrum protocol.}',
        '\\label{tab:family-collapse}',
        '\\resizebox{\\textwidth}{!}{%',
        '\\begin{tabular}{lrrrrrrrrrr}',
        '\\toprule',
        ' & \\multicolumn{5}{c}{ImageNet-200} & \\multicolumn{5}{c}{ImageNet-1K} \\\\',
        '\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}',
        'Family & Std.\\ Ov. & FS Ov. & Ov. Drop & Near Drop & Far Drop & '
        'Std.\\ Ov. & FS Ov. & Ov. Drop & Near Drop & Far Drop \\\\',
        '\\midrule',
    ]
    for family in families:
        row_200 = frames['ImageNet-200'].loc[frames['ImageNet-200']['family'] == family].iloc[0]
        row_1k = frames['ImageNet-1K'].loc[frames['ImageNet-1K']['family'] == family].iloc[0]
        lines.append(
            f'{labels.get(family, family)} & {format_number(row_200.mean_std_overall_auroc)} & '
            f'{format_number(row_200.mean_fs_overall_auroc)} & {format_number(row_200.mean_overall_drop)} & '
            f'{format_number(row_200.mean_near_drop)} & {format_number(row_200.mean_far_drop)} & '
            f'{format_number(row_1k.mean_std_overall_auroc)} & {format_number(row_1k.mean_fs_overall_auroc)} & '
            f'{format_number(row_1k.mean_overall_drop)} & {format_number(row_1k.mean_near_drop)} & '
            f'{format_number(row_1k.mean_far_drop)} \\\\'
        )
    # Mean row: average across the four families, per setting
    m200, m1k = frames['ImageNet-200'], frames['ImageNet-1K']
    lines.append('\\midrule')
    lines.append(
        f'Mean & {format_number(m200.mean_std_overall_auroc.mean())} & '
        f'{format_number(m200.mean_fs_overall_auroc.mean())} & {format_number(m200.mean_overall_drop.mean())} & '
        f'{format_number(m200.mean_near_drop.mean())} & {format_number(m200.mean_far_drop.mean())} & '
        f'{format_number(m1k.mean_std_overall_auroc.mean())} & {format_number(m1k.mean_fs_overall_auroc.mean())} & '
        f'{format_number(m1k.mean_overall_drop.mean())} & {format_number(m1k.mean_near_drop.mean())} & '
        f'{format_number(m1k.mean_far_drop.mean())} \\\\'
    )
    lines.extend(['\\bottomrule', '\\end{tabular}%', '}', '\\end{table}', ''])
    write_lines(GENERATED_DIR / 'tab_family_collapse.tex', lines)


def write_ranking_change_drivers():
    labels = {
        'near_ood_feature_covariance_alignment': (
            'Near-OOD feature similarity',
            'near-OOD remains close to ID in feature geometry',
        ),
        'threshold_transfer_gap_near': (
            'Clean-ID threshold accepts near-OOD',
            'a clean-ID operating point keeps too much near-OOD',
        ),
        'csid_near_overlap': (
            'cs-ID/near-OOD score overlap',
            'accepted shifted ID and rejected near-OOD receive similar scores',
        ),
        'score_overlap_shift': (
            'Protocol score-overlap shift',
            'the score geometry changes between protocols',
        ),
        'threshold_transfer_fpr_csid': (
            'Clean-ID threshold rejects cs-ID',
            'a clean-ID operating point rejects shifted ID',
        ),
    }
    dataset_columns = [('imagenet200', 'ImageNet-200'), ('imagenet', 'ImageNet-1K')]
    frame = pd.read_parquet(RANKING_CHANGE_ROOT / 'ranking_change_driver_summary.parquet')
    pivot = frame.pivot_table(index='top_driver', columns='dataset', values='n_methods', aggfunc='sum', fill_value=0)
    rows = []
    for driver, (label, interpretation) in labels.items():
        if driver not in pivot.index:
            continue
        counts = [int(pivot.loc[driver, dataset]) if dataset in pivot.columns else 0 for dataset, _ in dataset_columns]
        rows.append((label, counts, sum(counts), interpretation))

    lines = [
        '\\begin{table}[htbp]',
        '\\centering',
        '\\small',
        '\\caption{Heuristic drivers of standard-to-full-spectrum ranking changes after correcting threshold orientation. Counts show how many method rows selected each driver as the strongest joined explanation in the ranking-change analysis.}',
        '\\label{tab:ranking-change-drivers}',
        '\\resizebox{\\ifdim\\width>\\textwidth \\textwidth\\else\\width\\fi}{!}{%',
        '\\begin{tabular}{lrrrl}',
        '\\toprule',
        'Driver & ImageNet-200 & ImageNet-1K & Total & Interpretation \\\\',
        '\\midrule',
    ]
    for label, counts, total, interpretation in rows:
        lines.append(f'{label} & {counts[0]} & {counts[1]} & {total} & {interpretation} \\\\')
    lines.extend(['\\bottomrule', '\\end{tabular}%', '}', '\\end{table}', ''])
    write_lines(GENERATED_DIR / 'tab_ranking_change_drivers.tex', lines)


def write_hardness_threshold_transfer():
    rows = []
    for dataset_key, dataset_label_value in [('imagenet200', 'ImageNet-200'), ('imagenet', 'ImageNet-1K')]:
        hardness = pd.read_parquet(NEAR_OOD_HARDNESS_ROOT / dataset_key / 'dataset_hardness_metrics.parquet')
        alignment_col = 'feature_covariance_alignment_id_ood' if 'feature_covariance_alignment_id_ood' in hardness.columns else 'cka_id_ood'
        near_fca = hardness[hardness['ood_type'] == 'near'][alignment_col].mean()
        far_fca = hardness[hardness['ood_type'] == 'far'][alignment_col].mean()

        thresholds = pd.read_parquet(THRESHOLD_TRANSFER_ROOT / dataset_key / 'method_threshold_summary.parquet')
        thresholds = thresholds[
            (thresholds['protocol'] == 'full_spectrum')
            & (thresholds['target_tpr'].round(2) == 0.95)
        ].copy()
        thresholds['near_acceptance'] = 1.0 - thresholds['mean_tnr_near_ood']
        worst_cs = thresholds.sort_values('mean_fpr_cs_id', ascending=False).iloc[0]
        worst_near = thresholds.sort_values('near_acceptance', ascending=False).iloc[0]
        rows.append({
            'dataset': dataset_label_value,
            'near_fca': near_fca,
            'far_fca': far_fca,
            'fca_gap': near_fca - far_fca,
            'worst_cs_method': worst_cs['method'],
            'worst_cs_rejection': worst_cs['mean_fpr_cs_id'],
            'worst_near_method': worst_near['method'],
            'worst_near_acceptance': worst_near['near_acceptance'],
        })

    lines = [
        '\\begin{table}[htbp]',
        '\\centering',
        '\\small',
        '\\caption{Near-OOD difficulty by feature-covariance alignment (FCA), the cosine similarity between the ID and OOD feature covariance matrices. A larger gap between near-OOD and far-OOD means near-OOD is spread like ID in feature space and is therefore harder to separate by that geometry.}',
        '\\label{tab:hardness-threshold-transfer}',
        '\\begin{tabular}{lrrr}',
        '\\toprule',
        'Dataset & Near FCA & Far FCA & FCA gap \\\\',
        '\\midrule',
    ]
    for row in rows:
        lines.append(
            f'{row["dataset"]} & {row["near_fca"]:.3f} & {row["far_fca"]:.3f} & {row["fca_gap"]:.3f} \\\\'
        )
    lines.extend(['\\bottomrule', '\\end{tabular}', '\\end{table}', ''])
    write_lines(GENERATED_DIR / 'tab_hardness_threshold_transfer.tex', lines)


def write_threshold_transfer_all_methods():
    rows = []
    for dataset_key, dataset_label_value in [('imagenet200', 'ImageNet-200'), ('imagenet', 'ImageNet-1K')]:
        source = THRESHOLD_TRANSFER_ROOT / dataset_key / 'method_threshold_summary.parquet'
        if not source.exists():
            continue
        frame = pd.read_parquet(source)
        frame = frame[
            (frame['protocol'] == 'full_spectrum')
            & (frame['target_tpr'].round(2) == 0.95)
            & (frame['method'].isin(METHOD_ORDER))
        ].copy()
        frame['method_order'] = frame['method'].map(method_sort_key)
        frame = frame.sort_values('method_order')
        for row in frame.itertuples(index=False):
            rows.append({
                'dataset': dataset_label_value,
                'method': display_label(row.method),
                'cs_rejected': row.mean_fpr_cs_id,
                'near_accepted': 1.0 - row.mean_tnr_near_ood,
                'far_accepted': 1.0 - row.mean_tnr_far_ood,
            })

    lines = [
        '\\begin{table}[htbp]',
        '\\centering',
        '\\scriptsize',
        '\\caption{Per-method threshold diagnostics under full-spectrum evaluation. The clean-ID threshold is set at 95\\% TPR. Values are rates. Higher cs-ID rejection indicates weaker preservation of shifted ID. Higher OOD acceptance indicates weaker rejection of semantic novelty.}',
        '\\label{tab:threshold-transfer-all-methods}',
        '\\resizebox{\\ifdim\\width>\\textwidth \\textwidth\\else\\width\\fi}{!}{%',
        '\\begin{tabular}{llrrr}',
        '\\toprule',
        'Dataset & Method & cs-ID rejected & Near-OOD accepted & Far-OOD accepted \\\\',
        '\\midrule',
    ]
    body = [
        [
            row['dataset'], latex_escape(row['method']), f'{row["cs_rejected"]:.3f}',
            f'{row["near_accepted"]:.3f}', f'{row["far_accepted"]:.3f}',
        ]
        for row in rows
    ]
    lines.extend(grouped_table_rows(body, 1))
    lines.extend(['\\bottomrule', '\\end{tabular}%', '}', '\\end{table}', ''])
    write_lines(GENERATED_DIR / 'tab_threshold_transfer_all_methods.tex', lines)


def set_common_plot_style():
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 120,
    })


def family_legend_handles():
    return [
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor=FAMILY_COLORS[family],
            markeredgecolor=FAMILY_COLORS[family],
            markersize=6,
            label=label,
        )
        for family, label in FAMILY_LABELS.items()
    ]


def annotate_methods(ax, frame, x_column, y_column):
    for idx, row in frame.reset_index(drop=True).iterrows():
        dx = 5 if idx % 2 == 0 else -5
        dy = 4 if idx % 3 else -6
        ha = 'left' if dx > 0 else 'right'
        ax.annotate(
            display_label(row['method']),
            (row[x_column], row[y_column]),
            xytext=(dx, dy),
            textcoords='offset points',
            ha=ha,
            va='center',
            fontsize=6.2,
            color='#222222',
        )


def save_figure(fig, filename):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, dpi=220, bbox_inches='tight')
    plt.close(fig)


def read_boundary_overlap_summary():
    rows = []
    for dataset_key in ['imagenet200', 'imagenet']:
        source = SCORE_OVERLAP_ROOT / dataset_key / 'pairwise_overlap_metrics.parquet'
        if not source.exists():
            continue
        frame = pd.read_parquet(source)
        frame = frame[
            (frame['protocol'] == 'full_spectrum')
            & frame['comparison'].isin(['clean_id__vs__cs_id', 'cs_id__vs__near_ood'])
        ].copy()
        summary = (
            frame.groupby(['method', 'family', 'comparison'], observed=True)['overlap_coeff']
            .mean()
            .reset_index()
            .pivot(index=['method', 'family'], columns='comparison', values='overlap_coeff')
            .reset_index()
        )
        summary['dataset'] = dataset_key
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    result['method_order'] = result['method'].map(method_sort_key)
    return result.sort_values(['dataset', 'method_order']).reset_index(drop=True)


def write_boundary_overlap_figure():
    frame = read_boundary_overlap_summary()
    if frame.empty:
        return
    # Keep the canonical method panel defined above.
    frame = frame[frame['method'].isin(METHOD_ORDER)].copy()
    set_common_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharex=True, sharey=True)
    for ax, dataset_key in zip(axes, ['imagenet200', 'imagenet']):
        subset = frame[frame['dataset'] == dataset_key].copy()
        for family, family_subset in subset.groupby('family', observed=True):
            color = FAMILY_COLORS.get(family, '#777777')
            ax.scatter(
                family_subset['clean_id__vs__cs_id'],
                family_subset['cs_id__vs__near_ood'],
                s=42,
                color=color,
                edgecolor='white',
                linewidth=0.6,
                alpha=0.95,
            )
        annotate_methods(ax, subset, 'clean_id__vs__cs_id', 'cs_id__vs__near_ood')
        ax.set_title(DATASET_LABELS[dataset_key])
        ax.set_xlabel('clean-ID to cs-ID overlap')
        ax.grid(True, alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel('cs-ID to near-OOD overlap')
    axes[0].set_xlim(0.35, 0.86)
    axes[0].set_ylim(0.58, 1.00)
    axes[1].legend(handles=family_legend_handles(), loc='lower right', frameon=True)
    fig.tight_layout()
    save_figure(fig, 'all_method_boundary_overlap.png')


def write_threshold_transfer_figure():
    rows = []
    for dataset_key in ['imagenet200', 'imagenet']:
        source = THRESHOLD_TRANSFER_ROOT / dataset_key / 'method_threshold_summary.parquet'
        if not source.exists():
            continue
        frame = pd.read_parquet(source)
        frame = frame[
            (frame['protocol'] == 'full_spectrum')
            & (frame['target_tpr'].round(2) == 0.95)
        ].copy()
        frame['dataset'] = dataset_key
        frame['near_ood_acceptance'] = 1.0 - frame['mean_tnr_near_ood']
        frame['method_order'] = frame['method'].map(method_sort_key)
        rows.append(frame)
    if not rows:
        return
    frame = pd.concat(rows, ignore_index=True).sort_values(['dataset', 'method_order'])

    set_common_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharex=True, sharey=True)
    for ax, dataset_key in zip(axes, ['imagenet200', 'imagenet']):
        subset = frame[frame['dataset'] == dataset_key].copy()
        for family, family_subset in subset.groupby('family', observed=True):
            color = FAMILY_COLORS.get(family, '#777777')
            ax.scatter(
                family_subset['mean_fpr_cs_id'],
                family_subset['near_ood_acceptance'],
                s=42,
                color=color,
                edgecolor='white',
                linewidth=0.6,
                alpha=0.95,
            )
        annotate_methods(ax, subset, 'mean_fpr_cs_id', 'near_ood_acceptance')
        ax.set_title(DATASET_LABELS[dataset_key])
        ax.set_xlabel('cs-ID rejected')
        ax.grid(True, alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel('near-OOD accepted')
    axes[0].set_xlim(0.00, 0.50)
    axes[0].set_ylim(0.62, 1.00)
    axes[1].legend(handles=family_legend_handles(), loc='lower right', frameon=True)
    fig.tight_layout()
    save_figure(fig, 'all_method_threshold_transfer.png')


def write_protocol_shift_figure():
    source = RESULTS_DIR / 'ranking_stability' / 'raw_metrics.parquet'
    if not source.exists():
        return
    frame = pd.read_parquet(source)
    frame = frame[
        frame['dataset'].isin(['imagenet200', 'imagenet'])
        & frame['protocol'].isin(['standard', 'full_spectrum'])
    ].copy()
    summary = (
        frame.groupby(['dataset', 'method', 'family', 'protocol'], observed=True)['near_auroc']
        .mean()
        .reset_index()
        .pivot(index=['dataset', 'method', 'family'], columns='protocol', values='near_auroc')
        .reset_index()
        .dropna(subset=['standard', 'full_spectrum'])
    )
    if summary.empty:
        return
    summary['method_order'] = summary['method'].map(method_sort_key)
    methods = [method for method in METHOD_ORDER if method in set(summary['method']) and method not in CUSTOM_METHODS]
    y_positions = {method: idx for idx, method in enumerate(methods)}

    set_common_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), sharey=True)
    for ax, dataset_key in zip(axes, ['imagenet200', 'imagenet']):
        subset = summary[summary['dataset'] == dataset_key].sort_values('method_order')
        # Keep only panel methods with a y-slot.
        subset = subset[subset['method'].isin(y_positions)]
        for row in subset.itertuples(index=False):
            y = y_positions[row.method]
            color = FAMILY_COLORS.get(row.family, '#777777')
            is_custom = row.method in CUSTOM_METHODS
            edge = '#222222' if is_custom else color
            line_width = 2.0 if is_custom else 1.5
            point_size = 46 if is_custom else 36
            ax.plot([row.standard, row.full_spectrum], [y, y], color=color, alpha=0.62, linewidth=line_width)
            ax.scatter(row.standard, y, marker='o', facecolor='white', edgecolor=edge, linewidth=1.5, s=point_size, zorder=3)
            ax.scatter(row.full_spectrum, y, marker='s', facecolor=color, edgecolor=edge, linewidth=0.9, s=point_size + 2, zorder=3)
        ax.set_title(DATASET_LABELS[dataset_key], fontsize=15)
        ax.set_xlabel('near-OOD AUROC', fontsize=14)
        ax.grid(True, axis='x', alpha=0.25, linewidth=0.6)
        ax.set_xlim(40, 90)
        ax.tick_params(axis='x', labelsize=12)
    axes[0].set_yticks(np.arange(len(methods)))
    axes[0].set_yticklabels([display_label(method) for method in methods], fontsize=12.5)
    axes[0].invert_yaxis()
    protocol_handles = [
        Line2D([0], [0], marker='o', color='#333333', markerfacecolor='white', markeredgecolor='#333333', linestyle='none', label='Standard'),
        Line2D([0], [0], marker='s', color='#333333', markerfacecolor='#333333', markeredgecolor='#333333', linestyle='none', label='Full-spectrum'),
    ]
    fig.legend(handles=protocol_handles + family_legend_handles(), loc='lower center',
               ncol=6, frameon=True, fontsize=10.5)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    save_figure(fig, 'all_method_protocol_shift_near_auroc.png')


def write_all_method_figures():
    write_boundary_overlap_figure()
    write_threshold_transfer_figure()
    write_protocol_shift_figure()


def write_hard_case_overlap():
    pairwise = pd.read_parquet(SCORE_OVERLAP_ROOT / 'imagenet200' / 'pairwise_overlap_metrics.parquet')
    methods = ['KNN', 'MDS++', 'Mahalanobis', 'MSP', 'ASH', 'ReAct']
    frame = pairwise[
        pairwise['method'].isin(methods)
        & pairwise['comparison'].isin(['clean_id__vs__cs_id', 'cs_id__vs__near_ood'])
    ].pivot_table(index='method', columns='comparison', values='overlap_coeff', aggfunc='mean')
    frame = frame.reindex(methods).dropna(how='all')
    lines = [
        '\\begin{table}[htbp]',
        '\\centering',
        '\\small',
        '\\caption{Hard-case overlap diagnostics from the current ImageNet-200 score-overlap study. Clean-ID to cs-ID overlap measures score invariance under covariate shift, while lower cs-ID to near-OOD overlap indicates better separation at the hard boundary.}',
        '\\label{tab:hard-case-overlap}',
        '\\begin{tabular}{lrrr}',
        '\\toprule',
        'Method & clean-ID $\\rightarrow$ cs-ID & cs-ID $\\rightarrow$ near-OOD & Balance \\\\',
        '\\midrule',
    ]
    for method, row in frame.iterrows():
        clean_csid = row['clean_id__vs__cs_id']
        csid_near = row['cs_id__vs__near_ood']
        balance = clean_csid - csid_near
        lines.append(
            f'{display_label(method)} & {clean_csid:.3f} & {csid_near:.3f} & {balance:.3f} \\\\'
        )
    lines.extend(['\\bottomrule', '\\end{tabular}', '\\end{table}', ''])
    write_lines(GENERATED_DIR / 'tab_hard_case_overlap.tex', lines)


def write_class_conditioning():
    labels = {
        ('features', 'global_distance'): 'Features / global distance',
        ('features', 'predicted_class_distance'): 'Features / predicted-class distance',
        ('features', 'top2_margin'): 'Features / top-2 margin',
        ('logits', 'global_distance'): 'Logits / global distance',
        ('logits', 'predicted_class_distance'): 'Logits / predicted-class distance',
        ('logits', 'top2_margin'): 'Logits / top-2 margin',
    }
    frame = pd.read_parquet(CLASS_CONDITIONING_ROOT / 'imagenet200' / 'class_conditioning_summary.parquet')
    rows = []
    for key, label in labels.items():
        activation, scorer = key
        subset = frame[
            (frame['activation'] == activation)
            & (frame['scorer'] == scorer)
            & (frame['comparison'] == 'cs_id__vs__near_ood')
        ]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append((label, row))

    lines = [
        '\\begin{table}[htbp]',
        '\\centering',
        '\\footnotesize',
        '\\caption{Selected cs-ID versus near-OOD structures from the current ImageNet-200 class-conditioning analysis. The main comparison is cs-ID versus near-OOD.}',
        '\\label{tab:class-conditioning}',
        '\\begin{tabular}{lrrr}',
        '\\toprule',
        'Representation / scorer & clean-ID $\\rightarrow$ cs-ID & cs-ID $\\rightarrow$ near-OOD & AUROC \\\\',
        '\\midrule',
    ]
    for label, row in rows:
        lines.append(
            f'{label} & {row.clean_csid_overlap:.3f} & {row.overlap_coeff:.3f} & '
            f'{format_number(row.csid_near_auroc)} \\\\'
        )
    lines.extend(['\\bottomrule', '\\end{tabular}', '\\end{table}', ''])
    write_lines(GENERATED_DIR / 'tab_class_conditioning.tex', lines)


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    write_final_methods()
    write_core_baselines()
    write_openood_style_leaderboard()
    write_tss_temperature()
    write_tss_class_baseline()
    write_ablation_summary()
    write_family_collapse()
    write_boundary_inversion()
    write_finalist_overlap()
    write_ranking_change_drivers()
    write_hardness_threshold_transfer()
    write_threshold_transfer_all_methods()
    write_all_method_figures()
    write_hard_case_overlap()
    write_class_conditioning()
    copy_figures()
    print(f'Updated generated thesis tables in {GENERATED_DIR}')
    print(f'Updated thesis figures in {FIGURE_DIR}')


if __name__ == '__main__':
    main()
