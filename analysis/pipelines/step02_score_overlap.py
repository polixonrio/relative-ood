"""Step 02: extract detector score distributions and overlap summaries.

This step reads step 01 raw metrics to choose checkpoints, scores every
requested method on clean ID, CS-ID, near-OOD, and far-OOD splits, then writes
per-sample score shards plus compact group and pairwise-overlap tables.
"""

from __future__ import annotations

import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.core.benchmarks import ACTIVE_DATASETS, CACHE_DIR, CONFIG_ROOT, DATASET_NETWORKS, DATASET_PROTOCOLS, DATASET_SAMPLE_GROUPS, REBUILD_ANALYSIS_OUTPUTS, RESULTS_DIR
from analysis.core.model_cache import iter_dataset_model_caches
from analysis.core.table_store import figures_dir, write_artifacts, write_dataframe
from analysis.core.methods.registry import FAMILY_COLORS, FAMILY_ORDER, METHOD_ORDER, load_method_params, method_panel
from analysis.core.method_scoring import CachedMethodRunner
from analysis.core.scoring import compute_overlap_metrics, summarize_scores

OUTPUT_ROOT = RESULTS_DIR / 'score_overlap'

FULL_SPECTRUM_COMPARISONS = [('clean_id', 'cs_id'), ('cs_id', 'near_ood'), ('near_ood', 'far_ood'), ('clean_id', 'near_ood'), ('clean_id', 'far_ood'), ('cs_id', 'far_ood')]
STANDARD_COMPARISONS = [('clean_id', 'near_ood'), ('clean_id', 'far_ood'), ('near_ood', 'far_ood')]

GROUP_KEY_FIELDS = ('checkpoint_name', 'checkpoint_path', 'dataset', 'network', 'protocol', 'method', 'family', 'sample_group')


def cleanup_per_sample_outputs(output_dir: Path) -> None:
    """Remove stale per-sample score outputs before a forced rerun."""
    base = output_dir / 'per_sample_scores.parquet'
    if base.exists():
        base.unlink()
    parts_dir = output_dir / 'per_sample_scores_parts'
    if parts_dir.is_dir():
        shutil.rmtree(parts_dir)


def protocol_part_name(protocols: list[str]) -> str:
    """Build the filename component that records which protocols were scored."""
    return '_'.join(protocols)


def per_sample_part_path(parts_dir: Path, checkpoint_path: Path, checkpoint_name: str, method_spec: dict[str, str], protocols: list[str]) -> Path:
    """Return the per-checkpoint/method parquet shard path."""
    checkpoint_hash = hashlib.sha1(str(checkpoint_path.resolve()).encode('utf-8')).hexdigest()[:10]
    return parts_dir / (
        f'{checkpoint_name}__{checkpoint_hash}__'
        f'{method_spec["postprocessor"]}__{protocol_part_name(protocols)}.parquet'
    )


def iter_per_sample_frames(parts_dir: Path, columns: list[str] | None = None):
    """Yield all per-sample score shards, optionally column-pruned."""
    for part_path in sorted(parts_dir.glob('*.parquet')):
        yield pd.read_parquet(part_path, columns=columns)


def iter_split_specs(protocols: list[str], dataset_name: str) -> list[dict]:
    """Describe which model-cache splits belong to each analysis sample group."""
    base_protocols = [p for p in protocols if p in {'standard', 'full_spectrum'}]
    specs: list[dict] = []
    for sample_group, splits in DATASET_SAMPLE_GROUPS[dataset_name].items():
        if sample_group == 'cs_id' and 'full_spectrum' not in protocols:
            continue
        split_protocols = ['full_spectrum'] if sample_group == 'cs_id' else base_protocols
        if not split_protocols:
            continue
        for group_key, loader_name in splits:
            split_dataset = f'{dataset_name}_test' if sample_group == 'clean_id' else loader_name
            specs.append({'sample_group': sample_group, 'dataset_name': split_dataset,
                          'group_key': group_key, 'loader_name': loader_name, 'protocols': split_protocols})
    return specs


def sample_rows_to_frame(checkpoint_name: str, checkpoint_path: Path, analysis_dataset: str, network_name: str, protocol: str, method_label: str, postprocessor_name: str, family: str, split_dataset_name: str, sample_group: str, score: np.ndarray, true_label: np.ndarray, predicted_label: np.ndarray) -> pd.DataFrame:
    """Create one long-form per-sample score frame for a method/split/protocol."""
    frame = pd.DataFrame({'score': score, 'true_label': true_label, 'predicted_label': predicted_label})
    scalar_columns = {
        'checkpoint_name': checkpoint_name,
        'checkpoint_path': str(checkpoint_path),
        'dataset': analysis_dataset,
        'network': network_name,
        'protocol': protocol,
        'method': method_label,
        'postprocessor': postprocessor_name,
        'family': family,
        'dataset_name': split_dataset_name,
        'sample_group': sample_group,
    }
    for column, value in scalar_columns.items():
        frame[column] = value
    return frame


def plot_selected_checkpoint_distributions(per_sample_parts_dir: Path, selected_checkpoint: str, figures_dir: Path) -> None:
    """Plot score histograms for one checkpoint across methods and sample groups."""
    subset_frames = []
    columns = ['checkpoint_name', 'protocol', 'method', 'sample_group', 'score']
    for frame in iter_per_sample_frames(per_sample_parts_dir, columns=columns):
        filtered = frame[(frame['protocol'] == 'full_spectrum') & (frame['checkpoint_name'] == selected_checkpoint)]
        if not filtered.empty:
            subset_frames.append(filtered)
    if not subset_frames:
        return
    subset = pd.concat(subset_frames, ignore_index=True)
    if subset.empty:
        return

    groups = ['clean_id', 'cs_id', 'near_ood', 'far_ood']
    colors = {'clean_id': '#1f77b4', 'cs_id': '#17becf', 'near_ood': '#ff7f0e', 'far_ood': '#d62728'}

    fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=False, sharey=False)
    axes = axes.flatten()
    for ax, method in zip(axes, METHOD_ORDER):
        method_df = subset[subset['method'] == method]
        if method_df.empty:
            ax.set_visible(False)
            continue
        for group in groups:
            scores = method_df.loc[method_df['sample_group'] == group, 'score']
            if scores.empty:
                continue
            ax.hist(scores, bins=60, density=True, histtype='step', linewidth=1.4, color=colors[group], label=group)
        ax.set_title(method)
    for ax in axes[len(METHOD_ORDER):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=4)
    fig.suptitle(f'Full-Spectrum Score Distributions for {selected_checkpoint}', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figures_dir / 'selected_checkpoint_distributions.png', dpi=180)
    plt.close(fig)


def plot_csid_nearood_overlap(pairwise_metrics: pd.DataFrame, figures_dir: Path) -> None:
    """Plot full-spectrum CS-ID vs near-OOD overlap by method."""
    subset = pairwise_metrics[
        (pairwise_metrics['protocol'] == 'full_spectrum')
        & (pairwise_metrics['comparison'] == 'cs_id__vs__near_ood')
    ]
    if subset.empty:
        return
    summary = subset.groupby(['method', 'family'], observed=True)['overlap_coeff'].mean().reset_index()
    summary['method'] = pd.Categorical(summary['method'], categories=METHOD_ORDER, ordered=True)
    summary = summary.sort_values('method')
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(summary['method'].tolist(), summary['overlap_coeff'], color=[FAMILY_COLORS[str(family)] for family in summary['family']])
    ax.set_ylabel('Overlap coefficient')
    ax.set_title('cs_id__vs__near_ood (overlap_coeff)')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(figures_dir / 'csid_vs_nearood_overlap.png', dpi=180)
    plt.close(fig)


def build_outputs_from_records(group_rows: list[dict], pairwise_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build sorted output tables from accumulated summary records."""
    group_summary = pd.DataFrame.from_records(group_rows)
    pairwise_metrics = pd.DataFrame.from_records(pairwise_rows)
    if not group_summary.empty:
        group_summary['family'] = pd.Categorical(group_summary['family'], categories=FAMILY_ORDER, ordered=True)
        group_summary['method'] = pd.Categorical(group_summary['method'], categories=METHOD_ORDER, ordered=True)
        group_summary = group_summary.sort_values(['checkpoint_name', 'protocol', 'family', 'method', 'sample_group']).reset_index(drop=True)
    if not pairwise_metrics.empty:
        pairwise_metrics['family'] = pd.Categorical(pairwise_metrics['family'], categories=FAMILY_ORDER, ordered=True)
        pairwise_metrics['method'] = pd.Categorical(pairwise_metrics['method'], categories=METHOD_ORDER, ordered=True)
        pairwise_metrics = pairwise_metrics.sort_values(['checkpoint_name', 'protocol', 'family', 'method', 'comparison']).reset_index(drop=True)
    return group_summary, pairwise_metrics


def summarize_per_sample_table(per_sample_parts_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-sample shards into group summaries and pairwise metrics."""
    group_keys = list(GROUP_KEY_FIELDS)
    columns = group_keys + ['score']
    grouped_scores: dict[tuple, np.ndarray] = {}

    for frame in iter_per_sample_frames(per_sample_parts_dir, columns=columns):
        if frame.empty:
            continue
        for keys, group in frame.groupby(group_keys, sort=False, observed=True):
            score_values = group['score'].to_numpy(dtype=float, copy=False)
            existing = grouped_scores.get(keys)
            grouped_scores[keys] = score_values if existing is None else np.concatenate([existing, score_values])

    if not grouped_scores:
        return pd.DataFrame(), pd.DataFrame()

    group_rows: list[dict] = []
    pairwise_inputs: dict[tuple, dict[str, np.ndarray]] = defaultdict(dict)
    for keys, scores in grouped_scores.items():
        row = summarize_scores(scores)
        row.update(zip(GROUP_KEY_FIELDS, keys))
        group_rows.append(row)
        pairwise_inputs[keys[:-1]][keys[-1]] = scores

    pairwise_rows: list[dict] = []
    pairwise_fields = GROUP_KEY_FIELDS[:-1]
    for keys, by_group in pairwise_inputs.items():
        base = dict(zip(pairwise_fields, keys))
        comparisons = FULL_SPECTRUM_COMPARISONS if base['protocol'] == 'full_spectrum' else STANDARD_COMPARISONS
        for group_a, group_b in comparisons:
            if group_a not in by_group or group_b not in by_group:
                continue
            metrics = compute_overlap_metrics(by_group[group_a], by_group[group_b])
            metrics.update(base)
            metrics.update({'comparison': f'{group_a}__vs__{group_b}', 'group_a': group_a, 'group_b': group_b})
            pairwise_rows.append(metrics)

    return build_outputs_from_records(group_rows, pairwise_rows)


def prepare_run_context(dataset: str, protocols: list[str]):
    """Resolve cached checkpoints, methods, and split specs for the run."""
    cache_entries = list(iter_dataset_model_caches(CACHE_DIR, dataset))
    method_specs = method_panel()
    split_specs = iter_split_specs(protocols, dataset)
    return cache_entries, method_specs, split_specs


def write_method_scores(dataset: str, network: str, checkpoint_name: str, checkpoint_path: Path, method_spec: dict[str, str], runner: CachedMethodRunner, split_specs: list[dict]) -> list[pd.DataFrame] | None:
    """Score one method across all requested split specs."""
    frames = []
    for split_spec in split_specs:
        try:
            predicted, confidence, labels = runner.score_split(split_spec['group_key'], split_spec['loader_name'])
        except Exception as exc:
            print(f'    failed scoring {method_spec["label"]} on {split_spec["dataset_name"]}: {type(exc).__name__}: {exc}', flush=True)
            return None

        score = -confidence
        for protocol in split_spec['protocols']:
            frame = sample_rows_to_frame(checkpoint_name, checkpoint_path, dataset, network, protocol, method_spec['label'],
                                         method_spec['postprocessor'], method_spec['family'], split_spec['dataset_name'], split_spec['sample_group'], score,
                                         labels, predicted)
            frames.append(frame)
    return frames


def run_dataset(dataset: str) -> dict:
    """Extract missing score shards and summarize the complete shard directory."""
    network = DATASET_NETWORKS[dataset]
    protocols = DATASET_PROTOCOLS[dataset]
    cache_entries, method_specs, split_specs = prepare_run_context(dataset, protocols)

    output_dir = OUTPUT_ROOT / dataset
    per_sample_parts_dir = output_dir / 'per_sample_scores_parts'
    output_dir.mkdir(parents=True, exist_ok=True)
    if REBUILD_ANALYSIS_OUTPUTS:
        cleanup_per_sample_outputs(output_dir)
    per_sample_parts_dir.mkdir(parents=True, exist_ok=True)

    print(f'Extracting score overlap for {len(cache_entries)} checkpoints and '
          f'{len(method_specs)} methods...', flush=True)

    for checkpoint_index, (checkpoint_name, checkpoint_path, model_cache) in enumerate(cache_entries, start=1):
        print(f'[{checkpoint_index}/{len(cache_entries)}] {checkpoint_name}', flush=True)

        for method_index, method_spec in enumerate(method_specs, start=1):
            part_path = per_sample_part_path(per_sample_parts_dir, checkpoint_path, checkpoint_name, method_spec, protocols)
            if part_path.exists() and not REBUILD_ANALYSIS_OUTPUTS:
                print(f'  [{method_index}/{len(method_specs)}] {method_spec["label"]} '
                      '(skip: already complete)', flush=True)
                continue
            print(f'  [{method_index}/{len(method_specs)}] {method_spec["label"]}', flush=True)
            try:
                params = load_method_params(method_spec['postprocessor'], CONFIG_ROOT, dataset)
                runner = CachedMethodRunner(method_spec=method_spec, params=params, model_cache=model_cache)
            except Exception as exc:
                print(f'    failed setup for {method_spec["label"]}: {type(exc).__name__}: {exc}', flush=True)
                continue

            method_frames = write_method_scores(dataset, network, checkpoint_name, checkpoint_path, method_spec, runner, split_specs)
            if method_frames is None:
                continue

            write_dataframe(pd.concat(method_frames, ignore_index=True), part_path)
            print('    saved scores (cached scoring)', flush=True)

        del model_cache

    group_summary, pairwise_metrics = summarize_per_sample_table(per_sample_parts_dir)

    return {
        'per_sample_parts_dir': per_sample_parts_dir,
        'per_sample_format': 'parquet',
        'group_summary': group_summary,
        'pairwise_metrics': pairwise_metrics,
    }


def write_outputs(result: dict, output_dir: Path, selected_checkpoint: str | None) -> None:
    """Write summary tables and figures for step 02."""
    plot_dir = figures_dir(output_dir)
    write_artifacts(output_dir, {
        'group_score_summary': result['group_summary'],
        'pairwise_overlap_metrics': result['pairwise_metrics'],
    })

    if selected_checkpoint is None:
        if not result['group_summary'].empty:
            selected_checkpoint = sorted(result['group_summary']['checkpoint_name'].unique().tolist())[0]

    if selected_checkpoint is not None:
        plot_selected_checkpoint_distributions(result['per_sample_parts_dir'], selected_checkpoint, plot_dir)
    plot_csid_nearood_overlap(result['pairwise_metrics'], plot_dir)


def main() -> None:
    """CLI entrypoint for step 02."""
    for dataset in ACTIVE_DATASETS:
        output_dir = OUTPUT_ROOT / dataset
        result = run_dataset(dataset)
        write_outputs(result, output_dir, selected_checkpoint=None)
        print(f'Wrote outputs to {output_dir}')
        print('Generated:')
        print(f' - per_sample_scores_parts/ ({result["per_sample_format"]})')
        print(' - group_score_summary.parquet')
        print(' - pairwise_overlap_metrics.parquet')
        print(' - figures/')


if __name__ == '__main__':
    main()
