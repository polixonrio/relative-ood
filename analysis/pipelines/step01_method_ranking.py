"""Step 01: checkpoint ranking stability.

This step is the main producer for the rest of the analysis workflow. It runs
the requested detector methods against each checkpoint, writes one raw metric
row per checkpoint/method/protocol, then writes the final ranking summary used
by later cached analyses and the thesis export.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analysis.core.benchmarks import ACTIVE_DATASETS, CACHE_DIR, CONFIG_ROOT, DATASET_NETWORKS, DATASET_PROTOCOLS, DATASET_SAMPLE_GROUPS, REBUILD_ANALYSIS_OUTPUTS, RESULTS_DIR
from analysis.core.model_cache import iter_dataset_model_caches
from analysis.core.scoring import openood_metric_array
from analysis.core.table_store import write_artifacts, write_dataframe
from analysis.core.methods.registry import METHODS, load_method_params, method_panel
from analysis.core.method_scoring import CachedMethodRunner

OUTPUT_DIR = RESULTS_DIR / 'ranking_stability'
CHECKPOINT_ROOT = RESULTS_DIR
RANKING_COLUMNS = ['dataset', 'network', 'checkpoint_name', 'checkpoint_path', 'protocol', 'method', 'postprocessor', 'family', 'overall_auroc', 'overall_rank']


def parse_args() -> argparse.Namespace:
    """Parse optional local-run filters while preserving the default full run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=sorted(METHODS),
        help='Optional canonical method names to evaluate, for example: --methods gen scale.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR,
        help='Output directory for raw metrics and ranking summary.',
    )
    parser.add_argument(
        '--reuse-existing',
        action='store_true',
        help='Reuse existing per-method metric rows instead of forcing recomputation.',
    )
    parser.add_argument(
        '--append-existing-output',
        action='store_true',
        help='Merge newly evaluated rows into an existing output raw_metrics.parquet before rebuilding the ranking summary.',
    )
    return parser.parse_args()


def selected_method_panel(method_names: list[str] | None) -> list[dict[str, str]]:
    """Return either the full method panel or a CLI-filtered subset."""
    if not method_names:
        return method_panel()
    return [METHODS[name] for name in method_names]


def summarize_protocol_metrics(metrics: pd.DataFrame) -> dict[str, float]:
    index = list(metrics.index)
    if 'nearood' not in index or 'farood' not in index:
        raise ValueError('OpenOOD metrics frame is missing nearood/farood rows')

    near_boundary = index.index('nearood')
    far_boundary = index.index('farood')
    near_rows = index[:near_boundary]
    far_rows = index[near_boundary + 1:far_boundary]

    overall = metrics.loc[near_rows + far_rows].mean(axis=0)
    near = metrics.loc['nearood']
    far = metrics.loc['farood']

    return {
        'near_auroc': float(near['AUROC']),
        'far_auroc': float(far['AUROC']),
        'overall_auroc': float(overall['AUROC']),
        'overall_fpr95': float(overall['FPR@95']),
        'near_aupr_in': float(near['AUPR_IN']),
        'near_aupr_out': float(near['AUPR_OUT']),
        'far_aupr_in': float(far['AUPR_IN']),
        'far_aupr_out': float(far['AUPR_OUT']),
        'overall_aupr_in': float(overall['AUPR_IN']),
        'overall_aupr_out': float(overall['AUPR_OUT']),
        'near_fpr95': float(near['FPR@95']),
        'far_fpr95': float(far['FPR@95']),
        'id_or_csid_acc': float(overall['ACC']),
    }


def build_ranking_summary(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    if 'overall_auroc' not in raw_metrics.columns:
        return pd.DataFrame(columns=RANKING_COLUMNS)

    successful = raw_metrics[(raw_metrics['status'] == 'success') & raw_metrics['overall_auroc'].notna()].copy()
    if successful.empty:
        return pd.DataFrame(columns=RANKING_COLUMNS)

    group_keys = ['dataset', 'network', 'checkpoint_name', 'checkpoint_path', 'protocol']
    ranked = successful.sort_values(group_keys + ['overall_auroc', 'method'], ascending=[*([True] * len(group_keys)), False, True]).copy()
    ranked['overall_rank'] = ranked.groupby(group_keys, sort=False)['overall_auroc'].rank(method='min', ascending=False).astype(int)
    return ranked[RANKING_COLUMNS].reset_index(drop=True)


def base_classifier_accuracy(runner: CachedMethodRunner, dataset_name: str, protocol: str) -> float:
    outputs = [runner.model_cache.get_outputs('id', 'test')]
    if protocol == 'full_spectrum':
        outputs.extend(runner.model_cache.get_outputs(group, name) for group, name in DATASET_SAMPLE_GROUPS[dataset_name]['cs_id'])

    preds = np.concatenate([output.preds for output in outputs])
    labels = np.concatenate([output.labels for output in outputs])
    return 100.0 * float((preds == labels).sum()) / float(len(labels))


def evaluate_protocol_with_method(runner: CachedMethodRunner, dataset_name: str, protocol: str) -> pd.DataFrame:
    sample_groups = DATASET_SAMPLE_GROUPS[dataset_name]
    if protocol == 'full_spectrum':
        splits = [runner.score_split('id', 'test')]
        splits.extend(runner.score_split(group, name) for group, name in sample_groups['cs_id'])
        preds, confs, labels = zip(*splits)
        id_preds = np.concatenate(preds)
        id_conf = np.concatenate(confs)
        id_labels = np.concatenate(labels)
    else:
        id_preds, id_conf, id_labels = runner.score_split('id', 'test')

    id_acc = base_classifier_accuracy(runner, dataset_name, protocol)
    near_names = [name for _, name in sample_groups['near_ood']]
    far_names = [name for _, name in sample_groups['far_ood']]
    id_tuple = (id_preds, id_conf, id_labels)
    near_metrics = openood_metric_array(id_tuple, 'ood_near', near_names, runner.score_split, include_mean=True)
    far_metrics = openood_metric_array(id_tuple, 'ood_far', far_names, runner.score_split, include_mean=True)
    near_metrics[:, -1] = id_acc
    far_metrics[:, -1] = id_acc

    return pd.DataFrame(
        np.concatenate([near_metrics, far_metrics], axis=0),
        index=near_names + ['nearood'] + far_names + ['farood'],
        columns=['FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC'],
    )


def evaluate_checkpoints(dataset: str, network: str, protocols: list[str], method_specs: Iterable[dict[str, str]], metric_rows_dir: Path, force: bool = False) -> pd.DataFrame:
    cache_entries = list(iter_dataset_model_caches(CACHE_DIR, dataset))
    method_specs = list(method_specs)
    total_checkpoints = len(cache_entries)
    total_methods = len(method_specs)

    print(f'Evaluating {total_checkpoints} checkpoint(s) x {total_methods} method(s) '
          f'for dataset={dataset}, network={network}, protocols={", ".join(protocols)}')

    expected_metric_row_paths: list[Path] = []
    for checkpoint_index, (checkpoint_name, checkpoint_path, model_cache) in enumerate(cache_entries, start=1):
        print(f'[checkpoint {checkpoint_index}/{total_checkpoints}] {checkpoint_name}')

        checkpoint_hash = hashlib.sha1(str(checkpoint_path.resolve()).encode('utf-8')).hexdigest()[:10]
        for method_index, method_spec in enumerate(method_specs, start=1):
            protocol_paths = [
                (protocol, metric_rows_dir / f'{checkpoint_name}__{checkpoint_hash}__{protocol}__{method_spec["postprocessor"]}.parquet')
                for protocol in protocols
            ]
            expected_metric_row_paths.extend(path for _, path in protocol_paths)
            pending_protocols = [(protocol, path) for protocol, path in protocol_paths if force or not path.exists()]
            if not pending_protocols:
                print(f'  [method {method_index}/{total_methods}] {method_spec["label"]} (skip: already complete)')
                continue
            print(f'  [method {method_index}/{total_methods}] {method_spec["label"]}')

            runner_error = None
            try:
                params = load_method_params(method_spec['postprocessor'], CONFIG_ROOT, dataset)
                runner = CachedMethodRunner(method_spec=method_spec, params=params, model_cache=model_cache)
                runner_note = 'cached logits/features'
            except Exception as exc:
                runner_error = f'{type(exc).__name__}: {exc}'
                print(f'    setup failed: {runner_error}')

            for protocol_index, (protocol, metric_row_path) in enumerate(pending_protocols, start=1):
                print(f'    [protocol {protocol_index}/{len(pending_protocols)}] {protocol}')
                row = {
                    'checkpoint_name': checkpoint_name,
                    'checkpoint_path': str(checkpoint_path.resolve().relative_to(CHECKPOINT_ROOT.resolve())),
                    'dataset': dataset,
                    'network': network,
                    'protocol': protocol,
                    'method': method_spec['label'],
                    'postprocessor': method_spec['postprocessor'],
                    'family': method_spec['family'],
                    'status': 'failed',
                    'notes': '',
                }

                if runner_error is not None:
                    row['notes'] = runner_error
                    print(f'      failed before evaluation: {runner_error}')
                    write_dataframe(pd.DataFrame([row]), metric_row_path)
                    continue

                try:
                    metrics = evaluate_protocol_with_method(runner, dataset, protocol)
                    summary = summarize_protocol_metrics(metrics)
                    row.update(summary)
                    row['status'] = 'success'
                    row['notes'] = runner_note
                    print('      success: '
                          f"near_auroc={summary.get('near_auroc')}, "
                          f"far_auroc={summary.get('far_auroc')}, "
                          f"overall_auroc={summary.get('overall_auroc')}")
                except Exception as exc:
                    row['status'] = 'failed'
                    row['notes'] = f'{type(exc).__name__}: {exc}'
                    print(f"      failed: {row['notes']}")
                write_dataframe(pd.DataFrame([row]), metric_row_path)

    metric_row_paths = sorted(path for path in expected_metric_row_paths if path.is_file())
    if not metric_row_paths:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(path) for path in metric_row_paths), ignore_index=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    force = REBUILD_ANALYSIS_OUTPUTS and not args.reuse_existing
    metric_rows_dir = output_dir / 'raw_metric_rows'
    if force and not args.append_existing_output and metric_rows_dir.is_dir():
        shutil.rmtree(metric_rows_dir)

    method_specs = selected_method_panel(args.methods)
    raw_metric_frames = []
    for dataset in ACTIVE_DATASETS:
        network = DATASET_NETWORKS[dataset]
        protocols = DATASET_PROTOCOLS[dataset]
        dataset_metric_rows_dir = metric_rows_dir / dataset
        print(f'Starting ranking stability run for dataset={dataset}, network={network}, output_dir={output_dir}')
        dataset_metrics = evaluate_checkpoints(
            dataset,
            network,
            protocols,
            method_specs,
            metric_rows_dir=dataset_metric_rows_dir,
            force=force,
        )
        if not dataset_metrics.empty:
            raw_metric_frames.append(dataset_metrics)

    raw_metrics = pd.concat(raw_metric_frames, ignore_index=True) if raw_metric_frames else pd.DataFrame()
    existing_raw_metrics_path = output_dir / 'raw_metrics.parquet'
    if args.append_existing_output and existing_raw_metrics_path.is_file():
        existing_raw_metrics = pd.read_parquet(existing_raw_metrics_path)
        raw_metrics = pd.concat([existing_raw_metrics, raw_metrics], ignore_index=True)
    if not raw_metrics.empty:
        raw_metrics = raw_metrics.drop_duplicates(
            subset=['dataset', 'network', 'checkpoint_path', 'protocol', 'postprocessor'],
            keep='last',
        ).reset_index(drop=True)
    ranking_summary = build_ranking_summary(raw_metrics)
    write_artifacts(output_dir, {'raw_metrics': raw_metrics, 'ranking_summary': ranking_summary})

    print(f'Wrote raw metrics to {output_dir / "raw_metrics.parquet"}')
    print(f'Wrote ranking summary to {output_dir / "ranking_summary.parquet"}')


if __name__ == '__main__':
    main()
