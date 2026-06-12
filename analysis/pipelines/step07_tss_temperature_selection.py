"""Step 07: choose TSS/rTSS temperatures from cached logits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.core.benchmarks import CACHE_DIR, RESULTS_DIR
from analysis.core.methods.rtss import EPS, fit_tss_statistics, tss_from_logits
from analysis.core.model_cache import ModelCache, iter_model_caches
from analysis.core.scoring import openood_metric_array
from analysis.core.table_store import write_artifacts

OUTPUT_DIR = RESULTS_DIR / 'tss_temperature_sensitivity'
T2_VALUES = [0.25, 0.33, 0.5, 0.67, 0.75, 0.9]
DATASETS = {'cifar100', 'imagenet200', 'imagenet'}


def score_split(logits: np.ndarray, preds: np.ndarray, method: str, t2: float, states: dict) -> np.ndarray:
    scores = tss_from_logits(logits, t1=1.0, t2=t2)
    if method == 'TSS':
        return -scores
    class_mu, class_sigma = states[t2]
    return -((scores - class_mu[preds]) / np.maximum(class_sigma[preds], EPS))


def score_cache(model_cache: ModelCache, split_groups: list[tuple[str, str]], method: str, t2: float, states: dict) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    scored = {}
    for group, name in split_groups:
        logits, preds, labels = model_cache.get_arrays(group, name, 'logits', 'preds', 'labels')
        scored[(group, name)] = (preds, score_split(logits, preds, method, t2, states), labels)
    return scored


def summarize_performance(scored: dict, protocol: str, near_names: list[str], far_names: list[str], csid_names: list[str]) -> dict:
    if protocol == 'full_spectrum':
        id_items = [scored[('id', 'test')]] + [scored[('csid', name)] for name in csid_names]
        preds, confs, labels = zip(*id_items)
        id_tuple = (np.concatenate(preds), np.concatenate(confs), np.concatenate(labels))
    else:
        id_tuple = scored[('id', 'test')]

    score_split = lambda group, name: scored[(group, name)]
    near = openood_metric_array(id_tuple, 'ood_near', near_names, score_split)
    far = openood_metric_array(id_tuple, 'ood_far', far_names, score_split)
    overall = np.concatenate([near, far], axis=0).mean(axis=0)
    return {'near_auroc': float(near[:, 1].mean()), 'far_auroc': float(far[:, 1].mean()), 'overall_auroc': float(overall[1]), 'overall_fpr95': float(overall[0])}


def validation_metrics(scores: np.ndarray, preds: np.ndarray, method: str, t2: float, stats: dict) -> dict:
    if method == 'TSS':
        z = (scores - stats[t2]['global_mu']) / max(stats[t2]['global_sigma'], EPS)
    else:
        z = (scores - stats[t2]['class_mu'][preds]) / np.maximum(stats[t2]['class_sigma'][preds], EPS)
    z = np.asarray(z, dtype=np.float32)
    return {
        'val_z_mean': float(z.mean()),
        'val_z_std': float(z.std(ddof=0)),
        'val_z_abs_mean': float(np.abs(z).mean()),
        'val_z_abs95': float(np.quantile(np.abs(z), 0.95)),
        'id_stability_score': float(abs(z.mean()) + abs(z.std(ddof=0) - 1.0)),
    }


def cache_records(cache_root, metadata: dict, t2_values: list[float]) -> list[dict]:
    dataset = metadata['dataset']
    model_cache = ModelCache(cache_root)
    splits = model_cache.metadata['splits']
    near_names = list(splits.get('ood_near', []))
    far_names = list(splits.get('ood_far', []))
    csid_names = list(splits.get('csid', []))
    if dataset == 'imagenet':
        csid_names = [name for name in csid_names if name != 'imagenet_es']
    protocols = ['standard'] if dataset == 'cifar100' else ['standard', 'full_spectrum']
    split_groups = [('id', 'test'), *[('csid', name) for name in csid_names],
                    *[('ood_near', name) for name in near_names], *[('ood_far', name) for name in far_names]]

    stats = fit_tss_statistics(model_cache.iter_array('id', 'train', 'logits'), t2_values)
    states = {t2: (stats[t2]['class_mu'], stats[t2]['class_sigma']) for t2 in t2_values}
    val_logits, val_preds, _ = model_cache.get_arrays('id', 'val', 'logits', 'preds', 'labels')
    rows = []

    for t2 in t2_values:
        val_scores = tss_from_logits(val_logits, t1=1.0, t2=t2)
        for method in ['TSS', 'rTSS']:
            base = {'dataset': dataset, 'cache_key': cache_root.name, 'checkpoint_path': metadata.get('checkpoint_path'), 'method': method, 't1': 1.0, 't2': t2}
            rows.append({**base, 'table': 'validation', **validation_metrics(val_scores, val_preds, method, t2, stats)})
            scored = score_cache(model_cache, split_groups, method, t2, states)
            for protocol in protocols:
                rows.append({**base, 'table': 'performance', 'protocol': protocol, **summarize_performance(scored, protocol, near_names, far_names, csid_names)})

    return rows


def summarize_table(raw: pd.DataFrame, keys: list[str], metrics: dict) -> pd.DataFrame:
    return raw.groupby(keys, as_index=False).agg(**metrics, n=('cache_key', 'nunique'))


def write_outputs(records: list[dict]) -> None:
    raw = pd.DataFrame.from_records(records)
    performance = raw[raw['table'] == 'performance'].drop(columns='table')
    validation = raw[raw['table'] == 'validation'].drop(columns='table')
    write_artifacts(OUTPUT_DIR, {
        'raw_metrics': performance,
        'summary': summarize_table(performance, ['dataset', 'protocol', 'method', 't2'], {
        'near_auroc': ('near_auroc', 'mean'),
        'far_auroc': ('far_auroc', 'mean'),
        'overall_auroc': ('overall_auroc', 'mean'),
        'overall_fpr95': ('overall_fpr95', 'mean'),
        'near_std': ('near_auroc', 'std'),
        'overall_std': ('overall_auroc', 'std'),
    }),
        'id_validation_raw': validation,
        'id_validation_summary': summarize_table(validation, ['dataset', 'method', 't2'], {
        'val_z_mean': ('val_z_mean', 'mean'),
        'val_z_std': ('val_z_std', 'mean'),
        'val_z_abs_mean': ('val_z_abs_mean', 'mean'),
        'val_z_abs95': ('val_z_abs95', 'mean'),
        'id_stability_score': ('id_stability_score', 'mean'),
    }),
    })


def main() -> None:
    records = []
    for cache_root, metadata in iter_model_caches(CACHE_DIR, datasets=DATASETS):
        print(f'Sweeping TSS temperatures for {metadata["dataset"]} cache={cache_root.name}', flush=True)
        records.extend(cache_records(cache_root, metadata, T2_VALUES))

    write_outputs(records)
    print(f'Wrote {OUTPUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
