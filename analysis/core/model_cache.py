"""Read cached model outputs and materialize missing splits on demand.

This module is the on-disk cache layer. Each `(group, split_name)` pair (e.g.
`('id', 'test')`, `('ood_near', 'ssb_hard')`) maps to a directory of `.npy`
files: `logits.npy`, `features.npy`, `labels.npy`, `preds.npy`. The `ModelCache`
class is both reader and writer; pass `net=None` for a read-only handle, or
pass `net` + `dataloader_dict` + `metadata` for a handle that can compute and
persist missing splits.

The module also exposes lower-level helpers that read individual split files
without forcing callers to load full `ModelOutputs`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

CACHE_INDEX_NAME = 'cache_index.parquet'

@dataclass(frozen=True)
class ModelOutputs:
    """Arrays produced by one model forward pass over one logical split.

    `logits` / `features` / `labels` / `preds` are required and always populated.
    """

    logits: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    preds: np.ndarray


def _flatten_to_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce a 4D conv feature map to a 2D `[batch, channels]` matrix.

    Already-2D tensors (e.g. penultimate features after average pooling)
    pass through with only a reshape. 4D tensors get adaptive average pooling
    so every split's features have a uniform 2D shape on disk.
    """
    if tensor.ndim <= 2:
        return tensor.reshape(tensor.shape[0], -1)
    pooled = torch.nn.functional.adaptive_avg_pool2d(tensor, output_size=1)
    return pooled.reshape(pooled.shape[0], -1)


def _split_path(cache_root: Path, group: str, name: str) -> Path:
    """Return the canonical on-disk directory for one cache split."""
    return cache_root / f'{group}__{name}'


def read_manifest(cache_root: Path) -> dict:
    """Read the model-cache manifest written beside split arrays."""
    manifest_path = cache_root / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Missing cache manifest: {manifest_path}')
    return json.loads(manifest_path.read_text(encoding='utf-8'))


def write_cache_index(records: list[dict], cache_dir: Path) -> Path:
    """Write the sorted step00 model-cache index parquet and return its path."""
    path = Path(cache_dir) / CACHE_INDEX_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame = frame.sort_values(['dataset', 'network', 'checkpoint_name']).reset_index(drop=True)
    frame.to_parquet(path, index=False)
    return path


def read_cache_index(cache_dir: Path) -> pd.DataFrame:
    """Read the step00 model-cache index parquet, erroring if missing or empty."""
    path = Path(cache_dir) / CACHE_INDEX_NAME
    if not path.is_file():
        raise FileNotFoundError(f'Missing model cache index: {path}. Run step00_build_cache first.')
    frame = pd.read_parquet(path)
    if frame.empty:
        raise RuntimeError(f'Model cache index has no entries: {path}')
    return frame.reset_index(drop=True)


def iter_model_caches(cache_dir: Path, datasets: set[str] | None = None, max_per_dataset: int | None = None):
    """Yield cache roots and metadata from the step00 cache index."""
    datasets = set(datasets) if datasets is not None else None
    frame = read_cache_index(cache_dir)
    if datasets is not None:
        frame = frame[frame['dataset'].isin(datasets)].copy()
    counts = {}
    for metadata in frame.to_dict('records'):
        dataset = metadata['dataset']
        if max_per_dataset is not None:
            counts[dataset] = counts.get(dataset, 0) + 1
            if counts[dataset] > max_per_dataset:
                continue
        yield Path(metadata['cache_root']), metadata


def iter_dataset_model_caches(cache_dir: Path, dataset_name: str):
    """Yield checkpoint name, checkpoint path, and read-only cache for one dataset."""
    for cache_root, metadata in iter_model_caches(cache_dir, datasets={dataset_name}):
        yield metadata['checkpoint_name'], Path(metadata['checkpoint_path']), ModelCache(cache_root)


def _cache_fingerprint(metadata: dict) -> str:
    """Hash the cache-identity fields into a deterministic 16-char suffix.

    The fingerprint depends on checkpoint path + file size + mtime + dataset +
    network — enough so that retraining the same file (changing its size or
    mtime) produces a different cache directory and avoids reusing stale
    logits or features.
    """
    identity = {
        'checkpoint_path': metadata['checkpoint_path'],
        'checkpoint_size': metadata['checkpoint_size'],
        'checkpoint_mtime_ns': metadata['checkpoint_mtime_ns'],
        'dataset': metadata['dataset'],
        'network': metadata['network'],
    }
    payload = json.dumps(identity, sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


def cache_root_name(metadata: dict) -> Path:
    """Build the canonical cache path suffix for one checkpoint/dataset/network."""
    return Path(str(metadata['dataset'])) / str(metadata['network']) / f'{metadata["checkpoint_name"]}__{_cache_fingerprint(metadata)}'


def write_array_dir_atomic(path: Path, **arrays: np.ndarray) -> None:
    """Write a directory of `.npy` arrays via a temporary directory swap."""
    temp_path = path.with_name(f'{path.name}.tmp')
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(temp_path / f'{name}.npy', array)
    if path.exists():
        shutil.rmtree(path)
    temp_path.replace(path)


class ModelCache:
    """Read cached model outputs and (optionally) materialize missing splits.

    Operates in one of two modes depending on the constructor arguments:

    - **Read-only** (`net=None`, `dataloader_dict=None`, `metadata=None`):
      opens an existing cache and serves arrays from disk. `get_outputs` for
      a missing split raises `FileNotFoundError` because there is no way to
      compute it.
    - **Read+write** (all three provided): writes the manifest and FC weights
      up front, then computes any missing split on demand by running the
      network over the matching dataloader.

    Either way the FC classifier arrays are loaded into both numpy and torch
    form so adapters that need them (ReAct / VIM / ASH / DICE) don't need to
    hold a reference to the live network.
    """

    def __init__(self, cache_root: Path | str, net: Any | None = None, dataloader_dict: dict | None = None, metadata: dict | None = None) -> None:
        """Open an existing cache, or initialize a writable one with `metadata`.

        The `net is not None and metadata is not None` path is the writable
        case used by step00. Other consumers pass only `cache_root` to get a
        read-only handle.
        """
        # ── Bind paths + the (optional) network/dataloader the writer mode needs ──
        self.cache_root = Path(cache_root)
        self.manifest_path = self.cache_root / 'manifest.json'
        self.fc_weight_path = self.cache_root / 'fc_weight.npy'
        self.fc_bias_path = self.cache_root / 'fc_bias.npy'
        self.net = net
        self.dataloader_dict = dataloader_dict

        # ── Writer-mode init: create the cache dir, write the manifest, and
        # snapshot FC weights to disk so later read-only handles never need
        # the live network just to access the classifier head. ──
        if metadata is not None:
            self.cache_root.mkdir(parents=True, exist_ok=True)
            with self.manifest_path.open('w', encoding='utf-8') as handle:
                json.dump(metadata, handle, indent=2)
            if self.net is not None and hasattr(self.net, 'get_fc'):
                fc_weight, fc_bias = self.net.get_fc()
                np.save(self.fc_weight_path, fc_weight.astype(np.float32, copy=False))
                np.save(self.fc_bias_path, fc_bias.astype(np.float32, copy=False))

        # ── Load manifest + classifier arrays (both reader and writer modes).
        # Cached FC weights are kept in numpy AND torch form so adapters that
        # need either can grab the right one without an extra conversion. ──
        self.metadata = read_manifest(self.cache_root)
        self.fc_weight = None
        self.fc_bias = None
        self.fc_weight_t = None
        self.fc_bias_t = None
        if self.fc_weight_path.is_file() and self.fc_bias_path.is_file():
            self.fc_weight = np.load(self.fc_weight_path, allow_pickle=False).astype(np.float32, copy=False)
            self.fc_bias = np.load(self.fc_bias_path, allow_pickle=False).astype(np.float32, copy=False)
            self.fc_weight_t = torch.from_numpy(self.fc_weight).float()
            self.fc_bias_t = torch.from_numpy(self.fc_bias).float()

    def get_loader(self, group: str, split_name: str):
        """Return the OpenOOD dataloader for a logical cache split."""
        if self.dataloader_dict is None:
            raise FileNotFoundError(f'Missing cached split: {group}/{split_name} in {self.cache_root}')
        if group == 'id':
            return self.dataloader_dict['id'][split_name]
        if group == 'csid':
            return self.dataloader_dict['csid'][split_name]
        if group == 'ood_near':
            return self.dataloader_dict['ood']['near'][split_name]
        if group == 'ood_far':
            return self.dataloader_dict['ood']['far'][split_name]
        raise KeyError(f'Unknown split group {group}')

    def _read_outputs(self, cache_path: Path) -> ModelOutputs:
        """Load all four cached arrays for one split and assemble a `ModelOutputs`."""
        return ModelOutputs(
            logits=np.load(cache_path / 'logits.npy', allow_pickle=False),
            features=np.load(cache_path / 'features.npy', allow_pickle=False),
            labels=np.load(cache_path / 'labels.npy', allow_pickle=False).astype(np.int64, copy=False),
            preds=np.load(cache_path / 'preds.npy', allow_pickle=False).astype(np.int64, copy=False),
        )

    def _compute_outputs(self, group: str, split_name: str) -> ModelOutputs:
        """Run the network over one split's dataloader and assemble outputs.

        Iterates the dataloader once, collecting per-batch logits, features,
        labels, and preds, then concatenates them into one `ModelOutputs`.
        Caller (`get_outputs`) is responsible for persisting the result.
        """
        loader = self.get_loader(group, split_name)
        total_batches = len(loader) if hasattr(loader, '__len__') else None
        cache_path = _split_path(self.cache_root, group, split_name)
        print(f'[model-cache] computing {group}/{split_name} -> {cache_path}', flush=True)

        # ── Per-batch loop: forward pass + accumulate four parallel lists ──
        logits_parts = []
        feature_parts = []
        label_parts = []
        pred_parts = []

        with torch.no_grad():
            progress_bar = tqdm(loader, total=total_batches, desc=f'[cache] {group}/{split_name}', unit='batch', dynamic_ncols=False, ncols=100)
            for batch in progress_bar:
                data = batch['data'].cuda(non_blocking=True).float()
                labels = batch['label'].cpu().numpy()
                logits, features = self.net(data, return_feature=True)
                logits = logits.detach()
                logits_parts.append(logits.cpu().numpy())
                feature_parts.append(_flatten_to_matrix(features.detach()).cpu().numpy())
                label_parts.append(labels)
                pred_parts.append(logits.argmax(dim=1).cpu().numpy())

        # ── Concatenate parts into one `ModelOutputs` for the entire split ──
        logits = np.concatenate(logits_parts, axis=0).astype(np.float32, copy=False)
        features = np.concatenate(feature_parts, axis=0).astype(np.float32, copy=False)
        return ModelOutputs(
            logits=logits,
            features=features,
            labels=np.concatenate(label_parts, axis=0).astype(np.int64, copy=False),
            preds=np.concatenate(pred_parts, axis=0).astype(np.int64, copy=False),
        )

    def get_outputs(self, group: str, split_name: str) -> ModelOutputs:
        """Return one split's outputs, reading from disk or computing on miss.

        Behavior depends on cache state and writer mode:

        - **Hit on disk** → read the four `.npy` files and return.
        - **Miss + reader mode** (`net is None`) → raise `FileNotFoundError`
          because there's no way to compute the missing arrays.
        - **Miss + writer mode** → run the network over the matching
          dataloader, atomically write the four `.npy` files, and return.
        """
        cache_path = _split_path(self.cache_root, group, split_name)
        if cache_path.is_dir():
            if self.net is not None:
                print(f'[model-cache] hit {group}/{split_name}', flush=True)
            return self._read_outputs(cache_path)
        if self.net is None:
            raise FileNotFoundError(f'Missing cached split: {group}/{split_name} in {self.cache_root}')
        outputs = self._compute_outputs(group, split_name)
        write_array_dir_atomic(
            cache_path,
            logits=outputs.logits,
            features=outputs.features,
            labels=outputs.labels,
            preds=outputs.preds,
        )
        print(f'[model-cache] saved {group}/{split_name}', flush=True)
        return outputs

    def get_arrays(self, group: str, split_name: str, *array_names: str) -> tuple[np.ndarray, ...]:
        """Load named arrays from one cached split."""
        path = _split_path(self.cache_root, group, split_name)
        return tuple(np.load(path / f'{array_name}.npy', allow_pickle=False) for array_name in array_names)

    def iter_array(self, group: str, split_name: str, array_name: str, rows_per_chunk: int = 8192):
        """Stream one named split array."""
        array = np.load(_split_path(self.cache_root, group, split_name) / f'{array_name}.npy', mmap_mode='r', allow_pickle=False)
        for start in range(0, array.shape[0], max(1, rows_per_chunk)):
            yield np.asarray(array[start:start + rows_per_chunk])
