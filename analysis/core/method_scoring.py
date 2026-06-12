"""Evaluate OOD methods against cached inference outputs."""

from __future__ import annotations

import numpy as np

from .model_cache import ModelCache, write_array_dir_atomic
from .methods.adapters import build_method_adapter
from .methods.registry import MethodParams


class CachedMethodRunner:
    """Run one detector method against cached model outputs.

    The runner owns two caches: setup state derived from train/val splits and
    per-split detector scores. Adapters provide the method-specific math.
    """

    def __init__(self, method_spec: dict[str, str], params: MethodParams, model_cache: ModelCache):
        """Create the method adapter and load or fit its setup state."""
        self.params = params
        self.model_cache = model_cache
        self.score_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        self.name = method_spec['postprocessor']
        self.adapter = build_method_adapter(self)
        if not self._load_setup_cache():
            self.adapter.build_setup()

    def _setup_cache_path(self):
        cache_key = self.adapter.setup_cache_key()
        if cache_key is None:
            return None
        return self.model_cache.cache_root / 'method_setup' / f'{self.name}__{cache_key}'

    def _load_setup_cache(self):
        cache_path = self._setup_cache_path()
        if cache_path is None or not cache_path.is_dir():
            return False

        data = {path.stem: np.load(path, allow_pickle=False) for path in cache_path.glob('*.npy')}
        if not self.adapter.load_setup_cache(data):
            return False
        return True

    def _save_setup_cache(self, **arrays: np.ndarray):
        cache_path = self._setup_cache_path()
        if cache_path is None:
            return
        write_array_dir_atomic(cache_path, **arrays)

    def _score_cache_path(self, group: str, name: str):
        return self.model_cache.cache_root / 'method_scores' / f'{self.adapter.score_cache_key()}__{group}__{name}'

    def score_split(self, group: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return predictions, confidence scores, and labels for one split."""
        key = (group, name)
        cached = self.score_cache.get(key)
        if cached is not None:
            return cached

        cache_path = self._score_cache_path(group, name)
        if cache_path.is_dir():
            result = (
                np.load(cache_path / 'preds.npy', allow_pickle=False).astype(np.int64, copy=False),
                np.load(cache_path / 'conf.npy', allow_pickle=False).astype(np.float32, copy=False),
                np.load(cache_path / 'labels.npy', allow_pickle=False).astype(np.int64, copy=False),
            )
            self.score_cache[key] = result
            return result

        outputs = self.model_cache.get_outputs(group, name)
        preds, conf = self.adapter.score_outputs(outputs)
        result = (preds.astype(np.int64, copy=False), conf.astype(np.float32, copy=False), outputs.labels)
        self.score_cache[key] = result
        write_array_dir_atomic(cache_path, preds=result[0], conf=result[1], labels=result[2].astype(np.int64, copy=False))
        return result
