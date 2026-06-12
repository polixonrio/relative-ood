"""Dispatch cached evaluation methods to their adapter implementations."""

from __future__ import annotations

from .base import CachedMethodAdapter
from .distance import KnnAdapter, LogitMahalanobisAdapter, MahalanobisAdapter, RknnPlusPlusAdapter
from .feature_activation import AshAdapter, DiceAdapter, ReactAdapter, ScaleAdapter, VimAdapter
from .score_only import EboAdapter, GenAdapter, GradNormAdapter, MlsAdapter, MspAdapter, RtssAdapter, TssAdapter

ADAPTER_REGISTRY = {
    # Registry keys are postprocessor names, not display labels.
    'msp': MspAdapter,
    'mls': MlsAdapter,
    'ebo': EboAdapter,
    'gen': GenAdapter,
    'tss_raw': TssAdapter,
    'rtss': RtssAdapter,
    'knn': KnnAdapter,
    'rknnpp': RknnPlusPlusAdapter,
    'mds': MahalanobisAdapter,
    'mdspp': MahalanobisAdapter,
    'rmds': MahalanobisAdapter,
    'rmdspp': MahalanobisAdapter,
    'lsm': LogitMahalanobisAdapter,
    'lsmpp': LogitMahalanobisAdapter,
    'rlsm': LogitMahalanobisAdapter,
    'rlsmpp': LogitMahalanobisAdapter,
    'vim': VimAdapter,
    'react': ReactAdapter,
    'ash': AshAdapter,
    'dice': DiceAdapter,
    'scale': ScaleAdapter,
    'gradnorm': GradNormAdapter,
}


def build_method_adapter(runner):
    """Instantiate the cached adapter registered for the runner's method."""
    try:
        adapter_cls = ADAPTER_REGISTRY[runner.name]
    except KeyError as exc:
        raise NotImplementedError(f'Cached method adapter not implemented for {runner.name}') from exc
    return adapter_cls(runner)
