"""Temperature-scaled score and residual-TSS helper functions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Iterable

import numpy as np

EPS = 1e-6
METHOD_NAME = 'rtss'
METHOD_LABEL = 'rTSS'


@dataclass(frozen=True)
class RTSSState:
    """Train-derived residual statistics for rTSS scoring."""

    t1: float
    t2: float
    class_mu: np.ndarray
    class_sigma: np.ndarray


def tss_from_logits(logits: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """Compute the raw temperature-scaled score difference from logits."""
    logits = np.asarray(logits, dtype=np.float32)
    # TSS is the difference between two temperature-scaled log-sum-exp scores.
    scaled_t1 = logits / float(t1)
    scaled_t2 = logits / float(t2)
    e1 = float(t1) * np.logaddexp.reduce(scaled_t1, axis=1)
    e2 = float(t2) * np.logaddexp.reduce(scaled_t2, axis=1)
    return (e1 - e2).astype(np.float32, copy=False)


def fit_state(train_logits: np.ndarray, train_preds: np.ndarray, num_classes: int | None = None, t1: float = 1.0, t2: float = 0.5) -> RTSSState:
    """Fit rTSS class residual statistics from all train logits at once."""
    train_preds = np.asarray(train_preds, dtype=np.int64)
    train_tss = tss_from_logits(train_logits, t1=t1, t2=t2)
    if num_classes is None:
        num_classes = int(train_preds.max()) + 1
    num_classes = int(num_classes)
    global_std = float(train_tss.std(ddof=1)) if train_tss.size > 1 else 1.0
    sigma_floor = max(global_std * 1e-3, EPS)

    # Classes with too few predicted samples fall back to a global scale so
    # residual z-scores stay finite.
    class_mu = np.zeros(num_classes, dtype=np.float32)
    class_sigma = np.full(num_classes, global_std, dtype=np.float32)
    for class_index in range(num_classes):
        class_values = train_tss[train_preds == class_index]
        if class_values.size >= 2:
            class_mu[class_index] = np.float32(class_values.mean())
            class_sigma[class_index] = np.float32(max(class_values.std(ddof=1), sigma_floor))
        elif class_values.size == 1:
            class_mu[class_index] = np.float32(class_values[0])
            class_sigma[class_index] = np.float32(sigma_floor)
    return RTSSState(t1=float(t1), t2=float(t2), class_mu=class_mu, class_sigma=class_sigma)


def fit_tss_statistics(logit_chunks: Iterable[np.ndarray], t2_values: Iterable[float], t1: float = 1.0) -> dict[float, dict[str, np.ndarray | float]]:
    """Stream train logits and fit TSS statistics for several t2 values."""
    chunks = iter(logit_chunks)
    try:
        first_chunk = np.asarray(next(chunks), dtype=np.float32)
    except StopIteration as exc:
        raise ValueError('Expected at least one train-logit chunk') from exc

    t2_values = [float(value) for value in t2_values]
    num_classes = first_chunk.shape[1]
    counts = np.zeros(num_classes, dtype=np.float64)
    global_count = 0.0
    global_sum = {t2: 0.0 for t2 in t2_values}
    global_sumsq = {t2: 0.0 for t2 in t2_values}
    class_sum = {t2: np.zeros(num_classes, dtype=np.float64) for t2 in t2_values}
    class_sumsq = {t2: np.zeros(num_classes, dtype=np.float64) for t2 in t2_values}

    for logits in chain([first_chunk], chunks):
        logits = np.asarray(logits, dtype=np.float32)
        preds = logits.argmax(axis=1)
        counts += np.bincount(preds, minlength=num_classes)
        global_count += len(logits)
        for t2 in t2_values:
            scores = tss_from_logits(logits, t1=t1, t2=t2).astype(np.float64, copy=False)
            global_sum[t2] += float(scores.sum())
            global_sumsq[t2] += float((scores * scores).sum())
            class_sum[t2] += np.bincount(preds, weights=scores, minlength=num_classes)
            class_sumsq[t2] += np.bincount(preds, weights=scores * scores, minlength=num_classes)

    stats = {}
    present = counts > 0
    enough = counts >= 2
    for t2 in t2_values:
        mean = global_sum[t2] / global_count
        variance = (global_sumsq[t2] - global_sum[t2] * global_sum[t2] / global_count) / max(global_count - 1.0, 1.0)
        std = float(np.sqrt(max(variance, EPS)))
        sigma_floor = max(std * 1e-3, EPS)

        class_mu = np.zeros(num_classes, dtype=np.float32)
        class_sigma = np.full(num_classes, std, dtype=np.float32)
        class_mu[present] = (class_sum[t2][present] / counts[present]).astype(np.float32, copy=False)
        class_var = (class_sumsq[t2][enough] - class_sum[t2][enough] * class_sum[t2][enough] / counts[enough]) / (counts[enough] - 1.0)
        class_sigma[enough] = np.maximum(np.sqrt(np.maximum(class_var, 0.0)), sigma_floor).astype(np.float32, copy=False)
        class_sigma[present & ~enough] = np.float32(sigma_floor)
        stats[t2] = {'global_mu': float(mean), 'global_sigma': std, 'class_mu': class_mu, 'class_sigma': class_sigma}
    return stats


def score_outputs(state: RTSSState, logits: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Score outputs by negative predicted-class TSS residual z-score."""
    preds = np.asarray(preds, dtype=np.int64)
    tss = tss_from_logits(logits, t1=state.t1, t2=state.t2)
    residual = tss - state.class_mu[preds]
    residual_sigma = residual / np.maximum(state.class_sigma[preds], EPS)
    return -residual_sigma.astype(np.float32, copy=False)
