"""Cached adapters for methods that score directly from logits or final features."""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import logsumexp

from ..rtss import EPS, RTSSState, score_outputs as score_rtss_outputs, tss_from_logits

from .base import CachedMethodAdapter


class MspAdapter(CachedMethodAdapter):
    """Maximum-softmax-probability scoring from cached logits."""

    def score_outputs(self, outputs):
        """Return MSP confidence for one cached split."""
        probs = torch.softmax(torch.from_numpy(outputs.logits).float(), dim=1)
        conf, _ = probs.max(dim=1)
        return outputs.preds, conf.numpy()


class MlsAdapter(CachedMethodAdapter):
    """Maximum-logit scoring from cached logits."""

    def score_outputs(self, outputs):
        """Return max-logit confidence for one cached split."""
        return outputs.preds, outputs.logits.max(axis=1)


class EboAdapter(CachedMethodAdapter):
    """Energy-based OOD scoring from cached logits."""

    @property
    def temperature(self) -> float:
        """Return the energy temperature loaded from method parameters."""
        return float(self.params.get('temperature', 1.0))

    def score_cache_key(self) -> str:
        """Include temperature in the persisted score-cache key."""
        return f'{self.name}_temp{self.temperature}'

    def score_outputs(self, outputs):
        """Return energy confidence for one cached split."""
        temperature = self.temperature
        conf = temperature * logsumexp(outputs.logits / temperature, axis=1)
        return outputs.preds, conf.astype(np.float32, copy=False)


class GenAdapter(CachedMethodAdapter):
    """Generalized-entropy scoring from cached logits."""

    @property
    def gamma(self) -> float:
        """Return the generalized-entropy exponent."""
        return float(self.params.get('gamma', 0.1))

    @property
    def M(self) -> int:
        """Return the number of largest class probabilities retained."""
        return int(self.params.get('M', 100))

    def score_cache_key(self) -> str:
        """Include GEN hyperparameters in the persisted score-cache key."""
        return f'{self.name}_gamma{self.gamma}_M{self.M}'

    def score_outputs(self, outputs):
        """Return OpenOOD GEN confidence for one cached split."""
        logits = outputs.logits.astype(np.float64, copy=False)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        probs_sorted = np.sort(probs, axis=1)[:, -self.M:]
        conf = -np.sum((probs_sorted**self.gamma) * ((1.0 - probs_sorted) ** self.gamma), axis=1)
        return outputs.preds, conf.astype(np.float32, copy=False)


class GradNormAdapter(CachedMethodAdapter):
    """Closed-form GradNorm confidence from cached logits and final features."""

    def score_outputs(self, outputs):
        """Return OpenOOD GradNorm's final-FC weight gradient norm."""
        # This is algebraically equivalent to OpenOOD's per-sample final-layer
        # GradNorm score, without replaying a backward pass for every sample.
        probs = torch.softmax(torch.from_numpy(outputs.logits).float(), dim=1)
        class_factor = torch.abs(probs * probs.shape[1] - 1.0).sum(dim=1).numpy()
        feature_factor = np.abs(outputs.features).sum(axis=1)
        conf = feature_factor * class_factor
        return outputs.preds, conf.astype(np.float32, copy=False)


class TssAdapter(CachedMethodAdapter):
    """Raw temperature-scaled score-difference adapter."""

    @property
    def t1(self) -> float:
        """Return the first TSS temperature."""
        return float(self.params.get('t1', 1.0))

    @property
    def t2(self) -> float:
        """Return the second TSS temperature."""
        return float(self.params.get('t2', 0.5))

    def score_cache_key(self) -> str:
        """Include temperatures in the persisted score-cache key."""
        return f'{self.name}_t1{self.t1}_t2{self.t2}'

    def score_outputs(self, outputs):
        """Return negative raw TSS confidence for one cached split."""
        conf = -tss_from_logits(outputs.logits, self.t1, self.t2)
        return outputs.preds, conf


class RtssAdapter(TssAdapter):
    """Residual TSS adapter with train-derived class statistics."""

    def setup_cache_key(self) -> str:
        """Include temperatures in the persisted rTSS setup-cache key."""
        return f't1{self.t1}_t2{self.t2}'

    def load_setup_cache(self, data) -> bool:
        """Load persisted class residual statistics for rTSS."""
        self.state = RTSSState(t1=float(data['t1']), t2=float(data['t2']), class_mu=data['class_mu'], class_sigma=data['class_sigma'])
        return True

    def build_setup(self) -> None:
        """Stream train logits and fit predicted-class TSS residual statistics."""
        num_classes = self.runner.model_cache.fc_weight.shape[0]
        counts = np.zeros(num_classes, dtype=np.float64)
        class_sum = np.zeros(num_classes, dtype=np.float64)
        class_sumsq = np.zeros(num_classes, dtype=np.float64)

        # Stream train logits to fit per-predicted-class TSS residual statistics.
        for logits in self.runner.model_cache.iter_array('id', 'train', 'logits'):
            preds = logits.argmax(axis=1)
            scores = tss_from_logits(logits, self.t1, self.t2).astype(np.float64, copy=False)
            counts += np.bincount(preds, minlength=num_classes)
            class_sum += np.bincount(preds, weights=scores, minlength=num_classes)
            class_sumsq += np.bincount(preds, weights=scores * scores, minlength=num_classes)

        total = counts.sum()
        global_sum = class_sum.sum()
        global_sumsq = class_sumsq.sum()
        global_var = (global_sumsq - global_sum * global_sum / total) / max(total - 1.0, 1.0)
        global_std = float(np.sqrt(max(global_var, EPS)))
        sigma_floor = max(global_std * 1e-3, EPS)

        class_mu = np.zeros(num_classes, dtype=np.float32)
        class_sigma = np.full(num_classes, global_std, dtype=np.float32)
        present = counts > 0
        class_mu[present] = (class_sum[present] / counts[present]).astype(np.float32, copy=False)
        enough = counts >= 2
        class_var = (class_sumsq[enough] - class_sum[enough] * class_sum[enough] / counts[enough]) / (counts[enough] - 1.0)
        class_sigma[enough] = np.maximum(np.sqrt(np.maximum(class_var, 0.0)), sigma_floor).astype(np.float32, copy=False)
        class_sigma[present & ~enough] = np.float32(sigma_floor)

        self.state = RTSSState(t1=self.t1, t2=self.t2, class_mu=class_mu, class_sigma=class_sigma)
        self.runner._save_setup_cache(t1=np.array(self.t1, dtype=np.float32), t2=np.array(self.t2, dtype=np.float32), class_mu=class_mu, class_sigma=class_sigma)

    def score_outputs(self, outputs):
        """Return rTSS confidence for one cached split."""
        conf = score_rtss_outputs(self.state, outputs.logits, outputs.preds)
        return outputs.preds, conf
