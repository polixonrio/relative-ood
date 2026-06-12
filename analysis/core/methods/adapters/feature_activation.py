"""Cached adapters for feature-activation methods such as VIM, ReAct, ASH, and DICE."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
import torch
from scipy.special import logsumexp

from .base import CachedMethodAdapter


def ash_b_transform(x: torch.Tensor, percentile: int) -> torch.Tensor:
    """Apply ASH-B activation shaping to a batch of final features."""
    # Same transform as OpenOOD's ash_b, applied to cached final features.
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    batch_size, channels, height, width = x.shape
    sample_sum = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    flat = x.view((batch_size, channels * height * width))
    _, indices = torch.topk(flat, k, dim=1)
    fill = (sample_sum / k).unsqueeze(dim=1).expand(batch_size, k)
    flat.zero_().scatter_(dim=1, index=indices, src=fill)
    return x


class VimAdapter(CachedMethodAdapter):
    """VIM scoring from cached final features and train statistics."""

    @property
    def dim(self) -> int:
        """Return the principal subspace dimension removed by VIM."""
        return int(self.params.get('dim', 256))

    def setup_cache_key(self) -> str:
        """Include VIM dimension in the persisted setup-cache key."""
        return f'dim{self.dim}'

    def score_cache_key(self) -> str:
        """Include VIM dimension in the persisted score-cache key."""
        return f'{self.name}_dim{self.dim}'

    def load_setup_cache(self, data) -> bool:
        """Load persisted VIM eigenspace and calibration state."""
        self.w = self.runner.model_cache.fc_weight
        self.b = self.runner.model_cache.fc_bias
        self.u = data['u'].astype(np.float32, copy=False)
        self.eigen_vectors = data['eigen_vectors'].astype(np.float32, copy=False)
        self.eig_vals = data['eig_vals'].astype(np.float32, copy=False)
        self.NS = data['NS'].astype(np.float32, copy=False)
        self.alpha = float(data['alpha'])
        return True

    def build_setup(self) -> None:
        """Fit VIM null-space and alpha calibration from train outputs."""
        self.w = self.runner.model_cache.fc_weight
        self.b = self.runner.model_cache.fc_bias
        self.u = -np.matmul(np.linalg.pinv(self.w), self.b)

        # VIM needs train covariance and average max logit; stream both so large
        # ImageNet caches do not need to fit in memory.
        n_samples = 0
        covariance = np.zeros((self.w.shape[1], self.w.shape[1]), dtype=np.float64)
        max_logit_sum = 0.0
        for features in self.runner.model_cache.iter_array('id', 'train', 'features'):
            features = features.astype(np.float64, copy=False)
            centered = features - self.u
            covariance += centered.T @ centered
            n_samples += features.shape[0]
        for logits in self.runner.model_cache.iter_array('id', 'train', 'logits'):
            max_logit_sum += float(logits.max(axis=1).sum())
        covariance /= float(n_samples)
        eig_vals, eigen_vectors = eigh(covariance, check_finite=False)

        self.eigen_vectors = eigen_vectors
        self.eig_vals = eig_vals
        self.NS = np.ascontiguousarray((self.eigen_vectors.T[np.argsort(self.eig_vals * -1)[self.dim:]]).T)

        residual_norm_sum = 0.0
        for features in self.runner.model_cache.iter_array('id', 'train', 'features'):
            features = features.astype(np.float64, copy=False)
            residual_norm_sum += float(np.linalg.norm(np.matmul(features - self.u, self.NS), axis=-1).sum())
        self.alpha = (max_logit_sum / float(n_samples)) / (residual_norm_sum / float(n_samples))
        self.runner._save_setup_cache(
            u=self.u.astype(np.float32, copy=False),
            eigen_vectors=np.asarray(self.eigen_vectors, dtype=np.float32),
            eig_vals=np.asarray(self.eig_vals, dtype=np.float32),
            NS=np.asarray(self.NS, dtype=np.float32),
            alpha=np.asarray(self.alpha, dtype=np.float32),
        )

    def score_outputs(self, outputs):
        """Return VIM confidence for one cached split."""
        energy = logsumexp(outputs.logits, axis=-1)
        vlogit = np.linalg.norm(np.matmul(outputs.features - self.u, self.NS), axis=-1) * self.alpha
        conf = -vlogit + energy
        return outputs.preds, conf.astype(np.float32, copy=False)


class ReactAdapter(CachedMethodAdapter):
    """ReAct feature clipping followed by classifier energy scoring."""

    @property
    def percentile(self) -> int:
        """Return the validation-feature clipping percentile."""
        checkpoint_name = str(self.runner.model_cache.metadata.get('checkpoint_name', ''))
        return int(self.params.get_for_checkpoint(checkpoint_name, 'percentile', 90))

    def score_cache_key(self) -> str:
        """Include clipping percentile in the persisted score-cache key."""
        return f'{self.name}_p{self.percentile}'

    def build_setup(self) -> None:
        """Fit the ReAct clipping threshold on ID-validation features."""
        # ReAct clips features at an ID-validation percentile before applying
        # the final classifier.
        val = self.runner.model_cache.get_outputs('id', 'val')
        self.threshold = np.percentile(val.features.flatten(), self.percentile)

    def score_outputs(self, outputs):
        """Return ReAct confidence for one cached split."""
        clipped = np.clip(outputs.features, a_min=None, a_max=self.threshold)
        logits = clipped @ self.runner.model_cache.fc_weight.T + self.runner.model_cache.fc_bias
        conf = logsumexp(logits, axis=1)
        preds = logits.argmax(axis=1).astype(np.int64, copy=False)
        return preds, conf.astype(np.float32, copy=False)


class AshAdapter(CachedMethodAdapter):
    """ASH-B activation shaping followed by classifier energy scoring."""

    @property
    def percentile(self) -> int:
        """Return the ASH pruning percentile."""
        checkpoint_name = str(self.runner.model_cache.metadata.get('checkpoint_name', ''))
        return int(self.params.get_for_checkpoint(checkpoint_name, 'percentile', 90))

    def score_cache_key(self) -> str:
        """Include ASH percentile in the persisted score-cache key."""
        return f'{self.name}_p{self.percentile}'

    def score_outputs(self, outputs):
        """Return ASH confidence for one cached split."""
        feature_tensor = torch.from_numpy(outputs.features.copy()).float().view(outputs.features.shape[0], outputs.features.shape[1], 1, 1)
        transformed = ash_b_transform(feature_tensor, self.percentile)
        transformed = transformed.view(outputs.features.shape[0], outputs.features.shape[1])
        logits = transformed @ self.runner.model_cache.fc_weight_t.T + self.runner.model_cache.fc_bias_t
        logits_np = logits.numpy()
        conf = logsumexp(logits_np, axis=1)
        preds = logits_np.argmax(axis=1).astype(np.int64, copy=False)
        return preds, conf.astype(np.float32, copy=False)


class ScaleAdapter(CachedMethodAdapter):
    """SCALE feature rescaling followed by classifier energy scoring."""

    @property
    def percentile(self) -> int:
        """Return the percentile used to keep high-magnitude activations."""
        return int(self.params.get('percentile', 85))

    def score_cache_key(self) -> str:
        """Include SCALE percentile in the persisted score-cache key."""
        return f'{self.name}_p{self.percentile}'

    def score_outputs(self, outputs):
        """Return OpenOOD SCALE confidence for one cached split."""
        features = outputs.features.astype(np.float32, copy=True)
        assert 0 <= self.percentile <= 100
        k = features.shape[1] - int(np.round(features.shape[1] * self.percentile / 100.0))
        k = max(k, 1)
        topk = np.partition(features, features.shape[1] - k, axis=1)[:, -k:]
        original_sum = features.sum(axis=1)
        topk_sum = topk.sum(axis=1)
        scale = np.divide(original_sum, topk_sum, out=np.ones_like(original_sum), where=np.abs(topk_sum) > 1e-12)
        transformed = features * np.exp(scale)[:, None]
        logits = transformed @ self.runner.model_cache.fc_weight.T + self.runner.model_cache.fc_bias
        conf = logsumexp(logits, axis=1)
        preds = logits.argmax(axis=1).astype(np.int64, copy=False)
        return preds, conf.astype(np.float32, copy=False)


class DiceAdapter(CachedMethodAdapter):
    """DICE masked-classifier scoring from cached final features."""

    @property
    def percentile(self) -> int:
        """Return the contribution percentile used to mask classifier weights."""
        return int(self.params.get('p', 90))

    def score_cache_key(self) -> str:
        """Include DICE percentile in the persisted score-cache key."""
        return f'{self.name}_p{self.percentile}'

    def build_setup(self) -> None:
        """Fit the DICE classifier-weight mask from train feature means."""
        feature_sum = None
        count = 0
        # DICE keeps weights whose mean activation contribution exceeds the
        # configured percentile threshold.
        for features in self.runner.model_cache.iter_array('id', 'train', 'features'):
            feature_sum = features.sum(axis=0, dtype=np.float64) if feature_sum is None else feature_sum + features.sum(axis=0, dtype=np.float64)
            count += features.shape[0]
        mean_act = feature_sum / float(count)
        contrib = mean_act[None, :] * self.runner.model_cache.fc_weight
        thresh = np.percentile(contrib, self.percentile)
        mask = (contrib > thresh).astype(np.float32)
        self.masked_w = self.runner.model_cache.fc_weight * mask

    def score_outputs(self, outputs):
        """Return DICE confidence for one cached split."""
        logits = outputs.features @ self.masked_w.T + self.runner.model_cache.fc_bias
        conf = logsumexp(logits, axis=1)
        preds = logits.argmax(axis=1).astype(np.int64, copy=False)
        return preds, conf.astype(np.float32, copy=False)
