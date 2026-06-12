"""Cached adapters for kNN and Mahalanobis-style distance methods."""

from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from .base import CachedMethodAdapter, normalizer

MAHALANOBIS_SETTINGS = {
    'mds': {'normalize_features': False, 'relative': False},
    'mdspp': {'normalize_features': True, 'relative': False},
    'rmds': {'normalize_features': False, 'relative': True},
    'rmdspp': {'normalize_features': True, 'relative': True},
    'lsm': {'normalize_features': False, 'relative': False},
    'lsmpp': {'normalize_features': True, 'relative': False},
    'rlsm': {'normalize_features': False, 'relative': True},
    'rlsmpp': {'normalize_features': True, 'relative': True},
}

PRECISION_EIGEN_FLOOR = 1e-6
PRECISION_EIGEN_FLOOR_ABSOLUTE = 1e-12
PRECISION_CACHE_VERSION = 'eigfloor1e-6'
KNN_CACHE_VERSION = 'torchfp16d2'


def stable_precision(covariance: np.ndarray) -> np.ndarray:
    """Invert covariance with deterministic symmetric eigenvalue flooring."""
    symmetric = 0.5 * (covariance + covariance.T)
    dim = symmetric.shape[0]
    scale = float(np.trace(symmetric) / max(dim, 1))
    floor = max(scale * PRECISION_EIGEN_FLOOR, PRECISION_EIGEN_FLOOR_ABSOLUTE)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    except np.linalg.LinAlgError:
        regularized = symmetric + np.eye(dim, dtype=symmetric.dtype) * floor
        return np.linalg.pinv(regularized, hermitian=True)
    inverse = 1.0 / np.maximum(eigenvalues, floor)
    precision = (eigenvectors * inverse[None, :]) @ eigenvectors.T
    return 0.5 * (precision + precision.T)


class KnnAdapter(CachedMethodAdapter):
    """Global k-nearest-neighbor distance scoring in feature space."""

    default_chunk_size = 128

    @property
    def k(self) -> int:
        """Return the neighbor rank used as the distance score."""
        return int(self.params.get('K', 50))

    @property
    def chunk_size(self) -> int:
        """Return the query batch size for GPU feature search."""
        return int(self.params.get('chunk_size', self.default_chunk_size))

    def score_cache_key(self) -> str:
        """Include k in the persisted score-cache key."""
        return f'{self.name}_{KNN_CACHE_VERSION}_k{self.k}'

    def build_setup(self) -> None:
        """Normalize train features and keep them on GPU for exact search."""
        if not torch.cuda.is_available():
            raise RuntimeError('KNN requires CUDA for cached exact feature search')

        labels, = self.runner.model_cache.get_arrays('id', 'train', 'labels')
        first_chunk = next(self.runner.model_cache.iter_array('id', 'train', 'features', rows_per_chunk=1))
        train_norm = np.empty((labels.shape[0], first_chunk.shape[1]), dtype=np.float16)

        offset = 0
        print('[knn] normalizing train features', flush=True)
        for chunk in tqdm(self.runner.model_cache.iter_array('id', 'train', 'features'), desc='[knn] train', unit='chunk', dynamic_ncols=False, ncols=100):
            values = normalizer(chunk.astype(np.float32, copy=False))
            train_norm[offset:offset + len(values)] = values.astype(np.float16, copy=False)
            offset += len(values)
        if offset != labels.shape[0]:
            raise RuntimeError(f'Expected {labels.shape[0]} train features, read {offset}')

        self.train_gpu = torch.from_numpy(train_norm).cuda()
        del train_norm, labels
        print(f'[knn] train table on GPU: {tuple(self.train_gpu.shape)} float16', flush=True)

    def score_outputs(self, outputs):
        """Return negative kth-neighbor distance for one cached split."""
        features = normalizer(outputs.features.astype(np.float32, copy=False))
        conf = np.empty(len(features), dtype=np.float32)

        ranges = range(0, len(features), self.chunk_size)
        for start in tqdm(ranges, desc='[knn] score', unit='chunk', dynamic_ncols=False, ncols=100):
            stop = min(start + self.chunk_size, len(features))
            q = torch.from_numpy(features[start:stop]).half().cuda()
            dots = q @ self.train_gpu.T
            d2 = (2.0 - 2.0 * dots).clamp(min=0.0)
            kth_d2 = torch.topk(d2, self.k, dim=1, largest=False).values[:, self.k - 1]
            conf[start:stop] = (-kth_d2.float()).cpu().numpy()
            del q, dots, d2, kth_d2
        return outputs.preds, conf


class RknnPlusPlusAdapter(CachedMethodAdapter):
    """Relative kNN++ scoring using global and class-local neighbor distances."""

    K = 50
    neighbor_candidates = 300
    default_chunk_size = 128
    default_tie_break_weight = 0.001

    @property
    def k(self) -> int:
        """Return the neighbor rank used for global and class-local distances."""
        return int(self.params.get('K', self.K))

    @property
    def candidate_count(self) -> int:
        """Return how many global candidates to inspect for class-local matches."""
        return int(self.params.get('neighbor_candidates', self.neighbor_candidates))

    @property
    def chunk_size(self) -> int:
        """Return the query batch size for GPU feature search."""
        return int(self.params.get('chunk_size', self.default_chunk_size))

    @property
    def tie_break_weight(self) -> float:
        """Return the global-distance penalty used to break saturated ties."""
        return float(self.params.get('tie_break_weight', self.default_tie_break_weight))

    def score_cache_key(self) -> str:
        """Include scoring hyperparameters in the persisted score-cache key."""
        return f'{self.name}_{KNN_CACHE_VERSION}_k{self.k}_candidates{self.candidate_count}_tb{self.tie_break_weight:g}'

    def build_setup(self) -> None:
        """Normalize train features and keep them on GPU for exact search."""
        if not torch.cuda.is_available():
            raise RuntimeError('RkNN++ requires CUDA for ImageNet-1K feature search')

        labels, = self.runner.model_cache.get_arrays('id', 'train', 'labels')
        labels = labels.astype(np.int64, copy=False)
        first_chunk = next(self.runner.model_cache.iter_array('id', 'train', 'features', rows_per_chunk=1))
        train_norm = np.empty((labels.shape[0], first_chunk.shape[1]), dtype=np.float16)

        # Keep normalized train features on GPU as float16 for batched exact
        # search without building a separate FAISS index.
        offset = 0
        print('[rknnpp] normalizing train features', flush=True)
        for chunk in tqdm(self.runner.model_cache.iter_array('id', 'train', 'features'), desc='[rknnpp] train', unit='chunk', dynamic_ncols=False, ncols=100):
            values = normalizer(chunk.astype(np.float32, copy=False))
            train_norm[offset:offset + len(values)] = values.astype(np.float16, copy=False)
            offset += len(values)
        if offset != labels.shape[0]:
            raise RuntimeError(f'Expected {labels.shape[0]} train features, read {offset}')

        self.train_gpu = torch.from_numpy(train_norm).cuda()
        self.train_labels_gpu = torch.from_numpy(labels).cuda()
        del train_norm, labels
        print(f'[rknnpp] train table on GPU: {tuple(self.train_gpu.shape)} float16', flush=True)

    def score_outputs(self, outputs):
        """Return global-minus-class-local neighbor distance confidence."""
        features = normalizer(outputs.features.astype(np.float32, copy=False))
        preds = outputs.preds.astype(np.int64, copy=False)
        d_global = np.empty(len(features), dtype=np.float32)
        d_class = np.full(len(features), np.inf, dtype=np.float32)

        ranges = range(0, len(features), self.chunk_size)
        for start in tqdm(ranges, desc='[rknnpp] score', unit='chunk', dynamic_ncols=False, ncols=100):
            stop = min(start + self.chunk_size, len(features))
            q = torch.from_numpy(features[start:stop]).half().cuda()
            dots = q @ self.train_gpu.T
            d2 = (2.0 - 2.0 * dots).clamp(min=0.0)
            top_d2, top_idx = torch.topk(d2, self.candidate_count, dim=1, largest=False)
            d_global[start:stop] = top_d2[:, self.k - 1].float().sqrt().cpu().numpy()

            # Compare global neighbors with same-predicted-class neighbors; the
            # confidence is high when class-local neighbors are much closer.
            top_labels = self.train_labels_gpu[top_idx]
            pred_batch = torch.from_numpy(preds[start:stop]).cuda()
            match = top_labels == pred_batch.unsqueeze(1)
            top_d2_cpu = top_d2.float().cpu().numpy()
            match_cpu = match.cpu().numpy()
            for row in range(stop - start):
                matched = top_d2_cpu[row, match_cpu[row]]
                if len(matched) >= self.k:
                    d_class[start + row] = np.sqrt(matched[self.k - 1])
                elif len(matched) > 0:
                    d_class[start + row] = np.sqrt(matched[-1])
            del q, dots, d2, top_d2, top_idx, top_labels, pred_batch, match

        missing = np.isinf(d_class)
        if missing.any():
            d_class[missing] = d_global[missing] * 2.0
        conf = (d_global - d_class) - self.tie_break_weight * d_global
        return preds, conf.astype(np.float32, copy=False)


class MahalanobisAdapter(CachedMethodAdapter):
    """Feature-space Mahalanobis-family scoring."""

    @property
    def normalize_features(self) -> bool:
        """Return whether this variant normalizes features before fitting."""
        return bool(MAHALANOBIS_SETTINGS[self.name]['normalize_features'])

    @property
    def relative(self) -> bool:
        """Return whether this variant subtracts a global background score."""
        return bool(MAHALANOBIS_SETTINGS[self.name]['relative'])

    @property
    def num_classes(self) -> int:
        """Return the classifier class count."""
        return int(self.runner.model_cache.fc_weight.shape[0])

    def setup_cache_key(self) -> str:
        """Use feature-space setup state for this adapter family."""
        return f'features_{PRECISION_CACHE_VERSION}'

    def score_cache_key(self) -> str:
        """Include precision-estimation version in the persisted score key."""
        return f'{self.name}_{PRECISION_CACHE_VERSION}'

    def load_setup_cache(self, data) -> bool:
        """Load persisted Mahalanobis means and precision matrices."""
        self.class_mean = data['class_mean'].astype(np.float32, copy=False)
        self.precision = data['precision'].astype(np.float32, copy=False)
        if 'whole_mean' in data:
            self.whole_mean = data['whole_mean'].astype(np.float32, copy=False)
            self.whole_precision = data['whole_precision'].astype(np.float32, copy=False)
        return True

    def build_setup(self) -> None:
        """Fit feature-space class means and pooled precision."""
        self._fit_streamed_statistics('features')
        payload = {
            'class_mean': self.class_mean.astype(np.float32, copy=False),
            'precision': self.precision.astype(np.float32, copy=False),
        }
        if self.relative:
            payload['whole_mean'] = self.whole_mean.astype(np.float32, copy=False)
            payload['whole_precision'] = self.whole_precision.astype(np.float32, copy=False)
        self.runner._save_setup_cache(**payload)

    def _transform_chunk(self, values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float64, copy=True)
        if self.normalize_features:
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            values /= np.maximum(norms, 1e-12)
        return values

    def _fit_streamed_statistics(self, array_name: str) -> None:
        labels, = self.runner.model_cache.get_arrays('id', 'train', 'labels')
        labels = labels.astype(np.int64, copy=False)
        n_samples = labels.shape[0]
        n_classes = self.num_classes
        first_chunk = next(self.runner.model_cache.iter_array('id', 'train', array_name, rows_per_chunk=1))
        dim = first_chunk.shape[1]

        # First pass accumulates means; second pass accumulates covariance.
        counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
        class_sum = np.zeros((n_classes, dim), dtype=np.float64)
        whole_sum = np.zeros(dim, dtype=np.float64)
        offset = 0
        for chunk in self.runner.model_cache.iter_array('id', 'train', array_name):
            values = self._transform_chunk(chunk)
            chunk_labels = labels[offset:offset + values.shape[0]]
            np.add.at(class_sum, chunk_labels, values)
            whole_sum += values.sum(axis=0)
            offset += values.shape[0]

        class_mean = class_sum / counts[:, None]
        whole_mean = whole_sum / float(n_samples)
        class_scatter = np.zeros((dim, dim), dtype=np.float64)
        whole_scatter = np.zeros((dim, dim), dtype=np.float64) if self.relative else None
        offset = 0
        for chunk in self.runner.model_cache.iter_array('id', 'train', array_name):
            values = self._transform_chunk(chunk)
            chunk_labels = labels[offset:offset + values.shape[0]]
            centered = values - class_mean[chunk_labels]
            class_scatter += centered.T @ centered
            if whole_scatter is not None:
                whole_centered = values - whole_mean
                whole_scatter += whole_centered.T @ whole_centered
            offset += values.shape[0]

        self.class_mean = class_mean.astype(np.float32, copy=False)
        self.precision = stable_precision(class_scatter / float(n_samples)).astype(np.float32, copy=False)
        if self.relative:
            self.whole_mean = whole_mean.astype(np.float32, copy=False)
            self.whole_precision = stable_precision(whole_scatter / float(n_samples)).astype(np.float32, copy=False)

    def _transform_score_values(self, values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32, copy=False)
        if self.normalize_features:
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            values = values / np.maximum(norms, 1e-12)
        return values

    def class_scores(self, values: np.ndarray) -> np.ndarray:
        """Return one Mahalanobis-family score per sample and class."""
        values = self._transform_score_values(values)
        value_proj = values @ self.precision
        class_proj = self.class_mean @ self.precision
        value_quadratic = (value_proj * values).sum(axis=1, keepdims=True)
        class_quadratic = (class_proj * self.class_mean).sum(axis=1)[None, :]
        cross = value_proj @ self.class_mean.T
        class_scores = -(value_quadratic - 2.0 * cross + class_quadratic)
        if self.relative:
            centered = values - self.whole_mean[None, :]
            background = -((centered @ self.whole_precision) * centered).sum(axis=1)
            class_scores = class_scores - background[:, None]
        return class_scores.astype(np.float32, copy=False)

    def score_outputs(self, outputs):
        """Return maximum class score as confidence for one cached split."""
        class_scores = self.class_scores(outputs.features)
        conf = class_scores.max(axis=1)
        return outputs.preds, conf.astype(np.float32, copy=False)


class LogitMahalanobisAdapter(MahalanobisAdapter):
    """Logit-space Mahalanobis-family scoring."""

    def setup_cache_key(self) -> str:
        """Use logit-space setup state for this adapter family."""
        return f'logits_{PRECISION_CACHE_VERSION}'

    def score_cache_key(self) -> str:
        """Include logit-space identity in the persisted score-cache key."""
        return f'{self.name}_logits_{PRECISION_CACHE_VERSION}'

    def build_setup(self) -> None:
        """Fit logit-space class means and pooled precision."""
        self._fit_streamed_statistics('logits')
        payload = {
            'class_mean': self.class_mean.astype(np.float32, copy=False),
            'precision': self.precision.astype(np.float32, copy=False),
        }
        if self.relative:
            payload['whole_mean'] = self.whole_mean.astype(np.float32, copy=False)
            payload['whole_precision'] = self.whole_precision.astype(np.float32, copy=False)
        self.runner._save_setup_cache(**payload)

    def score_outputs(self, outputs):
        """Return maximum logit-space class score for one cached split."""
        class_scores = self.class_scores(outputs.logits)
        conf = class_scores.max(axis=1)
        return outputs.preds, conf.astype(np.float32, copy=False)
