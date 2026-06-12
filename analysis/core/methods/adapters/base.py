"""Base contract for scoring cached model outputs with one OOD method."""

from __future__ import annotations

import numpy as np


def normalizer(x: np.ndarray) -> np.ndarray:
    """Apply the OpenOOD kNN feature normalization convention."""
    # Match OpenOOD kNN normalization: unit length with a tiny positive offset.
    return x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class CachedMethodAdapter:
    """Base adapter interface for scoring cached model outputs."""

    def __init__(self, runner) -> None:
        """Attach the adapter to its method runner and model cache."""
        self.runner = runner

    @property
    def name(self) -> str:
        """Return the canonical postprocessor name for this adapter."""
        return self.runner.name

    @property
    def params(self):
        """Return dataset-specific method parameters."""
        return self.runner.params

    def setup_cache_key(self) -> str | None:
        """Return a setup-cache key, or None when no setup cache is needed."""
        # Return a key when build_setup is expensive and can be persisted.
        return None

    def score_cache_key(self) -> str:
        """Return the cache key used for persisted split scores."""
        return self.name

    def load_setup_cache(self, data) -> bool:
        """Load persisted setup arrays; return False to rebuild setup."""
        return False

    def build_setup(self) -> None:
        """Fit any train/validation-derived method state."""
        return

    def score_outputs(self, outputs) -> tuple[np.ndarray, np.ndarray]:
        """Return predicted labels and confidence scores for one split."""
        raise NotImplementedError
