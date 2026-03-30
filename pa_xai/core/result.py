"""Unified explanation result for all PA-XAI methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExplanationResult:
    """Result of a single protocol-aware explanation."""

    feature_names: list[str]
    attributions: np.ndarray
    method: str
    predicted_class: int | str | None
    num_samples: int | None

    # LIME-specific
    r_squared: float | None = None
    intercept: float | None = None
    local_prediction: float | None = None

    # IG / DeepLIFT
    convergence_delta: float | None = None
    baseline_used: np.ndarray | None = None

    # SHAP
    expected_value: float | None = None

    def top_features(self, k: int = 10, absolute: bool = True) -> list[tuple[str, float]]:
        """Return the top-k most important features by attribution magnitude."""
        if absolute:
            order = np.argsort(np.abs(self.attributions))[::-1]
        else:
            order = np.argsort(self.attributions)[::-1]
        return [(self.feature_names[i], float(self.attributions[i])) for i in order[:k]]

    def as_dict(self) -> dict[str, float]:
        """Return feature_name -> attribution mapping."""
        return dict(zip(self.feature_names, self.attributions.tolist()))

    @property
    def coefficients(self) -> np.ndarray:
        """Backwards-compatible alias for attributions (LIME terminology)."""
        return self.attributions
