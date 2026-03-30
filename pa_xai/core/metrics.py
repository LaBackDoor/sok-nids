"""Shared XAI evaluation metrics for all Protocol-Aware explainers."""

from __future__ import annotations

import numpy as np

from pa_xai.core.result import ExplanationResult


def sparsity(result: ExplanationResult, threshold: float = 0.01) -> int:
    """Count features with importance above threshold."""
    return int(np.sum(np.abs(result.attributions) > threshold))
