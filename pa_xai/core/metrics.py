"""Shared XAI evaluation metrics for all Protocol-Aware explainers."""

from __future__ import annotations

import numpy as np

from pa_xai.core.result import ExplanationResult


def sparsity(result: ExplanationResult, threshold: float = 0.01, relative: bool = True) -> float:
    """Fraction of features with importance above threshold.

    Args:
        result: Explanation result.
        threshold: If relative=True, fraction of max |attribution|.
                   If relative=False, absolute threshold.
        relative: Whether threshold is relative to max attribution.

    Returns:
        Fraction of features above threshold (0.0 to 1.0).
    """
    abs_attr = np.abs(result.attributions)
    if relative and abs_attr.max() > 0:
        cutoff = threshold * abs_attr.max()
    else:
        cutoff = threshold
    count = int(np.sum(abs_attr > cutoff))
    return count / len(result.attributions)
