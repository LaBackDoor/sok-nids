"""DeepLIFT-specific evaluation metrics."""

from __future__ import annotations

from pa_xai.core.result import ExplanationResult


def convergence_delta(result: ExplanationResult) -> float | None:
    """Return the completeness check value if available."""
    return result.convergence_delta
