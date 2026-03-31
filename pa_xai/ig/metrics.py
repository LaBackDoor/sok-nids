"""IG-specific evaluation metrics."""

from __future__ import annotations

from pa_xai.core.result import ExplanationResult


def path_convergence(result: ExplanationResult) -> float | None:
    """Return the convergence delta if available. Closer to 0 = better."""
    return result.convergence_delta
