"""SHAP-specific evaluation metrics."""

from __future__ import annotations

from pa_xai.core.result import ExplanationResult


def additivity_check(result: ExplanationResult) -> float | None:
    """Return the SHAP expected_value if available."""
    return result.expected_value
