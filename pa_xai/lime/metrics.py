"""LIME-specific evaluation metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from pa_xai.core.result import ExplanationResult


def fidelity(result: ExplanationResult) -> float:
    """Return the R-squared fidelity of a LIME explanation."""
    if result.r_squared is None:
        raise ValueError("fidelity requires r_squared (LIME explanations only)")
    return result.r_squared


def semantic_robustness(
    x_row: np.ndarray,
    explainer,
    predict_fn,
    epsilon: float = 0.05,
    n_iter: int = 50,
    num_samples: int = 5000,
    sigma: float = 0.1,
) -> float:
    """Evaluate explanation stability under semantically valid mutations.

    Returns:
        Mean Spearman rank correlation (float in [-1, 1]).
    """
    base = explainer.explain_instance(x_row, predict_fn, num_samples=num_samples, sigma=sigma)
    base_attr = base.attributions

    scores = []
    for _ in range(n_iter):
        mutated = explainer.fuzzer.generate(x_row, num_samples=2, sigma=epsilon)
        x_mut = mutated[1]

        mut_result = explainer.explain_instance(
            x_mut, predict_fn, num_samples=num_samples, sigma=sigma,
        )
        corr, _ = spearmanr(base_attr, mut_result.attributions)
        if not np.isnan(corr):
            scores.append(corr)

    if not scores:
        return 0.0
    return float(np.mean(scores))
