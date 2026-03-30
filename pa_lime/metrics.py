"""XAI evaluation metrics for Protocol-Aware LIME explanations."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from pa_lime.explainer import ProtocolAwareLIME
from pa_lime.result import ExplanationResult


def semantic_robustness(
    x_row: np.ndarray,
    explainer: ProtocolAwareLIME,
    predict_fn,
    epsilon: float = 0.05,
    n_iter: int = 50,
    num_samples: int = 5000,
    sigma: float = 0.1,
) -> float:
    """Evaluate explanation stability under semantically valid mutations.

    Args:
        x_row: 1D array — the instance to evaluate.
        explainer: A configured ProtocolAwareLIME instance.
        predict_fn: The black-box model's prediction function.
        epsilon: Maximum perturbation magnitude for mutations.
        n_iter: Number of mutations to test.
        num_samples: Samples per explanation.
        sigma: Perturbation scale for each explanation.

    Returns:
        Mean Spearman rank correlation (float in [-1, 1]).
    """
    base = explainer.explain_instance(x_row, predict_fn, num_samples=num_samples, sigma=sigma)
    base_coef = base.coefficients

    scores = []
    for _ in range(n_iter):
        mutated = explainer.fuzzer.generate(x_row, num_samples=2, sigma=epsilon)
        x_mut = mutated[1]  # row 0 is original; row 1 is mutation

        mut_result = explainer.explain_instance(
            x_mut, predict_fn, num_samples=num_samples, sigma=sigma,
        )
        corr, _ = spearmanr(base_coef, mut_result.coefficients)
        if not np.isnan(corr):
            scores.append(corr)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def fidelity(result: ExplanationResult) -> float:
    """Return the R-squared fidelity of an explanation."""
    return result.r_squared


def sparsity(result: ExplanationResult, threshold: float = 0.01) -> int:
    """Count features with importance above threshold."""
    return int(np.sum(np.abs(result.coefficients) > threshold))
