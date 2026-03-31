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
    predict_fn_or_model=None,
    schema=None,
    epsilon: float = 0.05,
    n_iter: int = 50,
    **explain_kwargs,
) -> float:
    """Evaluate explanation stability under semantically valid mutations.

    Works with any PA-XAI explainer. For LIME, pass predict_fn via explain_kwargs.
    For IG/DeepLIFT/SHAP, no extra args needed.

    Returns:
        Mean Spearman rank correlation (float in [-1, 1]).
    """
    from pa_xai.lime.fuzzer import DomainConstraintFuzzer

    # Get schema from explainer
    if schema is None:
        schema = explainer.schema

    fuzzer = DomainConstraintFuzzer(schema)

    # Forward predict_fn_or_model as predict_fn for LIME compatibility
    if predict_fn_or_model is not None and "predict_fn" not in explain_kwargs:
        explain_kwargs = {"predict_fn": predict_fn_or_model, **explain_kwargs}

    # Get base explanation
    base = explainer.explain_instance(x_row, **explain_kwargs)
    base_attr = base.attributions

    scores = []
    for _ in range(n_iter):
        mutated = fuzzer.generate(x_row, num_samples=2, sigma=epsilon)
        x_mut = mutated[1]

        mut_result = explainer.explain_instance(x_mut, **explain_kwargs)
        corr, _ = spearmanr(base_attr, mut_result.attributions)
        if not np.isnan(corr):
            scores.append(corr)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def semantic_robustness_pcap(
    pcap_path: str,
    explainer,
    predict_fn,
    feature_fn,
    feature_names: list[str],
    mode: str = "packet",
    n_iter: int = 50,
    num_samples: int = 5000,
    sigma: float = 0.1,
) -> float:
    """Evaluate explanation stability under semantically valid PCAP mutations.

    Returns:
        Mean Spearman rank correlation (float in [-1, 1]).
    """
    base = explainer.explain_pcap(
        pcap_path, predict_fn, feature_fn, feature_names,
        mode=mode, num_samples=num_samples, sigma=sigma,
    )
    base_attr = base.attributions

    scores = []
    for _ in range(n_iter):
        mut_result = explainer.explain_pcap(
            pcap_path, predict_fn, feature_fn, feature_names,
            mode=mode, num_samples=num_samples, sigma=sigma,
        )
        corr, _ = spearmanr(base_attr, mut_result.attributions)
        if not np.isnan(corr):
            scores.append(corr)

    if not scores:
        return 0.0
    return float(np.mean(scores))
