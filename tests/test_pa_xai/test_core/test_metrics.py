import numpy as np


def test_sparsity_counts_important_features():
    from pa_xai.core.result import ExplanationResult
    from pa_xai.core.metrics import sparsity
    result = ExplanationResult(
        feature_names=["a", "b", "c", "d"],
        attributions=np.array([0.5, 0.001, -0.3, 0.005]),
        method="pa_ig",
        predicted_class=0,
        num_samples=None,
    )
    assert sparsity(result, threshold=0.01) == 2
    assert sparsity(result, threshold=0.001) == 3


def test_sparsity_all_below_threshold():
    from pa_xai.core.result import ExplanationResult
    from pa_xai.core.metrics import sparsity
    result = ExplanationResult(
        feature_names=["a", "b"],
        attributions=np.array([0.001, 0.002]),
        method="pa_ig",
        predicted_class=0,
        num_samples=None,
    )
    assert sparsity(result, threshold=0.01) == 0
