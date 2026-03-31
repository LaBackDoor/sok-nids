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
    # relative=True, threshold=0.01: cutoff = 0.01 * 0.5 = 0.005
    # |0.5| > 0.005 YES, |0.001| > 0.005 NO, |0.3| > 0.005 YES, |0.005| > 0.005 NO
    assert sparsity(result, threshold=0.01) == 2 / 4  # 0.5


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
    # relative threshold: 0.01 * 0.002 = 0.00002, both above -> 1.0
    # Use absolute mode for "all below"
    assert sparsity(result, threshold=0.01, relative=False) == 0.0


def test_sparsity_absolute_mode():
    from pa_xai.core.result import ExplanationResult
    from pa_xai.core.metrics import sparsity

    result = ExplanationResult(
        feature_names=["a", "b", "c"],
        attributions=np.array([0.5, 0.005, -0.3]),
        method="pa_ig",
        predicted_class=0,
        num_samples=None,
    )
    assert sparsity(result, threshold=0.01, relative=False) == 2 / 3
