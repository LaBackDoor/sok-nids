import numpy as np


def test_explanation_result_stores_attributions():
    from pa_xai.core.result import ExplanationResult
    result = ExplanationResult(
        feature_names=["a", "b", "c"],
        attributions=np.array([0.5, -0.3, 0.1]),
        method="pa_ig",
        predicted_class=1,
        num_samples=None,
    )
    assert result.method == "pa_ig"
    assert len(result.attributions) == 3
    assert result.predicted_class == 1
    assert result.num_samples is None
    assert result.r_squared is None
    assert result.convergence_delta is None


def test_explanation_result_top_features():
    from pa_xai.core.result import ExplanationResult
    result = ExplanationResult(
        feature_names=["duration", "bytes", "proto", "flags"],
        attributions=np.array([0.1, 0.9, -0.5, 0.05]),
        method="pa_shap",
        predicted_class=0,
        num_samples=100,
    )
    top = result.top_features(k=2)
    assert len(top) == 2
    assert top[0][0] == "bytes"
    assert top[1][0] == "proto"


def test_explanation_result_top_features_not_absolute():
    from pa_xai.core.result import ExplanationResult
    result = ExplanationResult(
        feature_names=["a", "b", "c"],
        attributions=np.array([-0.9, 0.5, 0.1]),
        method="pa_ig",
        predicted_class=0,
        num_samples=None,
    )
    top = result.top_features(k=2, absolute=False)
    assert top[0][0] == "b"
    assert top[1][0] == "c"


def test_explanation_result_as_dict():
    from pa_xai.core.result import ExplanationResult
    result = ExplanationResult(
        feature_names=["x", "y"],
        attributions=np.array([1.0, 2.0]),
        method="pa_lime",
        predicted_class=None,
        num_samples=5000,
    )
    d = result.as_dict()
    assert d == {"x": 1.0, "y": 2.0}


def test_explanation_result_with_lime_fields():
    from pa_xai.core.result import ExplanationResult
    result = ExplanationResult(
        feature_names=["a", "b"],
        attributions=np.array([0.3, 0.7]),
        method="pa_lime",
        predicted_class=1,
        num_samples=5000,
        r_squared=0.95,
        intercept=0.1,
        local_prediction=0.85,
    )
    assert result.r_squared == 0.95
    assert result.intercept == 0.1
    assert result.local_prediction == 0.85


def test_explanation_result_with_ig_fields():
    from pa_xai.core.result import ExplanationResult
    baseline = np.array([0.0, 1.0])
    result = ExplanationResult(
        feature_names=["a", "b"],
        attributions=np.array([0.3, 0.7]),
        method="pa_ig",
        predicted_class=1,
        num_samples=None,
        convergence_delta=0.001,
        baseline_used=baseline,
    )
    assert result.convergence_delta == 0.001
    assert np.array_equal(result.baseline_used, baseline)


def test_explanation_result_coefficients_alias():
    from pa_xai.core.result import ExplanationResult
    result = ExplanationResult(
        feature_names=["a", "b"],
        attributions=np.array([0.3, 0.7]),
        method="pa_lime",
        predicted_class=None,
        num_samples=500,
    )
    assert np.array_equal(result.coefficients, result.attributions)
