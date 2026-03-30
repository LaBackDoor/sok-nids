import numpy as np
import pytest


def _make_schema():
    from pa_lime.schemas import DatasetSchema, HierarchicalConstraint
    return DatasetSchema(
        name="test",
        feature_names=["proto", "duration", "pkt_max", "pkt_mean", "pkt_min"],
        protocol_feature="proto",
        non_negative_features=["duration", "pkt_max", "pkt_mean", "pkt_min"],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[
            HierarchicalConstraint("pkt_max", "pkt_mean", "pkt_min"),
        ],
        protocol_encoding="integer",
    )


def _linear_predict_fn(X):
    """prediction = 0.1*duration + 0.5*pkt_max"""
    return 0.1 * X[:, 1] + 0.5 * X[:, 2]


def test_explain_instance_returns_explanation_result():
    from pa_lime.explainer import ProtocolAwareLIME
    from pa_lime.result import ExplanationResult
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0])
    result = explainer.explain_instance(x, _linear_predict_fn, num_samples=2000, sigma=1.0)
    assert isinstance(result, ExplanationResult)
    assert len(result.coefficients) == 5
    assert len(result.feature_names) == 5
    assert result.num_samples == 2000
    assert isinstance(result.r_squared, float)
    assert isinstance(result.intercept, float)
    assert isinstance(result.local_prediction, float)


def test_explain_instance_identifies_important_features():
    from pa_lime.explainer import ProtocolAwareLIME
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0])
    result = explainer.explain_instance(x, _linear_predict_fn, num_samples=5000, sigma=1.0)
    top = result.top_features(k=2)
    top_names = [name for name, _ in top]
    assert "pkt_max" in top_names
    assert "duration" in top_names


def test_explain_instance_high_fidelity_for_linear_model():
    from pa_lime.explainer import ProtocolAwareLIME
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0])
    result = explainer.explain_instance(x, _linear_predict_fn, num_samples=5000, sigma=1.0)
    assert result.r_squared > 0.8


def test_explain_instance_multiclass_predict_fn():
    from pa_lime.explainer import ProtocolAwareLIME
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)

    def multiclass_predict(X):
        probs = np.column_stack([
            0.1 * X[:, 1],
            0.5 * X[:, 2],
            0.01 * X[:, 3],
        ])
        return probs

    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0])
    result = explainer.explain_instance(
        x, multiclass_predict, num_samples=3000, sigma=1.0, class_to_explain=1,
    )
    assert result.predicted_class == 1
    top = result.top_features(k=1)
    assert top[0][0] == "pkt_max"


def test_explain_instance_auto_detects_class_for_1d():
    from pa_lime.explainer import ProtocolAwareLIME
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0])
    result = explainer.explain_instance(x, _linear_predict_fn, num_samples=2000, sigma=1.0)
    assert result.predicted_class is None


def test_explain_instance_with_feature_wise_sigma():
    from pa_lime.explainer import ProtocolAwareLIME
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0])
    sigma = np.array([0.0, 2.0, 1.0, 0.5, 0.2])
    result = explainer.explain_instance(x, _linear_predict_fn, num_samples=3000, sigma=sigma)
    assert result.r_squared > 0.5


def test_explain_instance_custom_kernel_width():
    from pa_lime.explainer import ProtocolAwareLIME
    from pa_lime.result import ExplanationResult
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0])
    result = explainer.explain_instance(
        x, _linear_predict_fn, num_samples=2000, sigma=1.0, kernel_width=0.5,
    )
    assert isinstance(result, ExplanationResult)


def test_as_dict_returns_feature_coefficient_mapping():
    from pa_lime.explainer import ProtocolAwareLIME
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0])
    result = explainer.explain_instance(x, _linear_predict_fn, num_samples=2000, sigma=1.0)
    d = result.as_dict()
    assert isinstance(d, dict)
    assert set(d.keys()) == {"proto", "duration", "pkt_max", "pkt_mean", "pkt_min"}
