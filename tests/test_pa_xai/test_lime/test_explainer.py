import numpy as np


def _make_schema():
    from pa_xai.core.schemas import DatasetSchema, HierarchicalConstraint
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
    return 0.1 * X[:, 1] + 0.5 * X[:, 2]


def test_lime_returns_explanation_result():
    from pa_xai.lime import ProtocolAwareLIME
    from pa_xai.core.result import ExplanationResult
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0])
    result = explainer.explain_instance(x, _linear_predict_fn, num_samples=2000, sigma=1.0)
    assert isinstance(result, ExplanationResult)
    assert result.method == "pa_lime"
    assert len(result.attributions) == 5
    assert result.r_squared is not None
    assert result.intercept is not None


def test_lime_identifies_important_features():
    from pa_xai.lime import ProtocolAwareLIME
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0])
    result = explainer.explain_instance(x, _linear_predict_fn, num_samples=5000, sigma=1.0)
    top = result.top_features(k=2)
    top_names = [name for name, _ in top]
    assert "pkt_max" in top_names


def test_lime_fidelity_metric():
    from pa_xai.lime import ProtocolAwareLIME, fidelity
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0])
    result = explainer.explain_instance(x, _linear_predict_fn, num_samples=5000, sigma=1.0)
    assert fidelity(result) == result.r_squared
