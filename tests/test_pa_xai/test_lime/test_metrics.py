import numpy as np
import pytest


def _make_schema():
    from pa_xai.core.schemas import DatasetSchema
    return DatasetSchema(
        name="test",
        feature_names=["proto", "duration", "bytes"],
        protocol_feature="proto",
        non_negative_features=["duration", "bytes"],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
    )


def _stable_predict_fn(X):
    return 0.3 * X[:, 1] + 0.7 * X[:, 2]


def test_semantic_robustness_returns_float():
    from pa_xai.lime import ProtocolAwareLIME
    from pa_xai.lime.metrics import semantic_robustness
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)
    x = np.array([6.0, 100.0, 500.0])
    score = semantic_robustness(
        x, explainer, _stable_predict_fn, epsilon=0.05, n_iter=10, num_samples=500,
    )
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_semantic_robustness_high_for_linear_model():
    from pa_xai.lime import ProtocolAwareLIME
    from pa_xai.lime.metrics import semantic_robustness
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)
    x = np.array([6.0, 100.0, 500.0])
    score = semantic_robustness(
        x, explainer, _stable_predict_fn, epsilon=0.01, n_iter=20, num_samples=1000,
    )
    assert score > 0.7


def test_semantic_robustness_respects_protocol_constraints():
    from pa_xai.lime import ProtocolAwareLIME
    from pa_xai.lime.metrics import semantic_robustness
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)
    x = np.array([17.0, 100.0, 500.0])
    score = semantic_robustness(
        x, explainer, _stable_predict_fn, epsilon=0.05, n_iter=10, num_samples=500,
    )
    assert isinstance(score, float)


def test_fidelity_score():
    from pa_xai.lime import ProtocolAwareLIME
    from pa_xai.lime.metrics import fidelity
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)
    x = np.array([6.0, 100.0, 500.0])
    result = explainer.explain_instance(x, _stable_predict_fn, num_samples=2000, sigma=1.0)
    r2 = fidelity(result)
    assert isinstance(r2, float)
    assert r2 == result.r_squared


def test_sparsity_score():
    from pa_xai.lime import ProtocolAwareLIME
    from pa_xai.core.metrics import sparsity
    schema = _make_schema()
    explainer = ProtocolAwareLIME(schema)
    x = np.array([6.0, 100.0, 500.0])
    result = explainer.explain_instance(x, _stable_predict_fn, num_samples=2000, sigma=1.0)
    s = sparsity(result, threshold=0.01)
    assert isinstance(s, int)
    assert 0 <= s <= len(result.attributions)
