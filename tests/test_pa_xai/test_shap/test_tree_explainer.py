import numpy as np
from sklearn.ensemble import RandomForestClassifier


def _make_schema():
    from pa_xai.core.schemas import DatasetSchema
    return DatasetSchema(
        name="test",
        feature_names=["proto", "duration", "bytes", "flags"],
        protocol_feature="proto",
        non_negative_features=["duration", "bytes", "flags"],
        tcp_only_features=["flags"],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
    )


def _make_rf_and_data():
    X_train = np.array([
        [6.0, 100.0, 500.0, 3.0],
        [6.0, 110.0, 510.0, 2.0],
        [6.0, 900.0, 9000.0, 5.0],
        [17.0, 50.0, 200.0, 0.0],
        [17.0, 60.0, 220.0, 0.0],
        [17.0, 800.0, 8000.0, 0.0],
    ], dtype=np.float32)
    y_train = np.array([0, 0, 1, 0, 0, 1])
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    return rf, X_train, y_train


def test_tree_shap_returns_result():
    from pa_xai.shap import ProtocolAwareSHAP
    from pa_xai.core.result import ExplanationResult
    schema = _make_schema()
    rf, X_train, y_train = _make_rf_and_data()
    explainer = ProtocolAwareSHAP(
        schema, rf, X_train, y_train, backend="tree", n_background=3,
    )
    x = np.array([6.0, 500.0, 5000.0, 4.0], dtype=np.float32)
    result = explainer.explain_instance(x, target=1)
    assert isinstance(result, ExplanationResult)
    assert result.method == "pa_shap"
    assert len(result.attributions) == 4


def test_tree_shap_auto_detects_target():
    from pa_xai.shap import ProtocolAwareSHAP
    schema = _make_schema()
    rf, X_train, y_train = _make_rf_and_data()
    explainer = ProtocolAwareSHAP(
        schema, rf, X_train, y_train, backend="tree", n_background=3,
    )
    x = np.array([6.0, 500.0, 5000.0, 4.0], dtype=np.float32)
    result = explainer.explain_instance(x, target=None)
    assert result.predicted_class is not None
