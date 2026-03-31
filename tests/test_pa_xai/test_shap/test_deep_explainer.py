import numpy as np
import torch
import torch.nn as nn


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


def _make_model_and_data():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(8, 3)
        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = SimpleModel().eval()
    X_train = np.array([
        [6.0, 100.0, 500.0, 3.0],
        [6.0, 110.0, 510.0, 2.0],
        [17.0, 50.0, 200.0, 0.0],
        [17.0, 60.0, 220.0, 0.0],
    ], dtype=np.float32)
    y_train = np.array([0, 0, 0, 0])
    return model, X_train, y_train


def test_deep_shap_returns_result():
    from pa_xai.shap import ProtocolAwareSHAP
    from pa_xai.core.result import ExplanationResult
    schema = _make_schema()
    model, X_train, y_train = _make_model_and_data()
    explainer = ProtocolAwareSHAP(
        schema, model, X_train, y_train, backend="deep", n_background=2,
    )
    x = np.array([6.0, 150.0, 600.0, 5.0], dtype=np.float32)
    result = explainer.explain_instance(x, target=0)
    assert isinstance(result, ExplanationResult)
    assert result.method == "pa_shap"
    assert len(result.attributions) == 4


def test_deep_shap_protocol_filtered_background():
    from pa_xai.shap import ProtocolAwareSHAP
    schema = _make_schema()
    model, X_train, y_train = _make_model_and_data()
    explainer = ProtocolAwareSHAP(
        schema, model, X_train, y_train, backend="deep", n_background=2,
    )
    x = np.array([17.0, 300.0, 1000.0, 0.0], dtype=np.float32)
    result = explainer.explain_instance(x, target=0)
    assert result.method == "pa_shap"
