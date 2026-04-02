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
        discrete_features=["flags"],
        hierarchical_constraints=[],
        protocol_encoding="integer",
    )


def _make_model_and_data():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)
            with torch.no_grad():
                self.fc.weight.copy_(torch.tensor([
                    [0.0, 0.5, 0.3, 0.0],
                    [0.0, 0.0, 0.0, 0.8],
                    [0.0, 0.1, 0.1, 0.1],
                ]))
                self.fc.bias.zero_()
        def forward(self, x):
            return self.fc(x)

    model = SimpleModel().eval()
    X_train = np.array([
        [6.0, 100.0, 500.0, 3.0],
        [6.0, 110.0, 510.0, 2.0],
        [6.0, 120.0, 520.0, 1.0],
        [17.0, 50.0, 200.0, 0.0],
        [17.0, 60.0, 220.0, 0.0],
        [17.0, 70.0, 240.0, 0.0],
    ], dtype=np.float32)
    y_train = np.array([0, 0, 0, 0, 0, 0])
    return model, X_train, y_train


def test_kernel_shap_returns_result():
    from pa_xai.shap import ProtocolAwareSHAP
    from pa_xai.core.result import ExplanationResult
    schema = _make_schema()
    model, X_train, y_train = _make_model_and_data()

    def predict_fn(X):
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            return torch.softmax(model(t), dim=1).numpy()

    explainer = ProtocolAwareSHAP(
        schema, predict_fn, X_train, y_train, backend="kernel", n_background=3,
    )
    x = np.array([6.0, 150.0, 600.0, 5.0], dtype=np.float32)
    result = explainer.explain_instance(x, target=0)
    assert isinstance(result, ExplanationResult)
    assert result.method == "pa_shap"
    assert len(result.attributions) == 4


def test_kernel_shap_different_protocol_input():
    from pa_xai.shap import ProtocolAwareSHAP
    schema = _make_schema()
    model, X_train, y_train = _make_model_and_data()

    def predict_fn(X):
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            return torch.softmax(model(t), dim=1).numpy()

    explainer = ProtocolAwareSHAP(
        schema, predict_fn, X_train, y_train, backend="kernel", n_background=3,
    )
    x = np.array([17.0, 800.0, 8000.0, 0.0], dtype=np.float32)
    result = explainer.explain_instance(x, target=0)
    assert result.method == "pa_shap"
    assert len(result.attributions) == 4
