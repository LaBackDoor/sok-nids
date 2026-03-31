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
    class LinearModel(nn.Module):
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

    model = LinearModel().eval()
    X_train = np.array([
        [6.0, 100.0, 500.0, 3.0],
        [6.0, 110.0, 510.0, 2.0],
        [17.0, 50.0, 200.0, 0.0],
        [17.0, 60.0, 220.0, 0.0],
    ], dtype=np.float32)
    y_train = np.array([0, 0, 0, 0])
    return model, X_train, y_train


def test_ig_returns_explanation_result():
    from pa_xai.ig import ProtocolAwareIG
    from pa_xai.core.result import ExplanationResult
    schema = _make_schema()
    model, X_train, y_train = _make_model_and_data()
    explainer = ProtocolAwareIG(schema, model, X_train, y_train)
    x = np.array([6.0, 150.0, 600.0, 5.0], dtype=np.float32)
    result = explainer.explain_instance(x, target=0, n_steps=50)
    assert isinstance(result, ExplanationResult)
    assert result.method == "pa_ig"
    assert len(result.attributions) == 4
    assert result.num_samples is None
    assert result.baseline_used is not None


def test_ig_baseline_is_protocol_matched():
    from pa_xai.ig import ProtocolAwareIG
    schema = _make_schema()
    model, X_train, y_train = _make_model_and_data()
    explainer = ProtocolAwareIG(schema, model, X_train, y_train)
    x = np.array([17.0, 800.0, 8000.0, 0.0], dtype=np.float32)
    result = explainer.explain_instance(x, target=0, n_steps=50)
    assert result.baseline_used[0] == 17.0


def test_ig_path_clamping_enforces_constraints():
    from pa_xai.ig import ProtocolAwareIG
    schema = _make_schema()
    model, X_train, y_train = _make_model_and_data()
    explainer = ProtocolAwareIG(schema, model, X_train, y_train, constrain_path=True)
    x = np.array([17.0, 800.0, 8000.0, 0.0], dtype=np.float32)
    result = explainer.explain_instance(x, target=0, n_steps=50)
    assert isinstance(result.attributions, np.ndarray)
    assert len(result.attributions) == 4


def test_ig_without_path_clamping():
    from pa_xai.ig import ProtocolAwareIG
    schema = _make_schema()
    model, X_train, y_train = _make_model_and_data()
    explainer = ProtocolAwareIG(schema, model, X_train, y_train, constrain_path=False)
    x = np.array([6.0, 150.0, 600.0, 5.0], dtype=np.float32)
    result = explainer.explain_instance(x, target=0, n_steps=50)
    assert result.method == "pa_ig"
    assert len(result.attributions) == 4


def test_ig_convergence_delta():
    from pa_xai.ig import ProtocolAwareIG
    schema = _make_schema()
    model, X_train, y_train = _make_model_and_data()
    explainer = ProtocolAwareIG(schema, model, X_train, y_train, constrain_path=False)
    x = np.array([6.0, 150.0, 600.0, 5.0], dtype=np.float32)
    result = explainer.explain_instance(x, target=0, n_steps=200, return_convergence_delta=True)
    assert result.convergence_delta is not None


def test_ig_auto_detects_target():
    from pa_xai.ig import ProtocolAwareIG
    schema = _make_schema()
    model, X_train, y_train = _make_model_and_data()
    explainer = ProtocolAwareIG(schema, model, X_train, y_train)
    x = np.array([6.0, 150.0, 600.0, 5.0], dtype=np.float32)
    result = explainer.explain_instance(x, target=None, n_steps=50)
    assert result.predicted_class is not None
    assert len(result.attributions) == 4
