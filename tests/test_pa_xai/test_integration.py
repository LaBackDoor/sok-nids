import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


def _make_schema():
    from pa_xai.core.schemas import DatasetSchema, HierarchicalConstraint
    return DatasetSchema(
        name="integration-test",
        feature_names=["proto", "duration", "bytes", "pkt_max", "pkt_mean", "pkt_min", "flags"],
        protocol_feature="proto",
        non_negative_features=["duration", "bytes", "pkt_max", "pkt_mean", "pkt_min", "flags"],
        tcp_only_features=["flags"],
        discrete_features=["flags"],
        hierarchical_constraints=[
            HierarchicalConstraint("pkt_max", "pkt_mean", "pkt_min"),
        ],
        protocol_encoding="integer",
    )


def _make_fixtures():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(7, 2)
        def forward(self, x):
            return self.fc(x)

    model = SimpleModel().eval()
    X_train = np.random.RandomState(42).uniform(0, 100, size=(20, 7)).astype(np.float32)
    X_train[:10, 0] = 6.0
    X_train[10:, 0] = 17.0
    y_train = np.zeros(20, dtype=int)
    return model, X_train, y_train


def test_all_public_api_imports():
    from pa_xai import (
        DatasetSchema, HierarchicalConstraint, ConstraintEnforcer,
        ExplanationResult, get_protocol_valid_baseline,
        CIC_IDS_2017, CSE_CIC_IDS2018, NSL_KDD,
        UNSW_NB15_NATIVE, UNSW_NB15_CIC,
        BUILTIN_SCHEMAS, get_schema,
        ProtocolAwareLIME, ProtocolAwareIG,
        ProtocolAwareDeepLIFT, ProtocolAwareSHAP,
        sparsity, fidelity, path_convergence,
        convergence_delta, additivity_check,
        semantic_robustness,
    )
    assert get_schema("CIC-IDS-2017") is CIC_IDS_2017


def test_all_methods_produce_consistent_results():
    from pa_xai import (
        ProtocolAwareLIME, ProtocolAwareIG,
        ProtocolAwareDeepLIFT, ProtocolAwareSHAP, sparsity,
    )

    schema = _make_schema()
    model, X_train, y_train = _make_fixtures()
    x = np.array([6.0, 150.0, 600.0, 100.0, 50.0, 10.0, 3.0], dtype=np.float32)

    # LIME
    lime = ProtocolAwareLIME(schema)
    lime_result = lime.explain_instance(
        x, lambda X: torch.softmax(model(torch.tensor(X, dtype=torch.float32)), dim=1).detach().numpy(),
        num_samples=500, sigma=1.0,
    )
    assert lime_result.method == "pa_lime"
    assert len(lime_result.attributions) == 7

    # IG
    ig = ProtocolAwareIG(schema, model, X_train, y_train, constrain_path=True)
    ig_result = ig.explain_instance(x, n_steps=30)
    assert ig_result.method == "pa_ig"
    assert len(ig_result.attributions) == 7

    # DeepLIFT
    dl = ProtocolAwareDeepLIFT(schema, model, X_train, y_train)
    dl_result = dl.explain_instance(x)
    assert dl_result.method == "pa_deeplift"
    assert len(dl_result.attributions) == 7

    # SHAP (Kernel)
    def predict_fn(X):
        with torch.no_grad():
            return torch.softmax(model(torch.tensor(X, dtype=torch.float32)), dim=1).numpy()

    shap_exp = ProtocolAwareSHAP(schema, predict_fn, X_train, y_train, backend="kernel", n_background=5)
    shap_result = shap_exp.explain_instance(x)
    assert shap_result.method == "pa_shap"
    assert len(shap_result.attributions) == 7

    # All methods produce sparsity
    for result in [lime_result, ig_result, dl_result, shap_result]:
        s = sparsity(result, threshold=0.001)
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0


def test_cross_method_top_features_overlap():
    from pa_xai import ProtocolAwareIG, ProtocolAwareDeepLIFT

    schema = _make_schema()

    class LinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(7, 2)
            with torch.no_grad():
                self.fc.weight[0] = torch.tensor([0, 0.1, 0.9, 0.05, 0.05, 0.05, 0])
                self.fc.weight[1] = torch.tensor([0, 0, 0, 0, 0, 0, 0.5])
                self.fc.bias.zero_()
        def forward(self, x):
            return self.fc(x)

    model = LinearModel().eval()
    X_train = np.random.RandomState(42).uniform(0, 100, size=(20, 7)).astype(np.float32)
    X_train[:, 0] = 6.0
    y_train = np.zeros(20, dtype=int)

    x = np.array([6.0, 1.0, 5.0, 2.0, 1.0, 0.5, 0.2], dtype=np.float32)

    ig = ProtocolAwareIG(schema, model, X_train, y_train, constrain_path=False)
    dl = ProtocolAwareDeepLIFT(schema, model, X_train, y_train)

    ig_result = ig.explain_instance(x, target=0, n_steps=100)
    dl_result = dl.explain_instance(x, target=0)

    ig_top = {name for name, _ in ig_result.top_features(k=3)}
    dl_top = {name for name, _ in dl_result.top_features(k=3)}

    assert "bytes" in ig_top
    assert "bytes" in dl_top
