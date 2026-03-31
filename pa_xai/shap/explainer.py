"""Protocol-Aware SHAP explainer for NIDS (multi-backend)."""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn

from pa_xai.core.constraints import ConstraintEnforcer
from pa_xai.core.result import ExplanationResult
from pa_xai.core.schemas import (
    DatasetSchema,
    TCP_PROTOCOL_INT,
    detect_protocol_encoding,
)


def _filter_background_by_protocol(
    X_train: np.ndarray,
    y_train: np.ndarray,
    protocol_value: float,
    schema: DatasetSchema,
    benign_label: int,
    n_background: int,
) -> np.ndarray:
    benign_mask = y_train == benign_label
    if schema.protocol_index is not None:
        proto_mask = X_train[:, schema.protocol_index] == protocol_value
        mask = benign_mask & proto_mask
    else:
        mask = benign_mask

    candidates = X_train[mask]
    if len(candidates) == 0:
        candidates = X_train[benign_mask]
    if len(candidates) == 0:
        raise ValueError("No benign samples found in training data.")

    if len(candidates) < 5:
        import logging
        logging.getLogger(__name__).warning(
            f"Only {len(candidates)} benign samples match the target protocol for SHAP background. "
            f"SHAP values may have high variance."
        )

    if len(candidates) > n_background:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(candidates), size=n_background, replace=False)
        candidates = candidates[idx]
    return candidates


def _has_rnn_modules(model: nn.Module) -> bool:
    return any(isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)) for m in model.modules())


def _extract_class_attributions(shap_values, target: int, n_features: int) -> np.ndarray:
    # Handle shap.Explanation objects (newer SHAP API)
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values

    if isinstance(shap_values, list):
        return np.asarray(shap_values[target]).flatten()[:n_features]
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            return shap_values[0, :, target]
        elif shap_values.ndim == 2:
            return shap_values.flatten()[:n_features]
        else:
            return shap_values.flatten()[:n_features]
    else:
        return np.asarray(shap_values).flatten()[:n_features]


class _ConstrainedKernelExplainer:
    def __init__(self, predict_fn, background, schema, enforcer,
                 protocol_value, protocol_encoding, tcp_label_value):
        import shap
        self._schema = schema
        self._enforcer = enforcer
        self._protocol_value = protocol_value
        self._protocol_encoding = protocol_encoding
        self._tcp_label_value = tcp_label_value

        original_predict = predict_fn
        def constrained_predict(X):
            X_clamped = X.copy()
            if len(X_clamped.shape) == 1:
                X_clamped = X_clamped.reshape(1, -1)
            self._enforcer.enforce(
                X_clamped, self._protocol_value,
                self._protocol_encoding, self._tcp_label_value,
            )
            return original_predict(X_clamped)

        self._explainer = shap.KernelExplainer(constrained_predict, background)

    def shap_values(self, X, **kwargs):
        return self._explainer.shap_values(X, **kwargs)

    @property
    def expected_value(self):
        return self._explainer.expected_value


class ProtocolAwareSHAP:
    """Protocol-Aware SHAP (multi-backend: kernel, deep, tree)."""

    def __init__(
        self,
        schema: DatasetSchema,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        benign_label: int = 0,
        backend: str = "kernel",
        n_background: int = 100,
        tcp_label_value: int = TCP_PROTOCOL_INT,
    ) -> None:
        if backend not in ("kernel", "deep", "tree"):
            raise ValueError(f"backend must be 'kernel', 'deep', or 'tree', got {backend!r}")
        self.schema = schema
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.benign_label = benign_label
        self.backend = backend
        self.n_background = n_background
        self.tcp_label_value = tcp_label_value
        self.enforcer = ConstraintEnforcer(schema)
        self._background_cache: dict[float | None, np.ndarray] = {}

    def _get_background(self, protocol_value: float | None) -> np.ndarray:
        """Get protocol-filtered background, cached per protocol value."""
        if protocol_value not in self._background_cache:
            self._background_cache[protocol_value] = _filter_background_by_protocol(
                self.X_train, self.y_train,
                protocol_value, self.schema,
                self.benign_label, self.n_background,
            )
        return self._background_cache[protocol_value]

    def _resolve_protocol_params(self, x_row):
        encoding = self.schema.protocol_encoding
        protocol_value = None
        tcp_val = self.tcp_label_value
        if self.schema.protocol_index is not None:
            protocol_value = x_row[self.schema.protocol_index]
            if encoding == "auto":
                encoding = detect_protocol_encoding(
                    x_row, self.schema.protocol_feature, self.schema.feature_names
                )
        return protocol_value, encoding, tcp_val

    def _predict_target(self, x_row):
        if self.backend == "tree":
            return int(self.model.predict(x_row.reshape(1, -1))[0])
        elif self.backend == "kernel":
            preds = self.model(x_row.reshape(1, -1))
            return int(np.argmax(preds[0]))
        else:
            with torch.no_grad():
                t = torch.tensor(x_row, dtype=torch.float32).unsqueeze(0)
                device = next(self.model.parameters()).device
                logits = self.model(t.to(device))
                return int(torch.argmax(logits, dim=1).item())

    def _explain_kernel(self, x_row, target, nsamples, background):
        protocol_value, encoding, tcp_val = self._resolve_protocol_params(x_row)
        explainer = _ConstrainedKernelExplainer(
            self.model, background, self.schema, self.enforcer,
            protocol_value, encoding, tcp_val,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            shap_values = explainer.shap_values(x_row.reshape(1, -1), nsamples=nsamples)
        n_features = len(x_row)
        attributions = _extract_class_attributions(shap_values, target, n_features)
        ev = explainer.expected_value
        expected_value = float(ev[target]) if isinstance(ev, (list, np.ndarray)) else float(ev)
        return attributions, expected_value

    def _explain_deep(self, x_row, target, background):
        import shap
        device = next(self.model.parameters()).device
        bg_tensor = torch.tensor(background, dtype=torch.float32).to(device)
        base_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        base_model.eval()
        use_gradient = _has_rnn_modules(base_model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if use_gradient:
                explainer = shap.GradientExplainer(base_model, bg_tensor)
            else:
                explainer = shap.DeepExplainer(base_model, bg_tensor)
            x_tensor = torch.tensor(x_row, dtype=torch.float32).unsqueeze(0).to(device)
            if use_gradient:
                shap_values = explainer.shap_values(x_tensor)
            else:
                shap_values = explainer.shap_values(x_tensor, check_additivity=False)
        n_features = len(x_row)
        attributions = _extract_class_attributions(shap_values, target, n_features)
        if use_gradient:
            # GradientExplainer does not set expected_value — compute manually
            with torch.no_grad():
                ev_tensor = base_model(bg_tensor).mean(0).cpu().numpy()
            ev = ev_tensor
        else:
            ev = explainer.expected_value
        expected_value = float(ev[target]) if isinstance(ev, (list, np.ndarray)) else float(ev)
        return attributions, expected_value

    def _explain_tree(self, x_row, target, background):
        import shap
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            explainer = shap.TreeExplainer(
                self.model, data=background, feature_perturbation="interventional",
            )
            shap_values = explainer.shap_values(x_row.reshape(1, -1))
        n_features = len(x_row)
        attributions = _extract_class_attributions(shap_values, target, n_features)
        ev = explainer.expected_value
        expected_value = float(ev[target]) if isinstance(ev, (list, np.ndarray)) else float(ev)
        return attributions, expected_value

    def explain_instance(
        self,
        x_row: np.ndarray,
        target: int | None = None,
        nsamples: int | str = "auto",
    ) -> ExplanationResult:
        protocol_value = None
        if self.schema.protocol_index is not None:
            protocol_value = x_row[self.schema.protocol_index]

        background = self._get_background(protocol_value)

        if target is None:
            target = self._predict_target(x_row)

        if self.backend == "kernel":
            attributions, expected_value = self._explain_kernel(x_row, target, nsamples, background)
        elif self.backend == "deep":
            attributions, expected_value = self._explain_deep(x_row, target, background)
        elif self.backend == "tree":
            attributions, expected_value = self._explain_tree(x_row, target, background)

        return ExplanationResult(
            feature_names=list(self.schema.feature_names),
            attributions=attributions,
            method="pa_shap",
            predicted_class=target,
            num_samples=nsamples if isinstance(nsamples, int) else None,
            expected_value=expected_value,
        )
