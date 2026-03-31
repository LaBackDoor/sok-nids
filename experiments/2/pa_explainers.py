"""Protocol-Aware XAI explainers for Experiment 2 robustness evaluation.

Wraps pa_xai explainer classes into the fn(X) -> np.ndarray interface
that robustness.py expects. Each function returns a closure that generates
attributions for a batch of samples.
"""

import copy
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

DATASET_SCHEMA_MAP = {
    "nsl-kdd": "NSL-KDD",
    "cic-ids-2017": "CIC-IDS-2017",
    "unsw-nb15": "UNSW-NB15-CICFlowMeter",
    "cse-cic-ids2018": "CSE-CIC-IDS2018",
}


def _get_schema(dataset_name: str):
    from pa_xai import get_schema
    return get_schema(DATASET_SCHEMA_MAP[dataset_name])


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DataParallel and return model in eval mode."""
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    base.eval()
    return base


def _clone_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Deep-copy a model to avoid Captum hook conflicts."""
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    cloned = copy.deepcopy(base)
    cloned.to(device)
    cloned.eval()
    return cloned


def make_pa_explain_fn(
    method: str,
    dnn_model: torch.nn.Module,
    dnn_wrapper,
    dataset,
    device: torch.device,
    config,
):
    """Create a PA explain function for robustness evaluation.

    Args:
        method: One of "PA-SHAP", "PA-LIME", "PA-IG", "PA-DeepLIFT".
        dnn_model: Trained DNN (NIDSNet) model.
        dnn_wrapper: DNNWrapper with predict_proba method.
        dataset: DatasetBundle with X_train, y_train, dataset_name.
        device: Torch device.
        config: Experiment2Config (uses config.robustness sub-config).

    Returns:
        callable(X: np.ndarray) -> np.ndarray of shape (n_samples, n_features)

    Raises:
        ValueError: If dataset has no pa_xai schema or method is unknown.
    """
    ds_name = dataset.dataset_name
    if ds_name not in DATASET_SCHEMA_MAP:
        raise ValueError(
            f"No pa_xai schema for '{ds_name}'. Cannot create PA explainer."
        )

    rob_cfg = config.robustness

    if method == "PA-SHAP":
        return _make_pa_shap_fn(
            dnn_model, dataset, device, rob_cfg
        )
    elif method == "PA-LIME":
        return _make_pa_lime_fn(
            dnn_wrapper, dataset, rob_cfg
        )
    elif method == "PA-IG":
        return _make_pa_ig_fn(
            dnn_model, dataset, device, rob_cfg
        )
    elif method == "PA-DeepLIFT":
        return _make_pa_deeplift_fn(
            dnn_model, dataset, device, rob_cfg
        )
    else:
        raise ValueError(f"Unknown PA method: {method}")


def _make_pa_shap_fn(dnn_model, dataset, device, rob_cfg):
    from pa_xai import ProtocolAwareSHAP

    schema = _get_schema(dataset.dataset_name)
    base_model = _unwrap_model(dnn_model)

    explainer = ProtocolAwareSHAP(
        schema, base_model, dataset.X_train, dataset.y_train,
        backend="deep", n_background=rob_cfg.shap_background_samples,
    )

    def fn(X: np.ndarray) -> np.ndarray:
        all_attrs = []
        for i in range(len(X)):
            result = explainer.explain_instance(X[i])
            all_attrs.append(result.attributions)
        return np.stack(all_attrs, axis=0)

    return fn


def _make_pa_lime_fn(dnn_wrapper, dataset, rob_cfg):
    from pa_xai import ProtocolAwareLIME

    schema = _get_schema(dataset.dataset_name)
    explainer = ProtocolAwareLIME(schema)

    def fn(X: np.ndarray) -> np.ndarray:
        all_attrs = []
        for i in range(len(X)):
            result = explainer.explain_instance(
                X[i], dnn_wrapper.predict_proba,
                num_samples=rob_cfg.lime_num_samples,
            )
            all_attrs.append(result.attributions)
        return np.stack(all_attrs, axis=0)

    return fn


def _make_pa_ig_fn(dnn_model, dataset, device, rob_cfg):
    from pa_xai import ProtocolAwareIG

    schema = _get_schema(dataset.dataset_name)
    # Clone model to avoid Captum hook conflicts with other methods
    model_clone = _clone_model(dnn_model, device)

    explainer = ProtocolAwareIG(
        schema, model_clone, dataset.X_train, dataset.y_train,
        constrain_path=True,
    )

    def fn(X: np.ndarray) -> np.ndarray:
        all_attrs = []
        for i in range(len(X)):
            result = explainer.explain_instance(
                X[i].astype(np.float32),
                n_steps=rob_cfg.ig_n_steps,
            )
            all_attrs.append(result.attributions)
        return np.stack(all_attrs, axis=0)

    return fn


def _make_pa_deeplift_fn(dnn_model, dataset, device, rob_cfg):
    from pa_xai import ProtocolAwareDeepLIFT

    schema = _get_schema(dataset.dataset_name)
    # Clone model to avoid Captum hook conflicts with other methods
    model_clone = _clone_model(dnn_model, device)

    explainer = ProtocolAwareDeepLIFT(
        schema, model_clone, dataset.X_train, dataset.y_train,
    )

    def fn(X: np.ndarray) -> np.ndarray:
        all_attrs = []
        for i in range(len(X)):
            result = explainer.explain_instance(
                X[i].astype(np.float32),
            )
            all_attrs.append(result.attributions)
        return np.stack(all_attrs, axis=0)

    return fn
