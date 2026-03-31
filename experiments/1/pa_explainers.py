"""Protocol-Aware XAI explanation methods using pa_xai.

Drop-in replacements for explainers.py functions that enforce
network protocol domain constraints during explanation generation.
"""

import copy
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from joblib import Parallel, delayed

from explainers import ExplanationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset name -> pa_xai schema mapping
# ---------------------------------------------------------------------------

DATASET_SCHEMA_MAP = {
    "nsl-kdd": "NSL-KDD",
    "cic-ids-2017": "CIC-IDS-2017",
    "unsw-nb15": "UNSW-NB15-CICFlowMeter",
    "cse-cic-ids2018": "CSE-CIC-IDS2018",
}


def _get_schema(dataset_name: str):
    from pa_xai import get_schema
    return get_schema(DATASET_SCHEMA_MAP[dataset_name])


def _clone_model_to_device(model: torch.nn.Module, target_device: torch.device):
    """Deep-copy a model to a different GPU. Returns the clone in eval mode."""
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    cloned = copy.deepcopy(base)
    cloned.to(target_device)
    cloned.eval()
    return cloned


# ---------------------------------------------------------------------------
# Protocol-Aware SHAP (DNN -- DeepExplainer backend)
# ---------------------------------------------------------------------------

def pa_explain_shap_dnn(
    model: torch.nn.Module,
    X_explain: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    dataset_name: str,
    device: torch.device,
    config,
) -> ExplanationResult:
    """Protocol-Aware SHAP for DNN using DeepExplainer backend."""
    from pa_xai import ProtocolAwareSHAP

    schema = _get_schema(dataset_name)
    logger.info(f"  PA-SHAP (DeepExplainer) on {len(X_explain)} samples")

    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    base_model.eval()

    explainer = ProtocolAwareSHAP(
        schema, base_model, X_train, y_train,
        backend="deep", n_background=config.shap_background_samples,
    )

    start = time.time()
    all_attrs = []
    for i in range(len(X_explain)):
        result = explainer.explain_instance(X_explain[i])
        all_attrs.append(result.attributions)
    elapsed = time.time() - start

    attributions = np.stack(all_attrs, axis=0)

    return ExplanationResult(
        attributions=attributions,
        method_name="SHAP",
        model_name="DNN",
        time_per_sample_ms=(elapsed / len(X_explain)) * 1000,
        total_time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Protocol-Aware SHAP (RF/XGB -- TreeExplainer backend)
# ---------------------------------------------------------------------------

def pa_explain_shap_tree(
    model,
    X_explain: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    dataset_name: str,
    config,
    model_name: str = "RF",
) -> ExplanationResult:
    """Protocol-Aware SHAP for tree models using TreeExplainer backend."""
    from pa_xai import ProtocolAwareSHAP

    schema = _get_schema(dataset_name)
    logger.info(f"  PA-SHAP (TreeExplainer) on {len(X_explain)} samples for {model_name}")

    explainer = ProtocolAwareSHAP(
        schema, model, X_train, y_train,
        backend="tree", n_background=config.shap_background_samples,
    )

    total_cpus = os.cpu_count() or 1
    n_jobs = max(1, int(total_cpus * 0.75))
    logger.info(f"  PA-SHAP Tree parallelizing with {n_jobs}/{total_cpus} CPUs")

    def _explain_one(i):
        return explainer.explain_instance(X_explain[i]).attributions

    start = time.time()
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=1)(
        delayed(_explain_one)(i) for i in range(len(X_explain))
    )
    elapsed = time.time() - start

    attributions = np.stack(results, axis=0)

    return ExplanationResult(
        attributions=attributions,
        method_name="SHAP",
        model_name=model_name,
        time_per_sample_ms=(elapsed / len(X_explain)) * 1000,
        total_time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Protocol-Aware LIME
# ---------------------------------------------------------------------------

def pa_explain_lime(
    predict_fn,
    X_explain: np.ndarray,
    dataset_name: str,
    model_name: str,
    config,
) -> ExplanationResult:
    """Protocol-Aware LIME explanations."""
    from pa_xai import ProtocolAwareLIME

    schema = _get_schema(dataset_name)
    logger.info(f"  PA-LIME on {len(X_explain)} samples for {model_name}")

    explainer = ProtocolAwareLIME(schema)

    total_cpus = os.cpu_count() or 1
    n_jobs = max(1, int(total_cpus * 0.75))
    logger.info(f"  PA-LIME parallelizing with {n_jobs}/{total_cpus} CPUs")

    def _explain_one(i):
        r = explainer.explain_instance(
            X_explain[i], predict_fn,
            num_samples=config.lime_num_samples,
        )
        return r.attributions

    start = time.time()
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=1)(
        delayed(_explain_one)(i) for i in range(len(X_explain))
    )
    elapsed = time.time() - start

    attributions = np.stack(results, axis=0)

    return ExplanationResult(
        attributions=attributions,
        method_name="LIME",
        model_name=model_name,
        time_per_sample_ms=(elapsed / len(X_explain)) * 1000,
        total_time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Protocol-Aware Integrated Gradients
# ---------------------------------------------------------------------------

def pa_explain_ig(
    model: torch.nn.Module,
    X_explain: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    dataset_name: str,
    device: torch.device,
    config,
    model_name: str = "DNN",
) -> ExplanationResult:
    """Protocol-Aware Integrated Gradients."""
    from pa_xai import ProtocolAwareIG

    schema = _get_schema(dataset_name)
    logger.info(f"  PA-IG on {len(X_explain)} samples for {model_name}")

    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    base_model.eval()

    explainer = ProtocolAwareIG(
        schema, base_model, X_train, y_train,
        constrain_path=True,
    )

    start = time.time()
    all_attrs = []
    for i in range(len(X_explain)):
        result = explainer.explain_instance(
            X_explain[i].astype(np.float32),
            n_steps=config.ig_n_steps,
        )
        all_attrs.append(result.attributions)
    elapsed = time.time() - start

    attributions = np.stack(all_attrs, axis=0)

    return ExplanationResult(
        attributions=attributions,
        method_name="IG",
        model_name=model_name,
        time_per_sample_ms=(elapsed / len(X_explain)) * 1000,
        total_time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Protocol-Aware DeepLIFT
# ---------------------------------------------------------------------------

def pa_explain_deeplift(
    model: torch.nn.Module,
    X_explain: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    dataset_name: str,
    device: torch.device,
    config,
    model_name: str = "DNN",
) -> ExplanationResult:
    """Protocol-Aware DeepLIFT."""
    from pa_xai import ProtocolAwareDeepLIFT

    schema = _get_schema(dataset_name)
    logger.info(f"  PA-DeepLIFT on {len(X_explain)} samples for {model_name}")

    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    base_model.eval()

    explainer = ProtocolAwareDeepLIFT(
        schema, base_model, X_train, y_train,
    )

    start = time.time()
    all_attrs = []
    for i in range(len(X_explain)):
        result = explainer.explain_instance(
            X_explain[i].astype(np.float32),
        )
        all_attrs.append(result.attributions)
    elapsed = time.time() - start

    attributions = np.stack(all_attrs, axis=0)

    return ExplanationResult(
        attributions=attributions,
        method_name="DeepLIFT",
        model_name=model_name,
        time_per_sample_ms=(elapsed / len(X_explain)) * 1000,
        total_time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Orchestrator (mirrors generate_all_explanations)
# ---------------------------------------------------------------------------

def pa_generate_all_explanations(
    dnn_model: torch.nn.Module,
    rf_model,
    dnn_wrapper,
    rf_wrapper,
    dataset,
    device: torch.device,
    config,
    xgb_model=None,
    xgb_wrapper=None,
) -> tuple[list[ExplanationResult], np.ndarray]:
    """Generate protocol-aware explanations for all applicable models."""
    from explainers import _fix_xgb_base_score

    n = min(config.num_explain_samples, len(dataset.X_test))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(dataset.X_test), size=n, replace=False)
    X_explain = dataset.X_test[indices]

    results = []
    ds_name = dataset.dataset_name

    # === DNN explanations ===
    if dnn_model is not None and dnn_wrapper is not None:
        logger.info("--- DNN PA-Explanations ---")
        try:
            results.append(pa_explain_shap_dnn(
                dnn_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, device, config,
            ))
        except Exception as e:
            logger.error(f"PA-SHAP DNN failed: {e}")

        try:
            results.append(pa_explain_lime(
                dnn_wrapper.predict_proba, X_explain, ds_name, "DNN", config,
            ))
        except Exception as e:
            logger.error(f"PA-LIME DNN failed: {e}")

        # IG and DeepLIFT: parallel on 2 GPUs if available.
        # Each gets its own model copy on a separate device so Captum hooks
        # don't conflict.
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            gpu1 = torch.device("cuda:1")
            model_gpu1 = _clone_model_to_device(dnn_model, gpu1)
            logger.info("  Running PA-IG (GPU 0) || PA-DeepLIFT (GPU 1) in parallel")

            with ThreadPoolExecutor(max_workers=2) as pool:
                ig_future = pool.submit(
                    pa_explain_ig, dnn_model, X_explain,
                    dataset.X_train, dataset.y_train,
                    ds_name, device, config,
                )
                dl_future = pool.submit(
                    pa_explain_deeplift, model_gpu1, X_explain,
                    dataset.X_train, dataset.y_train,
                    ds_name, gpu1, config,
                )

            try:
                results.append(ig_future.result())
            except Exception as e:
                logger.error(f"PA-IG failed: {e}")
            try:
                results.append(dl_future.result())
            except Exception as e:
                logger.error(f"PA-DeepLIFT failed: {e}")

            del model_gpu1
            torch.cuda.empty_cache()
        else:
            try:
                results.append(pa_explain_ig(
                    dnn_model, X_explain, dataset.X_train, dataset.y_train,
                    ds_name, device, config,
                ))
            except Exception as e:
                logger.error(f"PA-IG failed: {e}")

            try:
                results.append(pa_explain_deeplift(
                    dnn_model, X_explain, dataset.X_train, dataset.y_train,
                    ds_name, device, config,
                ))
            except Exception as e:
                logger.error(f"PA-DeepLIFT failed: {e}")

    # === RF explanations ===
    if rf_model is not None and rf_wrapper is not None:
        logger.info("--- RF PA-Explanations ---")
        try:
            results.append(pa_explain_shap_tree(
                rf_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, config, model_name="RF",
            ))
        except Exception as e:
            logger.error(f"PA-SHAP RF failed: {e}")

        try:
            results.append(pa_explain_lime(
                rf_wrapper.predict_proba, X_explain, ds_name, "RF", config,
            ))
        except Exception as e:
            logger.error(f"PA-LIME RF failed: {e}")

    # === XGBoost explanations ===
    if xgb_model is not None and xgb_wrapper is not None:
        logger.info("--- XGBoost PA-Explanations ---")
        try:
            _fix_xgb_base_score(xgb_model)
            results.append(pa_explain_shap_tree(
                xgb_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, config, model_name="XGB",
            ))
        except Exception as e:
            logger.error(f"PA-SHAP XGB failed: {e}")

        try:
            results.append(pa_explain_lime(
                xgb_wrapper.predict_proba, X_explain, ds_name, "XGB", config,
            ))
        except Exception as e:
            logger.error(f"PA-LIME XGB failed: {e}")

    logger.info(f"Generated {len(results)} PA explanation sets")
    return results, indices


# ---------------------------------------------------------------------------
# CNN explanations (protocol-aware)
# ---------------------------------------------------------------------------

def pa_generate_cnn_explanations(
    results: list[ExplanationResult],
    flat_model: torch.nn.Module,
    wrapper,
    model_name: str,
    X_explain: np.ndarray,
    dataset,
    device: torch.device,
    config,
) -> None:
    """Generate protocol-aware explanations for a CNN model."""
    ds_name = dataset.dataset_name

    try:
        r = pa_explain_shap_dnn(
            flat_model, X_explain, dataset.X_train, dataset.y_train,
            ds_name, device, config,
        )
        r.model_name = model_name
        results.append(r)
    except Exception as e:
        logger.warning(f"  PA-SHAP failed for {model_name}: {e}")

    try:
        r = pa_explain_lime(
            wrapper.predict_proba, X_explain, ds_name, model_name, config,
        )
        r.model_name = model_name
        results.append(r)
    except Exception as e:
        logger.warning(f"  PA-LIME failed for {model_name}: {e}")

    # IG and DeepLIFT: parallel on 2 GPUs if available
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        gpu1 = torch.device("cuda:1")
        model_gpu1 = _clone_model_to_device(flat_model, gpu1)
        logger.info(f"  Running PA-IG (GPU 0) || PA-DeepLIFT (GPU 1) for {model_name}")

        with ThreadPoolExecutor(max_workers=2) as pool:
            ig_future = pool.submit(
                pa_explain_ig, flat_model, X_explain,
                dataset.X_train, dataset.y_train,
                ds_name, device, config, model_name,
            )
            dl_future = pool.submit(
                pa_explain_deeplift, model_gpu1, X_explain,
                dataset.X_train, dataset.y_train,
                ds_name, gpu1, config, model_name,
            )

        try:
            r = ig_future.result()
            r.model_name = model_name
            results.append(r)
        except Exception as e:
            logger.warning(f"  PA-IG failed for {model_name}: {e}")
        try:
            r = dl_future.result()
            r.model_name = model_name
            results.append(r)
        except Exception as e:
            logger.warning(f"  PA-DeepLIFT failed for {model_name}: {e}")

        del model_gpu1
        torch.cuda.empty_cache()
    else:
        try:
            r = pa_explain_ig(
                flat_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, device, config, model_name=model_name,
            )
            r.model_name = model_name
            results.append(r)
        except Exception as e:
            logger.warning(f"  PA-IG failed for {model_name}: {e}")

        try:
            r = pa_explain_deeplift(
                flat_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, device, config, model_name=model_name,
            )
            r.model_name = model_name
            results.append(r)
        except Exception as e:
            logger.warning(f"  PA-DeepLIFT failed for {model_name}: {e}")


# ---------------------------------------------------------------------------
# Make explain function for evaluation (protocol-aware)
# ---------------------------------------------------------------------------

def pa_make_explain_fn(
    method_name: str,
    model_type: str,
    dnn_model,
    wrapper,
    dataset,
    device: torch.device,
    config,
    rf_model=None,
):
    """Create a callable explain function for metric evaluation (PA mode)."""
    ds_name = dataset.dataset_name

    if method_name == "SHAP" and model_type in ("DNN", "CNN-LSTM", "CNN-GRU"):
        def fn(X):
            r = pa_explain_shap_dnn(
                dnn_model, X, dataset.X_train, dataset.y_train,
                ds_name, device, config,
            )
            return r.attributions
    elif method_name == "SHAP" and model_type in ("RF", "XGB"):
        def fn(X):
            r = pa_explain_shap_tree(
                rf_model, X, dataset.X_train, dataset.y_train,
                ds_name, config, model_name=model_type,
            )
            return r.attributions
    elif method_name == "LIME":
        def fn(X):
            r = pa_explain_lime(
                wrapper.predict_proba, X, ds_name, model_type, config,
            )
            return r.attributions
    elif method_name == "IG":
        def fn(X):
            r = pa_explain_ig(
                dnn_model, X, dataset.X_train, dataset.y_train,
                ds_name, device, config, model_name=model_type,
            )
            return r.attributions
    elif method_name == "DeepLIFT":
        def fn(X):
            r = pa_explain_deeplift(
                dnn_model, X, dataset.X_train, dataset.y_train,
                ds_name, device, config, model_name=model_type,
            )
            return r.attributions
    else:
        raise ValueError(f"Unknown method: {method_name}")

    return fn
