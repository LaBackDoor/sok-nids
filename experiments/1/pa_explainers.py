"""Protocol-Aware XAI explanation methods using pa_xai.

Drop-in replacements for explainers.py functions that enforce
network protocol domain constraints during explanation generation.
"""

import copy
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed

from explainers import ExplanationResult

logger = logging.getLogger(__name__)

# How many LIME instances to process before writing a checkpoint to disk.
# If the run crashes, at most this many instances of work are lost.
CHECKPOINT_BATCH_SIZE = 500


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


@contextmanager
def _disable_cudnn_for_rnn(model):
    """Disable cuDNN if model contains RNN layers (fused kernels don't support backward in eval)."""
    has_rnn = any(isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)) for m in model.modules())
    if has_rnn:
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        if has_rnn:
            torch.backends.cudnn.enabled = prev


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
    with _disable_cudnn_for_rnn(base_model):
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
    X_train: np.ndarray,
    dataset_name: str,
    model_name: str,
    config,
    checkpoint_dir=None,
    n_jobs: int | None = None,
) -> ExplanationResult:
    """Protocol-Aware LIME with auto backend selection and checkpointing.

    Backend selection:
      - ``loky`` (process-based) for tree models (RF/XGB) to bypass the GIL.
      - ``threading`` for neural-net models whose predict_fn forwards to GPU.

    Checkpointing:
      Every CHECKPOINT_BATCH_SIZE instances the accumulated attributions are
      flushed to *checkpoint_dir*.  On restart the completed work is reloaded
      so only the remaining instances are computed.
    """
    from pa_xai import ProtocolAwareLIME

    schema = _get_schema(dataset_name)
    n = len(X_explain)
    logger.info(f"  PA-LIME on {n} samples for {model_name}")

    explainer = ProtocolAwareLIME(schema, X_train=X_train)

    # loky (process-based) for tree models to bypass GIL;
    # threading for neural nets whose predict_fn forwards to GPU.
    is_tree = model_name.upper() in ("RF", "XGB")
    backend = "loky" if is_tree else "threading"

    if n_jobs is None:
        total_cpus = os.cpu_count() or 1
        n_jobs = max(1, int(total_cpus * 0.75))
    logger.info(f"  PA-LIME {model_name}: {n_jobs} workers, backend={backend}")

    # ── Resume from checkpoint if one exists ──
    completed: dict[int, np.ndarray] = {}
    ckpt_path: Path | None = None
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / f"lime_{dataset_name}_{model_name.lower()}.npz"
        if ckpt_path.exists():
            ckpt = np.load(ckpt_path)
            for idx, attr in zip(ckpt["indices"], ckpt["attributions"]):
                completed[int(idx)] = attr
            logger.info(f"  Resumed {len(completed)}/{n} from checkpoint")

    remaining = [i for i in range(n) if i not in completed]

    if not remaining:
        logger.info(f"  All {n} samples already checkpointed — skipping")
        attributions = np.stack([completed[i] for i in range(n)])
        if ckpt_path is not None:
            ckpt_path.unlink(missing_ok=True)
        return ExplanationResult(
            attributions=attributions, method_name="LIME", model_name=model_name,
            time_per_sample_ms=0.0, total_time_s=0.0,
        )

    logger.info(f"  {len(remaining)}/{n} samples remaining")

    def _explain_one(i):
        r = explainer.explain_instance(
            X_explain[i], predict_fn,
            num_samples=config.lime_num_samples,
        )
        return i, r.attributions

    start = time.time()

    # Process in checkpoint-sized batches, reusing the worker pool.
    with Parallel(n_jobs=n_jobs, backend=backend, verbose=1) as parallel:
        for batch_off in range(0, len(remaining), CHECKPOINT_BATCH_SIZE):
            batch = remaining[batch_off : batch_off + CHECKPOINT_BATCH_SIZE]

            batch_results = parallel(
                delayed(_explain_one)(i) for i in batch
            )

            for idx, attrs in batch_results:
                completed[idx] = attrs

            # Flush checkpoint to disk
            if ckpt_path is not None:
                sorted_keys = sorted(completed.keys())
                np.savez(
                    ckpt_path,
                    indices=np.array(sorted_keys, dtype=np.int64),
                    attributions=np.stack([completed[k] for k in sorted_keys]),
                )
                logger.info(f"  Checkpoint saved: {len(completed)}/{n} complete")

    elapsed = time.time() - start
    attributions = np.stack([completed[i] for i in range(n)])

    # Clean up checkpoint after successful completion
    if ckpt_path is not None and ckpt_path.exists():
        ckpt_path.unlink()
        logger.info("  Checkpoint cleaned up after successful completion")

    return ExplanationResult(
        attributions=attributions,
        method_name="LIME",
        model_name=model_name,
        time_per_sample_ms=(elapsed / n) * 1000,
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

    explainer = ProtocolAwareIG(schema, base_model, X_train)

    start = time.time()
    all_attrs = []
    with _disable_cudnn_for_rnn(base_model):
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
        schema, base_model, X_train,
    )

    start = time.time()
    all_attrs = []
    with _disable_cudnn_for_rnn(base_model):
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
    checkpoint_dir=None,
) -> tuple[list[ExplanationResult], np.ndarray]:
    """Generate protocol-aware explanations for all applicable models.

    Execution is structured in three phases to maximise hardware utilisation:

      Phase 1 — DNN SHAP (fast, sequential, needs a clean GPU).
      Phase 2 — **All LIMEs + tree SHAPs run concurrently.**
                DNN LIME forwards predictions to GPU 0; RF/XGB are CPU-only,
                so there is no resource conflict.
      Phase 3 — DNN IG ‖ DeepLIFT (fast, distributed across available GPUs).
    """
    from explainers import _fix_xgb_base_score

    n = min(config.num_explain_samples, len(dataset.X_test))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(dataset.X_test), size=n, replace=False)
    X_explain = dataset.X_test[indices]

    results: list[ExplanationResult] = []
    ds_name = dataset.dataset_name
    num_gpus = torch.cuda.device_count()

    # ── Phase 1: DNN SHAP (fast, sequential) ──────────────────────────
    if dnn_model is not None and dnn_wrapper is not None:
        logger.info("--- DNN PA-Explanations ---")
        try:
            results.append(pa_explain_shap_dnn(
                dnn_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, device, config,
            ))
        except Exception as e:
            logger.error(f"PA-SHAP DNN failed: {e}")

    # ── Phase 2: All LIMEs + tree SHAPs (concurrent) ─────────────────
    # Count how many LIME jobs will compete for CPU cores so we can
    # divide them fairly.
    lime_job_count = sum([
        dnn_wrapper is not None,
        rf_wrapper is not None,
        xgb_wrapper is not None,
    ])
    total_cpus = os.cpu_count() or 1
    usable_cpus = max(1, int(total_cpus * 0.75))
    n_jobs_each = max(1, usable_cpus // max(1, lime_job_count))

    if lime_job_count > 1:
        logger.info(
            f"  Running {lime_job_count} LIME jobs concurrently "
            f"({n_jobs_each} CPUs each, {usable_cpus} usable)"
        )

    if xgb_model is not None:
        _fix_xgb_base_score(xgb_model)

    # +2 headroom for the tree-SHAP tasks that finish quickly
    max_workers = max(1, lime_job_count + 2)
    lime_futures: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        if dnn_wrapper is not None:
            lime_futures["PA-LIME DNN"] = pool.submit(
                pa_explain_lime, dnn_wrapper.predict_proba,
                X_explain, dataset.X_train, ds_name, "DNN", config,
                checkpoint_dir,
                n_jobs_each if lime_job_count > 1 else None,
            )

        if rf_model is not None and rf_wrapper is not None:
            logger.info("--- RF PA-Explanations ---")
            lime_futures["PA-SHAP RF"] = pool.submit(
                pa_explain_shap_tree, rf_model, X_explain,
                dataset.X_train, dataset.y_train, ds_name, config, "RF",
            )
            lime_futures["PA-LIME RF"] = pool.submit(
                pa_explain_lime, rf_wrapper.predict_proba,
                X_explain, dataset.X_train, ds_name, "RF", config,
                checkpoint_dir,
                n_jobs_each if lime_job_count > 1 else None,
            )

        if xgb_model is not None and xgb_wrapper is not None:
            logger.info("--- XGBoost PA-Explanations ---")
            lime_futures["PA-SHAP XGB"] = pool.submit(
                pa_explain_shap_tree, xgb_model, X_explain,
                dataset.X_train, dataset.y_train, ds_name, config, "XGB",
            )
            lime_futures["PA-LIME XGB"] = pool.submit(
                pa_explain_lime, xgb_wrapper.predict_proba,
                X_explain, dataset.X_train, ds_name, "XGB", config,
                checkpoint_dir,
                n_jobs_each if lime_job_count > 1 else None,
            )

    for name, future in lime_futures.items():
        try:
            results.append(future.result())
        except Exception as e:
            logger.error(f"{name} failed: {e}")

    # ── Phase 3: DNN IG ‖ DeepLIFT (fast, GPU-bound) ─────────────────
    if dnn_model is not None:
        if num_gpus >= 3:
            ig_clone = _clone_model_to_device(dnn_model, torch.device("cuda:1"))
            dl_clone = _clone_model_to_device(dnn_model, torch.device("cuda:2"))
            logger.info("  Running PA-IG (GPU 1) || PA-DeepLIFT (GPU 2)")
            with ThreadPoolExecutor(max_workers=2) as pool:
                ig_future = pool.submit(
                    pa_explain_ig, ig_clone, X_explain,
                    dataset.X_train, dataset.y_train,
                    ds_name, torch.device("cuda:1"), config,
                )
                dl_future = pool.submit(
                    pa_explain_deeplift, dl_clone, X_explain,
                    dataset.X_train, dataset.y_train,
                    ds_name, torch.device("cuda:2"), config,
                )
            try:
                results.append(ig_future.result())
            except Exception as e:
                logger.error(f"PA-IG failed: {e}")
            try:
                results.append(dl_future.result())
            except Exception as e:
                logger.error(f"PA-DeepLIFT failed: {e}")
            del ig_clone, dl_clone
            torch.cuda.empty_cache()

        elif num_gpus >= 2:
            gpu1 = torch.device("cuda:1")
            dl_clone = _clone_model_to_device(dnn_model, gpu1)
            logger.info("  Running PA-IG (GPU 0) || PA-DeepLIFT (GPU 1)")
            with ThreadPoolExecutor(max_workers=2) as pool:
                ig_future = pool.submit(
                    pa_explain_ig, dnn_model, X_explain,
                    dataset.X_train, dataset.y_train,
                    ds_name, device, config,
                )
                dl_future = pool.submit(
                    pa_explain_deeplift, dl_clone, X_explain,
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
            del dl_clone
            torch.cuda.empty_cache()

        else:
            # 1 GPU or CPU-only: sequential
            for name, fn, args in [
                ("PA-IG DNN", pa_explain_ig, (
                    dnn_model, X_explain, dataset.X_train, dataset.y_train,
                    ds_name, device, config)),
                ("PA-DeepLIFT DNN", pa_explain_deeplift, (
                    dnn_model, X_explain, dataset.X_train, dataset.y_train,
                    ds_name, device, config)),
            ]:
                try:
                    results.append(fn(*args))
                except Exception as e:
                    logger.error(f"{name} failed: {e}")

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
    checkpoint_dir=None,
) -> None:
    """Generate protocol-aware explanations for a CNN model."""
    ds_name = dataset.dataset_name
    num_gpus = torch.cuda.device_count()

    if num_gpus >= 3:
        # 3+ GPUs: all 4 methods in parallel, each GPU method on its own GPU.
        gpus = [torch.device(f"cuda:{i}") for i in range(3)]
        shap_clone = _clone_model_to_device(flat_model, gpus[0])
        ig_clone = _clone_model_to_device(flat_model, gpus[1])
        dl_clone = _clone_model_to_device(flat_model, gpus[2])
        logger.info(
            f"  {model_name}: SHAP(GPU 0) || LIME(CPU) "
            f"|| IG(GPU 1) || DeepLIFT(GPU 2)"
        )

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                "PA-SHAP": pool.submit(
                    pa_explain_shap_dnn, shap_clone, X_explain,
                    dataset.X_train, dataset.y_train,
                    ds_name, gpus[0], config,
                ),
                "PA-LIME": pool.submit(
                    pa_explain_lime, wrapper.predict_proba,
                    X_explain, dataset.X_train, ds_name, model_name, config,
                    checkpoint_dir,
                ),
                "PA-IG": pool.submit(
                    pa_explain_ig, ig_clone, X_explain,
                    dataset.X_train, dataset.y_train,
                    ds_name, gpus[1], config, model_name,
                ),
                "PA-DeepLIFT": pool.submit(
                    pa_explain_deeplift, dl_clone, X_explain,
                    dataset.X_train, dataset.y_train,
                    ds_name, gpus[2], config, model_name,
                ),
            }

        for name, future in futures.items():
            try:
                r = future.result()
                r.model_name = model_name
                results.append(r)
            except Exception as e:
                logger.warning(f"  {name} failed for {model_name}: {e}")

        del shap_clone, ig_clone, dl_clone
        torch.cuda.empty_cache()

    elif num_gpus >= 2:
        # 2 GPUs: SHAP+LIME sequential, then IG(GPU 0) || DeepLIFT(GPU 1)
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
                wrapper.predict_proba, X_explain, dataset.X_train, ds_name, model_name, config,
                checkpoint_dir=checkpoint_dir,
            )
            r.model_name = model_name
            results.append(r)
        except Exception as e:
            logger.warning(f"  PA-LIME failed for {model_name}: {e}")

        gpu1 = torch.device("cuda:1")
        dl_clone = _clone_model_to_device(flat_model, gpu1)
        logger.info(f"  Running PA-IG (GPU 0) || PA-DeepLIFT (GPU 1) for {model_name}")

        with ThreadPoolExecutor(max_workers=2) as pool:
            ig_future = pool.submit(
                pa_explain_ig, flat_model, X_explain,
                dataset.X_train, dataset.y_train,
                ds_name, device, config, model_name,
            )
            dl_future = pool.submit(
                pa_explain_deeplift, dl_clone, X_explain,
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

        del dl_clone
        torch.cuda.empty_cache()

    else:
        # 1 GPU: sequential
        for mname, fn, args in [
            ("PA-SHAP", pa_explain_shap_dnn, (
                flat_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, device, config)),
            ("PA-LIME", pa_explain_lime, (
                wrapper.predict_proba, X_explain, dataset.X_train, ds_name, model_name, config,
                checkpoint_dir)),
            ("PA-IG", pa_explain_ig, (
                flat_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, device, config, model_name)),
            ("PA-DeepLIFT", pa_explain_deeplift, (
                flat_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, device, config, model_name)),
        ]:
            try:
                r = fn(*args)
                r.model_name = model_name
                results.append(r)
            except Exception as e:
                logger.warning(f"  {mname} failed for {model_name}: {e}")


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
                wrapper.predict_proba, X, dataset.X_train, ds_name, model_type, config,
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
