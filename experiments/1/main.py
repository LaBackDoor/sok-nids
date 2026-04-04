#!/usr/bin/env python3
"""Experiment 1: Quantitative Benchmarking of Explanation Quality.

Run the full pipeline: data loading -> model training -> XAI explanations -> metric evaluation.

Usage:
    # Full experiment on all datasets
    python experiments/1/main.py

    # Specific dataset(s)
    python experiments/1/main.py --datasets nsl-kdd cic-ids-2017

    # Specific phase
    python experiments/1/main.py --phase train
    python experiments/1/main.py --phase explain
    python experiments/1/main.py --phase evaluate

    # Reduced sample count for faster testing
    python experiments/1/main.py --num-explain-samples 100
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch

# Suppress noisy sklearn/joblib compatibility warning from third-party libs
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

# Add experiment directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig, load_config
from data_loader import DatasetBundle, load_dataset
from explainers import (
    ExplanationResult,
    explain_deeplift,
    explain_ig,
    explain_lime,
    explain_shap_dnn,
    explain_shap_rf,
    generate_all_explanations,
    generate_and_time_summary_plots,
)
from pa_explainers import (
    pa_generate_all_explanations,
    pa_generate_cnn_explanations,
    pa_make_explain_fn,
)
from metrics import evaluate_all_metrics
from models import (
    CNNGRU,
    CNNLSTM,
    CNNGRUWrapper,
    CNNLSTMWrapper,
    DNNWrapper,
    FlatCNNGRU,
    FlatCNNLSTM,
    NIDSNet,
    RFWrapper,
    SoftmaxModel,
    XGBWrapper,
    _compute_grid_size,
    load_cnn_models,
    load_models,
    save_cnn_models,
    save_models,
    train_cnn_gru,
    train_cnn_lstm,
    train_dnn,
    train_rf,
    train_xgb,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("experiment1")


def setup_device() -> tuple[torch.device, int]:
    """Detect available GPUs and set up device."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device("cuda")
        logger.info(f"CUDA available: {num_gpus} GPU(s) detected")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"  GPU {i}: {props.name} | "
                f"{props.total_memory / 1e9:.1f} GB | "
                f"Compute {props.major}.{props.minor} | "
                f"SMs: {props.multi_processor_count}"
            )
        logger.info(f"  PyTorch CUDA version: {torch.version.cuda}")
        logger.info(f"  cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    else:
        device = torch.device("cpu")
        num_gpus = 0
        logger.warning("No CUDA GPUs available. Running on CPU.")
    return device, num_gpus


def log_gpu_memory(label: str = "") -> None:
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        logger.info(f"  GPU {i} memory [{label}]: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved")


ALL_MODELS = ["dnn", "rf", "xgb", "cnn-lstm", "cnn-gru"]


def phase_train(
    dataset: DatasetBundle, config: ExperimentConfig, device: torch.device,
    num_gpus: int, selected_models: list[str] | None = None,
) -> dict:
    """Train selected models on the dataset."""
    models_to_train = selected_models or ALL_MODELS
    logger.info(f"=== TRAINING on {dataset.dataset_name} (models: {models_to_train}) ===")
    log_gpu_memory("pre-train")
    output_dir = config.output_dir / dataset.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_results = {}
    dnn_model = rf_model = xgb_model = None
    xgb_label_map = None
    cnn_lstm_model = cnn_gru_model = None

    if "dnn" in models_to_train:
        dnn_model, dnn_wrapper, dnn_metrics = train_dnn(dataset, config.dnn, device, num_gpus)
        train_results["dnn"] = dnn_metrics
        log_gpu_memory("post-DNN-train")

    if "rf" in models_to_train:
        rf_model, rf_wrapper, rf_metrics = train_rf(dataset, config.rf)
        train_results["rf"] = rf_metrics

    if "xgb" in models_to_train:
        xgb_model, xgb_wrapper, xgb_metrics = train_xgb(dataset, config.xgb)
        train_results["xgb"] = xgb_metrics
        xgb_label_map = xgb_wrapper.label_map

    if "cnn-lstm" in models_to_train:
        cnn_lstm_model, cnn_lstm_wrapper, cnn_lstm_metrics = train_cnn_lstm(
            dataset, config.cnn_lstm, device, num_gpus,
        )
        train_results["cnn_lstm"] = cnn_lstm_metrics
        log_gpu_memory("post-CNN-LSTM-train")

    if "cnn-gru" in models_to_train:
        cnn_gru_model, cnn_gru_wrapper, cnn_gru_metrics = train_cnn_gru(
            dataset, config.cnn_gru, device, num_gpus,
        )
        train_results["cnn_gru"] = cnn_gru_metrics
        log_gpu_memory("post-CNN-GRU-train")

    # Save models
    if dnn_model is not None or rf_model is not None:
        save_models(
            dnn_model, rf_model, config.models_dir, dataset.dataset_name,
            xgb_model=xgb_model, xgb_label_map=xgb_label_map,
        )
    if cnn_lstm_model is not None or cnn_gru_model is not None:
        save_cnn_models(
            config.models_dir, dataset.dataset_name,
            cnn_lstm_model=cnn_lstm_model,
            cnn_lstm_config=config.cnn_lstm if cnn_lstm_model else None,
            cnn_gru_model=cnn_gru_model,
            cnn_gru_config=config.cnn_gru if cnn_gru_model else None,
        )

    # Save training metrics
    results_path = output_dir / "train_metrics.json"
    with open(results_path, "w") as f:
        json.dump(train_results, f, indent=2, default=_json_serialize)
    logger.info(f"Training metrics saved to {results_path}")

    return train_results


def phase_explain(
    dataset: DatasetBundle, config: ExperimentConfig, device: torch.device,
    selected_models: list[str] | None = None,
) -> tuple[list[ExplanationResult], np.ndarray]:
    """Generate XAI explanations using all methods."""
    models_to_explain = selected_models or ALL_MODELS
    logger.info(f"=== EXPLAINING on {dataset.dataset_name} (models: {models_to_explain}) ===")
    log_gpu_memory("pre-explain")
    output_dir = config.output_dir / dataset.dataset_name

    # Load trained models (only those selected)
    dnn_model = rf_model = xgb_model = xgb_label_map = None
    dnn_wrapper = rf_wrapper = xgb_wrapper = None

    if any(m in models_to_explain for m in ["dnn", "rf", "xgb"]):
        dnn_model, rf_model, xgb_model, xgb_label_map = load_models(
            config.models_dir, dataset.dataset_name,
            dataset.X_train.shape[1], dataset.num_classes,
            config.dnn, device,
        )
        if "dnn" in models_to_explain:
            dnn_wrapper = DNNWrapper(dnn_model, device)
        if "rf" in models_to_explain:
            rf_wrapper = RFWrapper(rf_model, num_classes=dataset.num_classes)
        if "xgb" in models_to_explain and xgb_model is not None:
            xgb_wrapper = XGBWrapper(xgb_model, num_classes=dataset.num_classes, label_map=xgb_label_map)

    # Load CNN models if selected
    cnn_lstm_model = cnn_lstm_wrapper = cnn_lstm_flat = None
    cnn_gru_model = cnn_gru_wrapper = cnn_gru_flat = None
    if any(m in models_to_explain for m in ["cnn-lstm", "cnn-gru"]):
        cnn_lstm_raw, cnn_lstm_cfg, cnn_gru_raw, cnn_gru_cfg = load_cnn_models(
            config.models_dir, dataset.dataset_name, dataset.num_classes, device,
        )
        if "cnn-lstm" in models_to_explain and cnn_lstm_raw is not None:
            gs = cnn_lstm_cfg.grid_size
            cnn_lstm_model = cnn_lstm_raw
            cnn_lstm_wrapper = CNNLSTMWrapper(cnn_lstm_raw, device, gs)
            cnn_lstm_flat = FlatCNNLSTM(cnn_lstm_raw, gs)
        if "cnn-gru" in models_to_explain and cnn_gru_raw is not None:
            gs = cnn_gru_cfg.input_spatial_size
            cnn_gru_model = cnn_gru_raw
            cnn_gru_wrapper = CNNGRUWrapper(cnn_gru_raw, device, gs)
            cnn_gru_flat = FlatCNNGRU(cnn_gru_raw, gs)

    # Use full training data for explanations (1TB RAM available)

    # Checkpoint directory for resumable LIME computation
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Generate explanations for classic models
    if config.xai_mode == "p":
        results, indices = pa_generate_all_explanations(
            dnn_model if "dnn" in models_to_explain else None,
            rf_model if "rf" in models_to_explain else None,
            dnn_wrapper, rf_wrapper,
            dataset, device, config.explainer,
            xgb_model=xgb_model if "xgb" in models_to_explain else None,
            xgb_wrapper=xgb_wrapper,
            checkpoint_dir=checkpoint_dir,
        )
    else:
        results, indices = generate_all_explanations(
            dnn_model if "dnn" in models_to_explain else None,
            rf_model if "rf" in models_to_explain else None,
            dnn_wrapper, rf_wrapper,
            dataset, device, config.explainer,
            xgb_model=xgb_model if "xgb" in models_to_explain else None,
            xgb_wrapper=xgb_wrapper,
        )

    # Generate explanations for CNN models (reuse same indices)
    X_explain = dataset.X_test[indices]
    if config.xai_mode == "p":
        if cnn_lstm_flat is not None:
            pa_generate_cnn_explanations(
                results, cnn_lstm_flat, cnn_lstm_wrapper,
                "CNN-LSTM", X_explain, dataset, device, config.explainer,
                checkpoint_dir=checkpoint_dir,
            )
        if cnn_gru_flat is not None:
            pa_generate_cnn_explanations(
                results, cnn_gru_flat, cnn_gru_wrapper,
                "CNN-GRU", X_explain, dataset, device, config.explainer,
                checkpoint_dir=checkpoint_dir,
            )
    else:
        if cnn_lstm_flat is not None:
            _generate_cnn_explanations(
                results, cnn_lstm_flat, cnn_lstm_wrapper,
                "CNN-LSTM", X_explain, dataset, device, config.explainer,
            )
        if cnn_gru_flat is not None:
            _generate_cnn_explanations(
                results, cnn_gru_flat, cnn_gru_wrapper,
                "CNN-GRU", X_explain, dataset, device, config.explainer,
            )

    # Generate and time global summary plots
    X_explain = dataset.X_test[indices]
    generate_and_time_summary_plots(results, X_explain, dataset.feature_names, output_dir)

    # Save explanations
    explain_dir = output_dir / "explanations"
    explain_dir.mkdir(parents=True, exist_ok=True)
    np.save(explain_dir / "explain_indices.npy", indices)
    for exp_result in results:
        fname = f"{exp_result.model_name}_{exp_result.method_name}_attributions.npy"
        np.save(explain_dir / fname, exp_result.attributions)
        logger.info(f"  Saved {fname} ({exp_result.attributions.shape})")

    # Save timing summary
    timing = {
        f"{r.model_name}_{r.method_name}": {
            "time_per_sample_ms": r.time_per_sample_ms,
            "total_time_s": r.total_time_s,
            "summary_plot_time_s": r.summary_plot_time_s,
        }
        for r in results
    }
    with open(explain_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    return results, indices


def phase_evaluate(
    dataset: DatasetBundle,
    config: ExperimentConfig,
    device: torch.device,
    explanation_results: list[ExplanationResult] | None = None,
    explain_indices: np.ndarray | None = None,
    selected_models: list[str] | None = None,
) -> list[dict]:
    """Evaluate all metrics on the generated explanations."""
    models_to_eval = selected_models or ALL_MODELS
    _explain_fn_maker = pa_make_explain_fn if config.xai_mode == "p" else _make_explain_fn
    logger.info(f"=== EVALUATING on {dataset.dataset_name} (models: {models_to_eval}) ===")
    log_gpu_memory("pre-evaluate")
    output_dir = config.output_dir / dataset.dataset_name
    explain_dir = output_dir / "explanations"

    # Load classic models
    dnn_model = rf_model = xgb_model = xgb_label_map = None
    dnn_wrapper = rf_wrapper = xgb_wrapper = None
    if any(m in models_to_eval for m in ["dnn", "rf", "xgb"]):
        dnn_model, rf_model, xgb_model, xgb_label_map = load_models(
            config.models_dir, dataset.dataset_name,
            dataset.X_train.shape[1], dataset.num_classes,
            config.dnn, device,
        )
        if "dnn" in models_to_eval:
            dnn_wrapper = DNNWrapper(dnn_model, device)
        if "rf" in models_to_eval:
            rf_wrapper = RFWrapper(rf_model, num_classes=dataset.num_classes)
        if "xgb" in models_to_eval and xgb_model is not None:
            xgb_wrapper = XGBWrapper(xgb_model, num_classes=dataset.num_classes, label_map=xgb_label_map)

    # Load CNN models
    cnn_lstm_wrapper = cnn_lstm_flat = None
    cnn_gru_wrapper = cnn_gru_flat = None
    if any(m in models_to_eval for m in ["cnn-lstm", "cnn-gru"]):
        cnn_lstm_raw, cnn_lstm_cfg, cnn_gru_raw, cnn_gru_cfg = load_cnn_models(
            config.models_dir, dataset.dataset_name, dataset.num_classes, device,
        )
        if "cnn-lstm" in models_to_eval and cnn_lstm_raw is not None:
            gs = cnn_lstm_cfg.grid_size
            cnn_lstm_wrapper = CNNLSTMWrapper(cnn_lstm_raw, device, gs)
            cnn_lstm_flat = FlatCNNLSTM(cnn_lstm_raw, gs)
        if "cnn-gru" in models_to_eval and cnn_gru_raw is not None:
            gs = cnn_gru_cfg.input_spatial_size
            cnn_gru_wrapper = CNNGRUWrapper(cnn_gru_raw, device, gs)
            cnn_gru_flat = FlatCNNGRU(cnn_gru_raw, gs)

    # Load or use provided explanations
    if explain_indices is None:
        explain_indices = np.load(explain_dir / "explain_indices.npy")

    if explanation_results is None:
        explanation_results = _load_explanations(explain_dir, dataset, config)

    # Filter to only selected models
    explanation_results = [
        r for r in explanation_results
        if _model_name_to_key(r.model_name) in models_to_eval
    ]

    all_metrics = []

    for exp_result in explanation_results:
        logger.info(f"  Evaluating {exp_result.model_name}/{exp_result.method_name}")

        # Select the right predict function and explain function
        if exp_result.model_name == "DNN":
            predict_fn = dnn_wrapper.predict_proba
            explain_fn = _explain_fn_maker(
                exp_result.method_name, "DNN", dnn_model, dnn_wrapper,
                dataset, device, config.explainer,
            )
        elif exp_result.model_name == "XGB":
            predict_fn = xgb_wrapper.predict_proba
            explain_fn = _explain_fn_maker(
                exp_result.method_name, "XGB", None, xgb_wrapper,
                dataset, device, config.explainer, rf_model=xgb_model,
            )
        elif exp_result.model_name == "CNN-LSTM":
            predict_fn = cnn_lstm_wrapper.predict_proba
            explain_fn = _explain_fn_maker(
                exp_result.method_name, "CNN-LSTM", cnn_lstm_flat, cnn_lstm_wrapper,
                dataset, device, config.explainer,
            )
        elif exp_result.model_name == "CNN-GRU":
            predict_fn = cnn_gru_wrapper.predict_proba
            explain_fn = _explain_fn_maker(
                exp_result.method_name, "CNN-GRU", cnn_gru_flat, cnn_gru_wrapper,
                dataset, device, config.explainer,
            )
        else:
            predict_fn = rf_wrapper.predict_proba
            explain_fn = _explain_fn_maker(
                exp_result.method_name, "RF", None, rf_wrapper,
                dataset, device, config.explainer, rf_model=rf_model,
            )

        metrics = evaluate_all_metrics(
            predict_fn=predict_fn,
            explain_fn=explain_fn,
            explanation_result=exp_result,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            explain_indices=explain_indices,
            num_classes=dataset.num_classes,
            config=config.metric,
        )
        metrics["dataset"] = dataset.dataset_name
        all_metrics.append(metrics)

        logger.info(
            f"    Faithfulness corr: {metrics.get('faithfulness_correlation', 'N/A'):.4f}, "
            f"Sparsity: {metrics.get('sparsity_mean', 'N/A'):.4f}, "
            f"Complexity: {metrics.get('complexity_normalized', 'N/A'):.4f}, "
            f"Stability: {metrics.get('stability_jaccard_mean', 'N/A'):.4f}"
        )

    # Save all metrics
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=_json_serialize)
    logger.info(f"Evaluation metrics saved to {metrics_path}")

    # Save summary table
    _save_summary_table(all_metrics, output_dir / "summary.csv")

    return all_metrics


def _model_name_to_key(model_name: str) -> str:
    """Map model display names (e.g. 'CNN-LSTM') to CLI keys (e.g. 'cnn-lstm')."""
    return model_name.lower().replace("_", "-")


def _generate_cnn_explanations(
    results: list[ExplanationResult],
    flat_model: torch.nn.Module,
    wrapper,
    model_name: str,
    X_explain: np.ndarray,
    dataset: DatasetBundle,
    device: torch.device,
    config,
) -> None:
    """Generate SHAP, LIME, IG, and DeepLIFT explanations for a CNN model.

    The flat_model (FlatCNNLSTM or FlatCNNGRU) accepts flat input, making it
    compatible with the standard explainer functions.
    """
    rng = np.random.RandomState(42)
    bg_indices = rng.choice(len(dataset.X_train), size=config.shap_background_samples, replace=False)
    X_background = dataset.X_train[bg_indices]

    # SHAP (DeepExplainer)
    try:
        r = explain_shap_dnn(flat_model, X_explain, X_background, device, config)
        r.model_name = model_name
        results.append(r)
    except Exception as e:
        logger.warning(f"  SHAP failed for {model_name}: {e}")

    # LIME
    try:
        r = explain_lime(
            wrapper.predict_proba, X_explain, dataset.X_train,
            dataset.feature_names, dataset.num_classes, model_name, config,
        )
        r.model_name = model_name
        results.append(r)
    except Exception as e:
        logger.warning(f"  LIME failed for {model_name}: {e}")

    # Integrated Gradients
    try:
        r = explain_ig(flat_model, X_explain, device, config)
        r.model_name = model_name
        results.append(r)
    except Exception as e:
        logger.warning(f"  IG failed for {model_name}: {e}")

    # DeepLIFT
    try:
        r = explain_deeplift(flat_model, X_explain, device, config)
        r.model_name = model_name
        results.append(r)
    except Exception as e:
        logger.warning(f"  DeepLIFT failed for {model_name}: {e}")


def _make_explain_fn(
    method_name: str,
    model_type: str,
    dnn_model,
    wrapper,
    dataset: DatasetBundle,
    device: torch.device,
    config,
    rf_model=None,
):
    """Create a callable explain function for metric evaluation."""
    rng = np.random.RandomState(42)
    bg_indices = rng.choice(len(dataset.X_train), size=config.shap_background_samples, replace=False)
    X_background = dataset.X_train[bg_indices]

    if method_name == "SHAP" and model_type in ("DNN", "CNN-LSTM", "CNN-GRU"):
        def fn(X):
            r = explain_shap_dnn(dnn_model, X, X_background, device, config)
            return r.attributions
    elif method_name == "SHAP" and model_type == "RF":
        def fn(X):
            r = explain_shap_rf(rf_model, X, config)
            return r.attributions
    elif method_name == "SHAP" and model_type == "XGB":
        def fn(X):
            r = explain_shap_rf(rf_model, X, config)  # TreeExplainer works for XGBoost
            return r.attributions
    elif method_name == "LIME":
        def fn(X):
            r = explain_lime(
                wrapper.predict_proba, X, dataset.X_train,
                dataset.feature_names, dataset.num_classes, model_type, config,
            )
            return r.attributions
    elif method_name == "IG":
        def fn(X):
            r = explain_ig(dnn_model, X, device, config)
            return r.attributions
    elif method_name == "DeepLIFT":
        def fn(X):
            r = explain_deeplift(dnn_model, X, device, config)
            return r.attributions
    else:
        raise ValueError(f"Unknown method: {method_name}")

    return fn


def _load_explanations(
    explain_dir: Path, dataset: DatasetBundle, config: ExperimentConfig
) -> list[ExplanationResult]:
    """Load saved explanation attributions from disk."""
    results = []
    for path in sorted(explain_dir.glob("*_attributions.npy")):
        parts = path.stem.replace("_attributions", "").split("_", 1)
        model_name, method_name = parts[0], parts[1]
        attrs = np.load(path)

        # Load timing info if available
        timing_path = explain_dir / "timing.json"
        time_per_sample = 0.0
        total_time = 0.0
        if timing_path.exists():
            with open(timing_path) as f:
                timing = json.load(f)
            key = f"{model_name}_{method_name}"
            if key in timing:
                time_per_sample = timing[key]["time_per_sample_ms"]
                total_time = timing[key]["total_time_s"]

        results.append(ExplanationResult(
            attributions=attrs,
            method_name=method_name,
            model_name=model_name,
            time_per_sample_ms=time_per_sample,
            total_time_s=total_time,
        ))
    return results


def _save_summary_table(all_metrics: list[dict], path: Path) -> None:
    """Save a CSV summary of key metrics across all methods."""
    import csv

    key_cols = [
        "dataset", "model", "method",
        "faithfulness_correlation", "sparsity_mean",
        "complexity_normalized", "time_per_sample_ms",
        "stability_jaccard_mean", "completeness_success_rate",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=key_cols, extrasaction="ignore")
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: f"{m.get(k, 'N/A')}" for k in key_cols})

    logger.info(f"Summary table saved to {path}")


def _json_serialize(obj):
    """JSON serializer for numpy types. Converts NaN to None for valid JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def run_experiment(
    config: ExperimentConfig, datasets: list[str], phases: list[str],
    selected_models: list[str] | None = None,
):
    """Main experiment runner."""
    device, num_gpus = setup_device()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(config.output_dir / "config.json", "w") as f:
        json.dump({
            "xai_mode": config.xai_mode,
            "datasets": datasets,
            "phases": phases,
            "models": selected_models or ALL_MODELS,
            "dnn": vars(config.dnn),
            "rf": vars(config.rf),
            "xgb": vars(config.xgb),
            "cnn_lstm": vars(config.cnn_lstm),
            "cnn_gru": vars(config.cnn_gru),
            "explainer": vars(config.explainer),
            "metric": vars(config.metric),
            "seed": config.seed,
        }, f, indent=2, default=_json_serialize)

    all_results = {}
    for ds_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"DATASET: {ds_name}")
        logger.info(f"{'='*60}")

        ds_start = time.time()

        try:
            dataset = load_dataset(ds_name, config.data)
        except Exception as e:
            logger.error(f"Failed to load {ds_name}: {e}")
            continue

        ds_results = {}

        # Phase: Train
        if "train" in phases or "all" in phases:
            try:
                train_results = phase_train(dataset, config, device, num_gpus, selected_models)
                ds_results["train"] = train_results
            except Exception as e:
                logger.error(f"Training failed for {ds_name}: {e}", exc_info=True)
                continue
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Phase: Explain
        explanation_results = None
        explain_indices = None
        if "explain" in phases or "all" in phases:
            try:
                explanation_results, explain_indices = phase_explain(dataset, config, device, selected_models)
                ds_results["explain"] = {
                    r.method_name + "_" + r.model_name: {
                        "time_per_sample_ms": r.time_per_sample_ms,
                        "total_time_s": r.total_time_s,
                        "summary_plot_time_s": r.summary_plot_time_s,
                    }
                    for r in explanation_results
                }
            except Exception as e:
                logger.error(f"Explanation failed for {ds_name}: {e}", exc_info=True)
                continue
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Phase: Evaluate
        if "evaluate" in phases or "all" in phases:
            try:
                eval_results = phase_evaluate(
                    dataset, config, device, explanation_results, explain_indices,
                    selected_models,
                )
                ds_results["evaluate"] = eval_results
            except Exception as e:
                logger.error(f"Evaluation failed for {ds_name}: {e}", exc_info=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        ds_elapsed = time.time() - ds_start
        logger.info(f"Dataset {ds_name} completed in {ds_elapsed:.1f}s")
        all_results[ds_name] = ds_results

    # Save combined results
    combined_path = config.output_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_serialize)
    logger.info(f"\nAll results saved to {combined_path}")

    # Generate summary plots
    try:
        generate_summary_plots(all_results, config.output_dir)
    except Exception as e:
        logger.error(f"Plot generation failed: {e}", exc_info=True)

    _print_final_summary(all_results)


def _print_final_summary(all_results: dict):
    """Print a human-readable summary of the experiment."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 1 SUMMARY: Quantitative Benchmarking of Explanation Quality")
    logger.info("=" * 80)

    for ds_name, ds_results in all_results.items():
        logger.info(f"\n--- {ds_name} ---")

        if "train" in ds_results:
            for model_name, model_metrics in ds_results["train"].items():
                acc = model_metrics.get("accuracy", "N/A")
                f1 = model_metrics.get("f1_weighted", "N/A")
                auc = model_metrics.get("auc_roc", "N/A")
                logger.info(
                    f"  {model_name.upper()}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
                    if isinstance(acc, float)
                    else f"  {model_name.upper()}: {model_metrics}"
                )

        if "evaluate" in ds_results:
            for m in ds_results["evaluate"]:
                logger.info(
                    f"  {m['model']}/{m['method']}: "
                    f"Faith={m.get('faithfulness_correlation', 0):.3f} "
                    f"Spars={m.get('sparsity_mean', 0):.3f} "
                    f"Cmplx={m.get('complexity_normalized', 0):.3f} "
                    f"Effic={m.get('time_per_sample_ms', 0):.1f}ms "
                    f"Stab={m.get('stability_jaccard_mean', 0):.3f} "
                    f"Comp={m.get('completeness_success_rate', 0):.3f}"
                )


def generate_summary_plots(all_results: dict, output_dir: Path) -> None:
    """Generate global summary bar plots comparing XAI methods across metrics."""
    import time as _time

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    plot_start = _time.time()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Collect all evaluation metrics into a DataFrame
    rows = []
    for ds_name, ds_results in all_results.items():
        if "evaluate" not in ds_results:
            continue
        for m in ds_results["evaluate"]:
            rows.append(m)

    if not rows:
        logger.warning("No evaluation results to plot.")
        return

    df = pd.DataFrame(rows)
    df["label"] = df["model"] + "/" + df["method"]

    # Key metrics to plot
    plot_metrics = [
        ("faithfulness_correlation", "Faithfulness Correlation"),
        ("sparsity_mean", "Sparsity"),
        ("complexity_normalized", "Complexity (Normalized Entropy)"),
        ("time_per_sample_ms", "Efficiency (ms/sample)"),
        ("stability_jaccard_mean", "Stability (Jaccard)"),
        ("completeness_success_rate", "Completeness"),
    ]

    for metric_key, metric_title in plot_metrics:
        if metric_key not in df.columns:
            continue

        for ds_name in df["dataset"].unique():
            ds_df = df[df["dataset"] == ds_name].copy()
            ds_df = ds_df.dropna(subset=[metric_key])
            if ds_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(ds_df["label"], ds_df[metric_key].astype(float))
            ax.set_title(f"{metric_title} — {ds_name}")
            ax.set_ylabel(metric_title)
            ax.set_xlabel("Model / Method")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            fname = f"{ds_name}_{metric_key}.png"
            fig.savefig(plot_dir / fname, dpi=150)
            plt.close(fig)

    # Combined heatmap across datasets (if multiple)
    if len(df["dataset"].unique()) > 1:
        for metric_key, metric_title in plot_metrics:
            if metric_key not in df.columns:
                continue
            pivot = df.pivot_table(index="label", columns="dataset", values=metric_key)
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            import seaborn as sns
            sns.heatmap(pivot.astype(float), annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
            ax.set_title(f"{metric_title} across datasets")
            plt.tight_layout()
            fig.savefig(plot_dir / f"heatmap_{metric_key}.png", dpi=150)
            plt.close(fig)

    plot_elapsed = _time.time() - plot_start
    logger.info(f"Summary plots saved to {plot_dir} ({plot_elapsed:.1f}s)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 1: Quantitative Benchmarking of Explanation Quality"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=["nsl-kdd", "cic-ids-2017", "unsw-nb15", "cse-cic-ids2018"],
        help="Datasets to process (default: all)",
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        default=["all"],
        choices=["all", "train", "explain", "evaluate"],
        help="Experiment phase(s) to run",
    )
    parser.add_argument(
        "--num-explain-samples",
        type=int,
        default=None,
        help="Number of test samples to generate explanations for (default: from YAML)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=ALL_MODELS,
        help=f"Models to run (default: all). Choices: {', '.join(ALL_MODELS)}",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from YAML)",
    )
    parser.add_argument(
        "-x", "--xai-mode",
        choices=["n", "p", "b"],
        default="n",
        help="XAI mode: n=normal, p=protocol-aware, b=both (runs n then p)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which XAI modes to run
    modes = ["n", "p"] if args.xai_mode == "b" else [args.xai_mode]

    datasets = args.datasets
    phases = args.phase
    selected_models = args.models

    for mode in modes:
        config = load_config()
        config.xai_mode = mode
        mode_dir = "normal" if mode == "n" else "protocol-aware"
        config.output_dir = Path("experiments/1/results") / mode_dir
        config.models_dir = Path("experiments/1/results")

        # CLI overrides (only when explicitly passed)
        if args.output_dir:
            config.output_dir = Path(args.output_dir) / mode_dir
        if args.no_smote:
            config.data.apply_smote = False
        if args.seed is not None:
            config.seed = args.seed
        if args.num_explain_samples is not None:
            config.explainer.num_explain_samples = args.num_explain_samples

        # Set seeds
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        ds_list = datasets or config.ALL_DATASETS
        mode_label = "protocol-aware" if mode == "p" else "normal"

        logger.info("Experiment 1: Quantitative Benchmarking of Explanation Quality")
        logger.info(f"XAI mode: {mode_label}")
        logger.info(f"Datasets: {ds_list}")
        logger.info(f"Phases: {phases}")
        logger.info(f"Models: {selected_models or ALL_MODELS}")
        logger.info(f"Output: {config.output_dir}")
        logger.info(f"Models dir: {config.models_dir}")
        logger.info(f"Explain samples: {config.explainer.num_explain_samples}")
        logger.info(f"CPU fraction: {config.cpu_fraction}")

        run_experiment(config, ds_list, phases, selected_models)


if __name__ == "__main__":
    main()
