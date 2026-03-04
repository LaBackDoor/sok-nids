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
from functools import partial
from pathlib import Path

import numpy as np
import torch

# Add experiment directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig
from data_loader import DatasetBundle, load_dataset
from explainers import (
    ExplanationResult,
    explain_deeplift,
    explain_ig,
    explain_lime,
    explain_shap_dnn,
    explain_shap_rf,
    generate_all_explanations,
)
from metrics import evaluate_all_metrics
from models import (
    DNNWrapper,
    NIDSNet,
    RFWrapper,
    SoftmaxModel,
    XGBWrapper,
    load_models,
    save_models,
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


def phase_train(
    dataset: DatasetBundle, config: ExperimentConfig, device: torch.device, num_gpus: int
) -> dict:
    """Train DNN and RF models on the dataset."""
    logger.info(f"=== TRAINING on {dataset.dataset_name} ===")
    log_gpu_memory("pre-train")
    output_dir = config.output_dir / dataset.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train DNN
    dnn_model, dnn_wrapper, dnn_metrics = train_dnn(dataset, config.dnn, device, num_gpus)
    log_gpu_memory("post-DNN-train")

    # Train RF
    rf_model, rf_wrapper, rf_metrics = train_rf(dataset, config.rf)

    # Train XGBoost
    xgb_model, xgb_wrapper, xgb_metrics = train_xgb(dataset, config.xgb)

    # Save models
    save_models(
        dnn_model, rf_model, config.output_dir, dataset.dataset_name,
        xgb_model=xgb_model, xgb_label_map=xgb_wrapper.label_map,
    )

    # Save training metrics
    train_results = {"dnn": dnn_metrics, "rf": rf_metrics, "xgb": xgb_metrics}
    results_path = output_dir / "train_metrics.json"
    with open(results_path, "w") as f:
        json.dump(train_results, f, indent=2, default=_json_serialize)
    logger.info(f"Training metrics saved to {results_path}")

    return train_results


def phase_explain(
    dataset: DatasetBundle, config: ExperimentConfig, device: torch.device
) -> tuple[list[ExplanationResult], np.ndarray]:
    """Generate XAI explanations using all methods."""
    logger.info(f"=== EXPLAINING on {dataset.dataset_name} ===")
    log_gpu_memory("pre-explain")
    output_dir = config.output_dir / dataset.dataset_name

    # Load trained models
    dnn_model, rf_model, xgb_model, xgb_label_map = load_models(
        config.output_dir, dataset.dataset_name,
        dataset.X_train.shape[1], dataset.num_classes,
        config.dnn, device,
    )
    dnn_wrapper = DNNWrapper(dnn_model, device)
    rf_wrapper = RFWrapper(rf_model, num_classes=dataset.num_classes)
    xgb_wrapper = XGBWrapper(xgb_model, num_classes=dataset.num_classes, label_map=xgb_label_map) if xgb_model else None

    # Generate explanations
    results, indices = generate_all_explanations(
        dnn_model, rf_model, dnn_wrapper, rf_wrapper,
        dataset, device, config.explainer,
        xgb_model=xgb_model, xgb_wrapper=xgb_wrapper,
    )

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
) -> list[dict]:
    """Evaluate all metrics on the generated explanations."""
    logger.info(f"=== EVALUATING on {dataset.dataset_name} ===")
    log_gpu_memory("pre-evaluate")
    output_dir = config.output_dir / dataset.dataset_name
    explain_dir = output_dir / "explanations"

    # Load models
    dnn_model, rf_model, xgb_model, xgb_label_map = load_models(
        config.output_dir, dataset.dataset_name,
        dataset.X_train.shape[1], dataset.num_classes,
        config.dnn, device,
    )
    dnn_wrapper = DNNWrapper(dnn_model, device)
    rf_wrapper = RFWrapper(rf_model, num_classes=dataset.num_classes)
    xgb_wrapper = XGBWrapper(xgb_model, num_classes=dataset.num_classes, label_map=xgb_label_map) if xgb_model else None

    # Load or use provided explanations
    if explain_indices is None:
        explain_indices = np.load(explain_dir / "explain_indices.npy")

    if explanation_results is None:
        explanation_results = _load_explanations(explain_dir, dataset, config)

    all_metrics = []

    for exp_result in explanation_results:
        logger.info(f"  Evaluating {exp_result.model_name}/{exp_result.method_name}")

        # Select the right predict function and explain function
        if exp_result.model_name == "DNN":
            predict_fn = dnn_wrapper.predict_proba
            explain_fn = _make_explain_fn(
                exp_result.method_name, "DNN", dnn_model, dnn_wrapper,
                dataset, device, config.explainer,
            )
        elif exp_result.model_name == "XGB":
            predict_fn = xgb_wrapper.predict_proba
            explain_fn = _make_explain_fn(
                exp_result.method_name, "XGB", None, xgb_wrapper,
                dataset, device, config.explainer, rf_model=xgb_model,
            )
        else:
            predict_fn = rf_wrapper.predict_proba
            explain_fn = _make_explain_fn(
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

    if method_name == "SHAP" and model_type == "DNN":
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
        "robustness_mean_deviation", "robustness_rank_correlation_mean",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=key_cols, extrasaction="ignore")
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: f"{m.get(k, 'N/A')}" for k in key_cols})

    logger.info(f"Summary table saved to {path}")


def _json_serialize(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def run_experiment(config: ExperimentConfig, datasets: list[str], phases: list[str]):
    """Main experiment runner."""
    device, num_gpus = setup_device()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(config.output_dir / "config.json", "w") as f:
        json.dump({
            "datasets": datasets,
            "phases": phases,
            "dnn": vars(config.dnn),
            "rf": vars(config.rf),
            "xgb": vars(config.xgb),
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
                train_results = phase_train(dataset, config, device, num_gpus)
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
                explanation_results, explain_indices = phase_explain(dataset, config, device)
                ds_results["explain"] = {
                    r.method_name + "_" + r.model_name: {
                        "time_per_sample_ms": r.time_per_sample_ms,
                        "total_time_s": r.total_time_s,
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
                    dataset, config, device, explanation_results, explain_indices
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
                    f"Comp={m.get('completeness_success_rate', 0):.3f} "
                    f"Robust={m.get('robustness_rank_correlation_mean', 0):.3f}"
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
        ("robustness_mean_deviation", "Robustness (Mean Deviation)"),
        ("robustness_rank_correlation_mean", "Robustness (Rank Correlation)"),
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
        choices=["nsl-kdd", "cic-ids-2017", "unsw-nb15", "cse-cic-ids2018", "cic-iov-2024"],
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
        default=10000,
        help="Number of test samples to generate explanations for",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig()

    # Apply CLI overrides
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.no_smote:
        config.data.apply_smote = False
    config.seed = args.seed
    config.explainer.num_explain_samples = args.num_explain_samples

    datasets = args.datasets or config.ALL_DATASETS
    phases = args.phase

    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger.info("Experiment 1: Quantitative Benchmarking of Explanation Quality")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Phases: {phases}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Explain samples: {config.explainer.num_explain_samples}")

    run_experiment(config, datasets, phases)


if __name__ == "__main__":
    main()
