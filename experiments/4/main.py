#!/usr/bin/env python3
"""Experiment 4: XAI-Driven Dimensionality Reduction vs Statistical Baselines.

Full pipeline: data loading -> feature selection (statistical + XAI) ->
downstream model training -> inference benchmarking -> metric evaluation.

Usage:
    # Full experiment on all datasets (both normal + PA-XAI)
    python experiments/4/main.py

    # Specific dataset(s)
    python experiments/4/main.py --datasets nsl-kdd cic-ids-2017

    # Specific phase
    python experiments/4/main.py --phase select       # Feature selection only
    python experiments/4/main.py --phase benchmark    # Downstream training + eval only
    python experiments/4/main.py --phase all          # Full pipeline

    # XAI mode selection
    python experiments/4/main.py --xai-mode n         # Normal XAI only
    python experiments/4/main.py --xai-mode p         # Protocol-Aware XAI only
    python experiments/4/main.py --xai-mode both      # Both modes (default from YAML)

    # Skip specific pipelines
    python experiments/4/main.py --skip-statistical
    python experiments/4/main.py --skip-xai
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

# Add experiment directories to path for local imports
# Exp1 must be added first (lower priority), then exp4 (higher priority)
_exp4_dir = os.path.dirname(os.path.abspath(__file__))
_commons_dir = os.path.join(_exp4_dir, "..", "commons")
exp1_dir = os.path.join(_exp4_dir, "..", "1")
sys.path.insert(0, _commons_dir)
sys.path.insert(0, exp1_dir)
sys.path.insert(0, _exp4_dir)

from config import ExperimentConfig, load_experiment_config
from data_loader import DatasetBundle, load_dataset as load_dataset_exp1
from evaluation import compute_reduction_summary, evaluate_downstream_model
from feature_selection import FeatureSelectionResult, run_statistical_pipeline
from models import (
    NIDSNet,
    NNWrapper,
    SKLearnWrapper,
    train_all_downstream,
)
from visualization import generate_all_plots, generate_summary_csv
from xai_selection import run_xai_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("experiment4")


def setup_device() -> tuple[torch.device, int]:
    """Detect available GPUs."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device("cuda")
        logger.info(f"CUDA available: {num_gpus} GPU(s)")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"  GPU {i}: {props.name} | {props.total_memory / 1e9:.1f} GB | "
                f"Compute {props.major}.{props.minor}"
            )
    else:
        device = torch.device("cpu")
        num_gpus = 0
        logger.warning("No CUDA GPUs. Running on CPU.")
    return device, num_gpus


def log_gpu_memory(label: str = "") -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        logger.info(f"  GPU {i} [{label}]: {alloc:.2f} GB alloc, {reserved:.2f} GB reserved")


def _json_serialize(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Load or train baseline models for XAI pipeline
# ---------------------------------------------------------------------------

def load_or_train_baseline_models(
    dataset: DatasetBundle,
    config: ExperimentConfig,
    device: torch.device,
    num_gpus: int,
) -> tuple:
    """Load Exp1 models if available, otherwise train fresh baseline DNN + RF."""
    input_dim = dataset.X_train.shape[1]
    exp1_model_dir = config.exp1_output_dir / "models" / dataset.dataset_name

    dnn_path = exp1_model_dir / "dnn.pt"
    rf_path = exp1_model_dir / "rf.joblib"

    if dnn_path.exists() and rf_path.exists():
        logger.info(f"  Loading Experiment 1 models from {exp1_model_dir}")
        dnn = NIDSNet(
            input_dim=input_dim,
            num_classes=dataset.num_classes,
            hidden_layers=config.dnn.hidden_layers,
            dropout_rate=config.dnn.dropout_rate,
        )
        dnn.load_state_dict(torch.load(dnn_path, map_location=device, weights_only=True))
        dnn = dnn.to(device)
        dnn.eval()

        if device.type == "cuda":
            with torch.no_grad():
                dnn(torch.zeros(1, input_dim, device=device))

        rf = joblib.load(rf_path)
        dnn_wrapper = NNWrapper(dnn, device)
        rf_wrapper = SKLearnWrapper(rf, dataset.num_classes)
        logger.info("  Loaded pre-trained baseline models from Experiment 1")
    else:
        logger.info("  Experiment 1 models not found. Training fresh baselines...")
        from models import train_dnn as _train_dnn, train_rf as _train_rf

        dnn, dnn_wrapper, _ = _train_dnn(
            dataset.X_train, dataset.y_train, dataset.X_val, dataset.y_val,
            dataset.num_classes, config.dnn, device, num_gpus,
        )
        rf_model, rf_wrapper, _ = _train_rf(
            dataset.X_train, dataset.y_train, dataset.num_classes, config.rf,
        )
        rf = rf_model
        # Save for reuse
        save_dir = config.output_dir / "baseline_models" / dataset.dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        base = dnn.module if isinstance(dnn, torch.nn.DataParallel) else dnn
        torch.save(base.state_dict(), save_dir / "dnn.pt")
        joblib.dump(rf, save_dir / "rf.joblib")

    return dnn, rf, dnn_wrapper, rf_wrapper


# ---------------------------------------------------------------------------
# Feature selection phase
# ---------------------------------------------------------------------------

def phase_select(
    dataset: DatasetBundle,
    config: ExperimentConfig,
    device: torch.device,
    num_gpus: int,
    skip_statistical: bool = False,
    skip_xai: bool = False,
) -> list[FeatureSelectionResult]:
    """Run both feature selection pipelines."""
    logger.info(f"=== FEATURE SELECTION on {dataset.dataset_name} ===")
    log_gpu_memory("pre-select")

    output_dir = config.output_dir / dataset.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine target feature count for this dataset
    n_features = dataset.X_train.shape[1]
    target = config.xai.target_features.get(dataset.dataset_name, max(15, n_features // 4))
    logger.info(f"  Original features: {n_features}, target: ~{target}")

    all_selections: list[FeatureSelectionResult] = []

    # --- Statistical pipeline ---
    if not skip_statistical:
        stat_results = run_statistical_pipeline(
            dataset.X_train, dataset.y_train,
            dataset.feature_names, target, config.statistical,
        )
        all_selections.extend(stat_results)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- XAI pipeline ---
    if not skip_xai:
        dnn, rf, dnn_wrapper, rf_wrapper = load_or_train_baseline_models(
            dataset, config, device, num_gpus,
        )

        xai_results = run_xai_pipeline(
            dnn, rf, dnn_wrapper, rf_wrapper,
            dataset, device, config,
        )
        all_selections.extend(xai_results)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save selection results
    sel_dir = output_dir / "selections"
    sel_dir.mkdir(parents=True, exist_ok=True)
    for sel in all_selections:
        np.save(sel_dir / f"{sel.method_name}_indices.npy", sel.selected_indices)
        np.save(sel_dir / f"{sel.method_name}_rankings.npy", sel.feature_rankings)

    sel_summary = [{
        "method_name": s.method_name,
        "n_original": s.n_original,
        "n_selected": s.n_selected,
        "selection_time_s": s.selection_time_s,
        "selected_features": s.selected_feature_names,
    } for s in all_selections]

    with open(sel_dir / "selection_summary.json", "w") as f:
        json.dump(sel_summary, f, indent=2, default=_json_serialize)

    logger.info(f"  Feature selection results saved to {sel_dir}")
    return all_selections


# ---------------------------------------------------------------------------
# Benchmark phase
# ---------------------------------------------------------------------------

def phase_benchmark(
    dataset: DatasetBundle,
    selections: list[FeatureSelectionResult],
    config: ExperimentConfig,
    device: torch.device,
    num_gpus: int,
) -> list[dict]:
    """Train and evaluate all downstream models on each feature subset."""
    logger.info(f"=== BENCHMARKING on {dataset.dataset_name} ===")
    log_gpu_memory("pre-benchmark")

    output_dir = config.output_dir / dataset.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_benchmarks = []

    # --- Baseline: full feature set ---
    logger.info("  --- Full feature baseline ---")
    full_models = train_all_downstream(
        dataset.X_train, dataset.y_train,
        dataset.X_val, dataset.y_val,
        dataset.num_classes,
        config.dnn, config.cnn, config.rf, config.svm,
        device, num_gpus,
    )

    full_feature_metrics = {}
    for model_name, (wrapper, train_time) in full_models.items():
        metrics = evaluate_downstream_model(
            wrapper, dataset.X_test, dataset.y_test,
            dataset.num_classes, model_name, "Full",
            dataset.X_train.shape[1], train_time,
        )
        all_benchmarks.append(metrics)
        full_feature_metrics[model_name] = metrics

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Reduced feature subsets ---
    for sel in selections:
        logger.info(f"  --- {sel.method_name} ({sel.n_selected} features) ---")
        idx = sel.selected_indices

        X_train_reduced = dataset.X_train[:, idx]
        X_val_reduced = dataset.X_val[:, idx]
        X_test_reduced = dataset.X_test[:, idx]

        reduced_models = train_all_downstream(
            X_train_reduced, dataset.y_train,
            X_val_reduced, dataset.y_val,
            dataset.num_classes,
            config.dnn, config.cnn, config.rf, config.svm,
            device, num_gpus,
        )

        for model_name, (wrapper, train_time) in reduced_models.items():
            metrics = evaluate_downstream_model(
                wrapper, X_test_reduced, dataset.y_test,
                dataset.num_classes, model_name, sel.method_name,
                sel.n_selected, train_time,
            )

            # Add reduction summary
            if model_name in full_feature_metrics:
                reduction = compute_reduction_summary(
                    full_feature_metrics[model_name],
                    metrics,
                    dataset.X_train.shape[1],
                )
                metrics["reduction_summary"] = reduction

            all_benchmarks.append(metrics)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save benchmarks
    bench_path = output_dir / "benchmark_results.json"
    with open(bench_path, "w") as f:
        json.dump(all_benchmarks, f, indent=2, default=_json_serialize)
    logger.info(f"  Benchmark results saved to {bench_path}")

    return all_benchmarks


# ---------------------------------------------------------------------------
# Load saved selections
# ---------------------------------------------------------------------------

def load_selections(dataset_name: str, config: ExperimentConfig) -> list[FeatureSelectionResult]:
    """Load previously saved feature selection results."""
    sel_dir = config.output_dir / dataset_name / "selections"
    if not sel_dir.exists():
        raise FileNotFoundError(f"No saved selections at {sel_dir}")

    summary_path = sel_dir / "selection_summary.json"
    with open(summary_path) as f:
        summaries = json.load(f)

    results = []
    for s in summaries:
        method = s["method_name"]
        indices = np.load(sel_dir / f"{method}_indices.npy")
        rankings = np.load(sel_dir / f"{method}_rankings.npy")
        results.append(FeatureSelectionResult(
            method_name=method,
            selected_indices=indices,
            feature_rankings=rankings,
            selected_feature_names=s["selected_features"],
            n_original=s["n_original"],
            n_selected=s["n_selected"],
            selection_time_s=s["selection_time_s"],
        ))

    logger.info(f"  Loaded {len(results)} saved feature selections for {dataset_name}")
    return results


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    config: ExperimentConfig,
    datasets: list[str],
    phases: list[str],
    skip_statistical: bool = False,
    skip_xai: bool = False,
):
    """Run the full Experiment 4 pipeline."""
    device, num_gpus = setup_device()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(config.output_dir / "config.json", "w") as f:
        json.dump({
            "datasets": datasets,
            "phases": phases,
            "xai_modes": config.xai_modes,
            "dnn": vars(config.dnn),
            "cnn": vars(config.cnn),
            "rf": vars(config.rf),
            "svm": vars(config.svm),
            "explainer": vars(config.explainer),
            "statistical": vars(config.statistical),
            "xai": vars(config.xai),
            "seed": config.seed,
        }, f, indent=2, default=_json_serialize)

    all_results = {}

    for ds_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"DATASET: {ds_name}")
        logger.info(f"{'='*60}")

        ds_start = time.time()

        try:
            dataset = load_dataset_exp1(ds_name, config.data)
        except Exception as e:
            logger.error(f"Failed to load {ds_name}: {e}", exc_info=True)
            continue

        ds_results = {}

        # Phase: Feature Selection
        selections = []
        if "select" in phases or "all" in phases:
            try:
                selections = phase_select(
                    dataset, config, device, num_gpus,
                    skip_statistical, skip_xai,
                )
                ds_results["feature_selections"] = [{
                    "method_name": s.method_name,
                    "n_original": s.n_original,
                    "n_selected": s.n_selected,
                    "selection_time_s": s.selection_time_s,
                } for s in selections]
            except Exception as e:
                logger.error(f"Feature selection failed for {ds_name}: {e}", exc_info=True)
                continue
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Phase: Benchmark
        if "benchmark" in phases or "all" in phases:
            # Load selections if not already available
            if not selections:
                try:
                    selections = load_selections(ds_name, config)
                except FileNotFoundError:
                    logger.error(f"No feature selections found for {ds_name}. Run 'select' phase first.")
                    continue

            try:
                benchmarks = phase_benchmark(
                    dataset, selections, config, device, num_gpus,
                )
                ds_results["benchmarks"] = benchmarks
            except Exception as e:
                logger.error(f"Benchmarking failed for {ds_name}: {e}", exc_info=True)
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

    # Generate plots and summary
    try:
        generate_all_plots(all_results, config.output_dir)
        generate_summary_csv(all_results, config.output_dir)
    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)

    _print_final_summary(all_results)


def _print_final_summary(all_results: dict):
    """Print human-readable summary."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 4 SUMMARY: XAI-Driven Dimensionality Reduction vs Statistical Baselines")
    logger.info("=" * 80)

    for ds_name, ds_data in all_results.items():
        logger.info(f"\n--- {ds_name} ---")

        if "feature_selections" in ds_data:
            logger.info("  Feature Selections:")
            for sel in ds_data["feature_selections"]:
                logger.info(
                    f"    {sel['method_name']}: {sel['n_original']} -> {sel['n_selected']} "
                    f"features ({sel['selection_time_s']:.1f}s)"
                )

        if "benchmarks" in ds_data:
            logger.info("  Benchmark Results:")

            # Group by selection method
            by_method: dict[str, list] = {}
            for b in ds_data["benchmarks"]:
                method = b["selection_method"]
                by_method.setdefault(method, []).append(b)

            for method, entries in by_method.items():
                logger.info(f"    [{method}]")
                for e in entries:
                    line = (
                        f"      {e['model']:4s}: "
                        f"Acc={e['accuracy']:.4f} "
                        f"F1w={e['f1_weighted']:.4f} "
                        f"FPR={e['fpr_macro']:.4f} "
                        f"Lat={e['inference_ms_per_sample_mean']:.4f}ms"
                    )
                    if "reduction_summary" in e:
                        rs = e["reduction_summary"]
                        line += (
                            f" | Δacc={rs['accuracy_change']:+.4f} "
                            f"ΔF1={rs['f1_weighted_change']:+.4f} "
                            f"speedup={rs['inference_latency_speedup']:.2f}x"
                        )
                    logger.info(line)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 4: XAI-Driven Dimensionality Reduction vs Statistical Baselines"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        choices=["nsl-kdd", "cic-ids-2017", "unsw-nb15", "cse-cic-ids2018"],
        help="Datasets to process (default: all)",
    )
    parser.add_argument(
        "--phase", nargs="+", default=["all"],
        choices=["all", "select", "benchmark"],
        help="Experiment phase(s) to run",
    )
    parser.add_argument(
        "--xai-mode", type=str, default=None,
        choices=["n", "p", "both"],
        help="XAI mode: n (normal), p (protocol-aware), both (default from config.yaml)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-statistical", action="store_true",
        help="Skip statistical feature selection pipeline",
    )
    parser.add_argument(
        "--skip-xai", action="store_true",
        help="Skip XAI-driven feature selection pipeline",
    )
    parser.add_argument(
        "--no-smote", action="store_true",
        help="Disable SMOTE oversampling",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default from config.yaml)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: experiments/commons/config.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_experiment_config(args.config)

    # CLI overrides
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.no_smote:
        config.data.apply_smote = False
    if args.seed is not None:
        config.seed = args.seed
    if args.xai_mode:
        if args.xai_mode == "both":
            config.xai_modes = ["normal", "pa"]
        elif args.xai_mode == "n":
            config.xai_modes = ["normal"]
        elif args.xai_mode == "p":
            config.xai_modes = ["pa"]

    datasets = args.datasets or config.ALL_DATASETS
    phases = args.phase

    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger.info("Experiment 4: XAI-Driven Dimensionality Reduction vs Statistical Baselines")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Phases: {phases}")
    logger.info(f"XAI modes: {config.xai_modes}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Skip statistical: {args.skip_statistical}")
    logger.info(f"Skip XAI: {args.skip_xai}")

    run_experiment(config, datasets, phases, args.skip_statistical, args.skip_xai)


if __name__ == "__main__":
    main()
