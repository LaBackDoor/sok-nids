#!/usr/bin/env python3
"""Experiment 3: Feature Interaction, Consensus, and Human-in-the-Loop Alignment.

Run the full pipeline: data loading -> model training -> XAI explanations ->
consensus analysis -> feature interactions -> expert alignment -> visualization.

Usage:
    # Full experiment on all datasets
    python experiments/3/main.py

    # Specific dataset(s)
    python experiments/3/main.py --datasets nsl-kdd cic-ids-2017

    # Specific phase(s)
    python experiments/3/main.py --phase train explain consensus

    # Quick test with fewer samples
    python experiments/3/main.py --num-explain-samples 100 --datasets nsl-kdd
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add experiment directories to path for local imports.
# exp1 must be inserted first so that exp3 ends up at index 0 (higher priority),
# ensuring exp3's config.py is found before exp1's.
_exp3_dir = os.path.dirname(os.path.abspath(__file__))
_exp1_dir = os.path.join(_exp3_dir, "..", "1")
sys.path.insert(0, _exp1_dir)
sys.path.insert(0, _exp3_dir)

from alignment import alignment_to_dict, compute_alignment_scores
from config import Experiment3Config
from consensus import (
    compute_pairwise_consensus,
    compute_per_attack_consensus,
    consensus_to_dict,
)
from data_loader import DatasetBundle, load_dataset
from explainers import (
    ExplanationResult,
    explain_deeplift,
    explain_ig,
    explain_lime,
    explain_shap_dnn,
    explain_shap_rf,
)
from interactions import (
    aggregate_interaction_matrix,
    compare_interaction_vs_main_effects,
    compute_shap_interaction_values_dnn,
    compute_shap_interaction_values_rf,
    get_top_interactions,
)
from models import (
    DNNWrapper,
    NIDSNet,
    RFWrapper,
    SoftmaxModel,
    load_models,
    save_models,
    train_dnn,
    train_rf,
)
from visualizations import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("experiment3")


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
                f"Compute {props.major}.{props.minor}"
            )
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
        logger.info(f"  GPU {i} [{label}]: {alloc:.2f} GB alloc, {reserved:.2f} GB reserved")


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


# ============================================================================
# Phase: Train
# ============================================================================
def phase_train(
    dataset: DatasetBundle,
    config: Experiment3Config,
    device: torch.device,
    num_gpus: int,
) -> None:
    """Train DNN and RF models, reusing Experiment 1 models if available."""
    logger.info(f"=== TRAINING on {dataset.dataset_name} ===")

    # Check for existing Experiment 1 models first
    exp1_model_dir = config.exp1_output_dir / "models" / dataset.dataset_name
    exp3_model_dir = config.output_dir / "models" / dataset.dataset_name

    if (exp1_model_dir / "dnn.pt").exists() and (exp1_model_dir / "rf.joblib").exists():
        logger.info(f"  Reusing pre-trained models from {exp1_model_dir}")
        # Copy model references by setting output_dir temporarily
        exp3_model_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        if not (exp3_model_dir / "dnn.pt").exists():
            shutil.copy2(exp1_model_dir / "dnn.pt", exp3_model_dir / "dnn.pt")
        if not (exp3_model_dir / "rf.joblib").exists():
            shutil.copy2(exp1_model_dir / "rf.joblib", exp3_model_dir / "rf.joblib")
        return

    if (exp3_model_dir / "dnn.pt").exists() and (exp3_model_dir / "rf.joblib").exists():
        logger.info(f"  Models already exist at {exp3_model_dir}")
        return

    # Train fresh
    logger.info("  No pre-trained models found. Training fresh models...")
    log_gpu_memory("pre-train")

    dnn_model, dnn_wrapper, dnn_metrics = train_dnn(dataset, config.dnn, device, num_gpus)
    log_gpu_memory("post-DNN-train")

    rf_model, rf_wrapper, rf_metrics = train_rf(dataset, config.rf)

    save_models(dnn_model, rf_model, config.output_dir, dataset.dataset_name)

    # Save training metrics
    output_dir = config.output_dir / dataset.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    train_results = {"dnn": dnn_metrics, "rf": rf_metrics}
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(train_results, f, indent=2, default=_json_serialize)


# ============================================================================
# Phase: Explain
# ============================================================================
def phase_explain(
    dataset: DatasetBundle,
    config: Experiment3Config,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Generate XAI explanations using all methods on both models.

    Returns:
        Tuple of (explanations_dict, explain_indices).
        explanations_dict maps "MODEL_METHOD" -> attributions array (n_samples, n_features).
    """
    logger.info(f"=== EXPLAINING on {dataset.dataset_name} ===")
    output_dir = config.output_dir / dataset.dataset_name
    explain_dir = output_dir / "explanations"
    explain_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    dnn_model, rf_model = load_models(
        config.output_dir, dataset.dataset_name,
        dataset.X_train.shape[1], dataset.num_classes,
        config.dnn, device,
    )
    dnn_wrapper = DNNWrapper(dnn_model, device)
    rf_wrapper = RFWrapper(rf_model, num_classes=dataset.num_classes)

    # Sample attack instances only (filter out benign)
    n = min(config.consensus.num_explain_samples, len(dataset.X_test))
    rng = np.random.RandomState(config.seed)
    indices = rng.choice(len(dataset.X_test), size=n, replace=False)
    X_explain = dataset.X_test[indices]
    y_explain = dataset.y_test[indices]

    # Background data for SHAP
    bg_indices = rng.choice(len(dataset.X_train), size=config.explainer.shap_background_samples, replace=False)
    X_background = dataset.X_train[bg_indices]

    explanations: dict[str, np.ndarray] = {}

    # === DNN explanations ===
    logger.info("--- DNN Explanations ---")
    for method_name, explain_fn in [
        ("SHAP", lambda X: explain_shap_dnn(dnn_model, X, X_background, device, config.explainer)),
        ("LIME", lambda X: explain_lime(
            dnn_wrapper.predict_proba, X, dataset.X_train,
            dataset.feature_names, dataset.num_classes, "DNN", config.explainer,
        )),
        ("IG", lambda X: explain_ig(dnn_model, X, device, config.explainer)),
        ("DeepLIFT", lambda X: explain_deeplift(dnn_model, X, device, config.explainer)),
    ]:
        key = f"DNN_{method_name}"
        try:
            logger.info(f"  Generating {key}...")
            result = explain_fn(X_explain)
            explanations[key] = result.attributions
            np.save(explain_dir / f"{key}_attributions.npy", result.attributions)
            logger.info(f"  {key}: {result.attributions.shape}, {result.time_per_sample_ms:.2f} ms/sample")
        except Exception as e:
            logger.error(f"  {key} failed: {e}", exc_info=True)

    # === RF explanations ===
    logger.info("--- RF Explanations ---")
    for method_name, explain_fn in [
        ("SHAP", lambda X: explain_shap_rf(rf_model, X, config.explainer)),
        ("LIME", lambda X: explain_lime(
            rf_wrapper.predict_proba, X, dataset.X_train,
            dataset.feature_names, dataset.num_classes, "RF", config.explainer,
        )),
    ]:
        key = f"RF_{method_name}"
        try:
            logger.info(f"  Generating {key}...")
            result = explain_fn(X_explain)
            explanations[key] = result.attributions
            np.save(explain_dir / f"{key}_attributions.npy", result.attributions)
            logger.info(f"  {key}: {result.attributions.shape}, {result.time_per_sample_ms:.2f} ms/sample")
        except Exception as e:
            logger.error(f"  {key} failed: {e}", exc_info=True)

    # Save indices and labels
    np.save(explain_dir / "explain_indices.npy", indices)
    np.save(explain_dir / "explain_labels.npy", y_explain)

    logger.info(f"  Generated {len(explanations)} explanation sets")
    return explanations, indices


def _load_explanations(
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load saved explanations from disk."""
    explain_dir = output_dir / "explanations"
    explanations = {}

    for path in sorted(explain_dir.glob("*_attributions.npy")):
        key = path.stem.replace("_attributions", "")
        explanations[key] = np.load(path)

    indices = np.load(explain_dir / "explain_indices.npy")
    y_explain = np.load(explain_dir / "explain_labels.npy")

    return explanations, indices, y_explain


# ============================================================================
# Phase: Consensus
# ============================================================================
def phase_consensus(
    dataset: DatasetBundle,
    config: Experiment3Config,
    explanations: dict[str, np.ndarray],
    y_explain: np.ndarray,
) -> dict:
    """Run pairwise consensus analysis between all explainers."""
    logger.info(f"=== CONSENSUS ANALYSIS on {dataset.dataset_name} ===")
    output_dir = config.output_dir / dataset.dataset_name

    # Overall consensus
    logger.info("  Computing overall pairwise consensus...")
    overall_results = compute_pairwise_consensus(explanations, config.consensus)

    for r in overall_results:
        logger.info(
            f"    {r.explainer_a} vs {r.explainer_b}: "
            f"Spearman={r.spearman_mean:.3f}+-{r.spearman_std:.3f}, "
            f"Kendall={r.kendall_mean:.3f}+-{r.kendall_std:.3f}, "
            f"Top-5={r.top_k_intersection.get(5, 0):.3f}, "
            f"Top-10={r.top_k_intersection.get(10, 0):.3f}, "
            f"Wilcoxon p={r.wilcoxon_p_value:.2e} ({'REJECT' if r.wilcoxon_reject_h0 else 'ACCEPT'} H0)"
        )

    # Per-attack-type consensus
    logger.info("  Computing per-attack-type consensus...")
    label_names = list(dataset.label_encoder.classes_)
    per_attack = compute_per_attack_consensus(
        explanations, y_explain, label_names, config.consensus,
    )

    # Save results
    results = {
        "overall": consensus_to_dict(overall_results),
        "per_attack": {
            attack: consensus_to_dict(results_list)
            for attack, results_list in per_attack.items()
        },
    }
    with open(output_dir / "consensus_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_serialize)

    return {"overall": overall_results, "per_attack": per_attack}


# ============================================================================
# Phase: Interactions
# ============================================================================
def phase_interactions(
    dataset: DatasetBundle,
    config: Experiment3Config,
    explanations: dict[str, np.ndarray],
    explain_indices: np.ndarray,
    device: torch.device,
) -> dict:
    """Compute SHAP interaction values and compare with LIME main effects."""
    logger.info(f"=== FEATURE INTERACTION ANALYSIS on {dataset.dataset_name} ===")
    output_dir = config.output_dir / dataset.dataset_name

    # Load models
    dnn_model, rf_model = load_models(
        config.output_dir, dataset.dataset_name,
        dataset.X_train.shape[1], dataset.num_classes,
        config.dnn, device,
    )

    X_explain = dataset.X_test[explain_indices]

    rng = np.random.RandomState(config.seed)
    bg_indices = rng.choice(len(dataset.X_train), size=config.interaction.shap_background_samples, replace=False)
    X_background = dataset.X_train[bg_indices]

    interaction_matrices = {}
    top_interactions_all = {}
    comparison_results = {}

    # RF interaction values (efficient with TreeExplainer)
    try:
        logger.info("  Computing RF SHAP interaction values...")
        rf_interactions = compute_shap_interaction_values_rf(
            rf_model, X_explain, config.interaction,
        )
        rf_agg = aggregate_interaction_matrix(rf_interactions)
        interaction_matrices["RF"] = rf_agg
        top_interactions_all["RF"] = get_top_interactions(
            rf_agg, dataset.feature_names, config.interaction.top_n_interactions,
        )
        logger.info(f"  Top RF interactions:")
        for pair in top_interactions_all["RF"][:5]:
            logger.info(f"    {pair['feature_a']} <-> {pair['feature_b']}: {pair['interaction_strength']:.6f}")

        # Compare with LIME if available
        if "RF_LIME" in explanations:
            comparison_results["RF"] = compare_interaction_vs_main_effects(
                rf_agg, explanations["RF_LIME"], dataset.feature_names,
            )
    except Exception as e:
        logger.error(f"  RF interaction analysis failed: {e}", exc_info=True)

    # DNN interaction values (approximate)
    try:
        logger.info("  Computing DNN SHAP interaction values (approximate)...")
        dnn_interactions = compute_shap_interaction_values_dnn(
            dnn_model, X_explain, X_background, device, config.interaction,
        )
        dnn_agg = aggregate_interaction_matrix(dnn_interactions)
        interaction_matrices["DNN"] = dnn_agg
        top_interactions_all["DNN"] = get_top_interactions(
            dnn_agg, dataset.feature_names, config.interaction.top_n_interactions,
        )
        logger.info(f"  Top DNN interactions:")
        for pair in top_interactions_all["DNN"][:5]:
            logger.info(f"    {pair['feature_a']} <-> {pair['feature_b']}: {pair['interaction_strength']:.6f}")

        # Compare with LIME if available
        if "DNN_LIME" in explanations:
            comparison_results["DNN"] = compare_interaction_vs_main_effects(
                dnn_agg, explanations["DNN_LIME"], dataset.feature_names,
            )
    except Exception as e:
        logger.error(f"  DNN interaction analysis failed: {e}", exc_info=True)

    # Save results
    results = {
        "top_interactions": top_interactions_all,
        "lime_comparison": comparison_results,
    }
    with open(output_dir / "interaction_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_serialize)

    # Save matrices as numpy
    for model_name, matrix in interaction_matrices.items():
        np.save(output_dir / f"interaction_matrix_{model_name}.npy", matrix)

    return {
        "matrices": interaction_matrices,
        "top_interactions": top_interactions_all,
        "comparison": comparison_results,
    }


# ============================================================================
# Phase: Alignment
# ============================================================================
def phase_alignment(
    dataset: DatasetBundle,
    config: Experiment3Config,
    explanations: dict[str, np.ndarray],
    y_explain: np.ndarray,
) -> list[dict]:
    """Score explainer alignment with expert ground truth."""
    logger.info(f"=== EXPERT ALIGNMENT on {dataset.dataset_name} ===")
    output_dir = config.output_dir / dataset.dataset_name

    label_names = list(dataset.label_encoder.classes_)
    results = compute_alignment_scores(
        explanations, y_explain, label_names,
        dataset.feature_names, dataset.dataset_name,
        config.alignment,
    )

    for r in results:
        logger.info(
            f"  {r.attack_type} / {r.explainer_key}: "
            f"RRA={r.rra_score:.3f}, RMA={r.rma_score:.3f} "
            f"({r.n_resolved}/{r.n_expert_features} features resolved)"
        )

    results_dict = alignment_to_dict(results)
    with open(output_dir / "alignment_results.json", "w") as f:
        json.dump(results_dict, f, indent=2, default=_json_serialize)

    return results_dict


# ============================================================================
# Phase: Visualize
# ============================================================================
def phase_visualize(
    dataset: DatasetBundle,
    config: Experiment3Config,
    consensus_results: dict | None,
    interaction_results: dict | None,
    alignment_results: list[dict] | None,
    explanations: dict[str, np.ndarray] | None,
    explain_indices: np.ndarray | None,
) -> None:
    """Generate all visualization plots."""
    logger.info(f"=== VISUALIZATION for {dataset.dataset_name} ===")
    plot_dir = config.output_dir / "plots"

    # Prepare SHAP values for dependence plots
    shap_values = None
    X_data = None
    if explanations and explain_indices is not None:
        if "DNN_SHAP" in explanations:
            shap_values = explanations["DNN_SHAP"]
            X_data = dataset.X_test[explain_indices]
        elif "RF_SHAP" in explanations:
            shap_values = explanations["RF_SHAP"]
            X_data = dataset.X_test[explain_indices]

    generate_all_plots(
        consensus_results=consensus_results.get("overall") if consensus_results else None,
        per_attack_consensus=consensus_results.get("per_attack") if consensus_results else None,
        interaction_matrices=interaction_results.get("matrices") if interaction_results else None,
        top_interactions=interaction_results.get("top_interactions") if interaction_results else None,
        alignment_results=alignment_results,
        feature_names=dataset.feature_names,
        dataset_name=dataset.dataset_name,
        plot_dir=plot_dir,
        config=config,
        shap_values_for_dependence=shap_values,
        X_data_for_dependence=X_data,
    )


# ============================================================================
# Main Pipeline
# ============================================================================
def run_experiment(config: Experiment3Config, datasets: list[str], phases: list[str]):
    """Main experiment runner."""
    device, num_gpus = setup_device()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    all_phases = {"train", "explain", "consensus", "interactions", "alignment", "visualize"}
    run_all = "all" in phases
    active_phases = all_phases if run_all else set(phases)

    # Save config
    with open(config.output_dir / "config.json", "w") as f:
        json.dump({
            "datasets": datasets,
            "phases": list(active_phases),
            "consensus": vars(config.consensus),
            "interaction": vars(config.interaction),
            "alignment": vars(config.alignment),
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
            logger.error(f"Failed to load {ds_name}: {e}", exc_info=True)
            continue

        ds_output = config.output_dir / ds_name
        ds_output.mkdir(parents=True, exist_ok=True)
        ds_results = {}

        # Phase: Train
        if "train" in active_phases:
            try:
                phase_train(dataset, config, device, num_gpus)
            except Exception as e:
                logger.error(f"Training failed for {ds_name}: {e}", exc_info=True)
                continue
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Phase: Explain
        explanations = None
        explain_indices = None
        y_explain = None
        if "explain" in active_phases:
            try:
                explanations, explain_indices = phase_explain(dataset, config, device)
                y_explain = dataset.y_test[explain_indices]
            except Exception as e:
                logger.error(f"Explanation failed for {ds_name}: {e}", exc_info=True)
                continue
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Load explanations if not generated in this run
        if explanations is None:
            try:
                explanations, explain_indices, y_explain = _load_explanations(ds_output)
                logger.info(f"  Loaded {len(explanations)} explanation sets from disk")
            except Exception as e:
                logger.error(f"Could not load explanations for {ds_name}: {e}")
                if any(p in active_phases for p in ["consensus", "interactions", "alignment", "visualize"]):
                    logger.error("  Cannot proceed without explanations. Skipping remaining phases.")
                    continue

        # Phase: Consensus
        consensus_results = None
        if "consensus" in active_phases and explanations:
            try:
                consensus_results = phase_consensus(dataset, config, explanations, y_explain)
                ds_results["consensus"] = consensus_results
            except Exception as e:
                logger.error(f"Consensus failed for {ds_name}: {e}", exc_info=True)

        # Phase: Interactions
        interaction_results = None
        if "interactions" in active_phases and explanations:
            try:
                interaction_results = phase_interactions(
                    dataset, config, explanations, explain_indices, device,
                )
                ds_results["interactions"] = interaction_results
            except Exception as e:
                logger.error(f"Interaction analysis failed for {ds_name}: {e}", exc_info=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Phase: Alignment
        alignment_results = None
        if "alignment" in active_phases and explanations:
            try:
                alignment_results = phase_alignment(dataset, config, explanations, y_explain)
                ds_results["alignment"] = alignment_results
            except Exception as e:
                logger.error(f"Alignment failed for {ds_name}: {e}", exc_info=True)

        # Phase: Visualize
        if "visualize" in active_phases:
            try:
                phase_visualize(
                    dataset, config,
                    consensus_results, interaction_results, alignment_results,
                    explanations, explain_indices,
                )
            except Exception as e:
                logger.error(f"Visualization failed for {ds_name}: {e}", exc_info=True)

        ds_elapsed = time.time() - ds_start
        logger.info(f"Dataset {ds_name} completed in {ds_elapsed:.1f}s")
        all_results[ds_name] = ds_results

    # Save combined results summary
    _save_summary(all_results, config.output_dir)
    _print_final_summary(all_results)


def _save_summary(all_results: dict, output_dir: Path) -> None:
    """Save a lightweight summary of all results."""
    summary = {}
    for ds_name, ds_results in all_results.items():
        ds_summary = {}
        if "consensus" in ds_results and ds_results["consensus"]:
            overall = ds_results["consensus"].get("overall", [])
            if overall:
                ds_summary["consensus_mean_spearman"] = float(
                    np.mean([r.spearman_mean for r in overall])
                )
                ds_summary["consensus_mean_kendall"] = float(
                    np.mean([r.kendall_mean for r in overall])
                )
                ds_summary["n_significant_divergences"] = sum(
                    1 for r in overall if r.wilcoxon_reject_h0
                )
                ds_summary["total_pairs"] = len(overall)

        if "alignment" in ds_results and ds_results["alignment"]:
            alignment = ds_results["alignment"]
            ds_summary["mean_rra"] = float(np.mean([r["rra_score"] for r in alignment]))
            ds_summary["mean_rma"] = float(np.mean([r["rma_score"] for r in alignment]))

        summary[ds_name] = ds_summary

    with open(output_dir / "experiment3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_serialize)


def _print_final_summary(all_results: dict) -> None:
    """Print human-readable summary."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 3 SUMMARY: Feature Interaction, Consensus & Alignment")
    logger.info("=" * 80)

    for ds_name, ds_results in all_results.items():
        logger.info(f"\n--- {ds_name} ---")

        if "consensus" in ds_results and ds_results["consensus"]:
            overall = ds_results["consensus"].get("overall", [])
            if overall:
                mean_sp = np.mean([r.spearman_mean for r in overall])
                mean_kt = np.mean([r.kendall_mean for r in overall])
                n_sig = sum(1 for r in overall if r.wilcoxon_reject_h0)
                logger.info(
                    f"  Consensus: Mean Spearman={mean_sp:.3f}, Mean Kendall={mean_kt:.3f}, "
                    f"{n_sig}/{len(overall)} pairs show significant divergence"
                )

        if "alignment" in ds_results and ds_results["alignment"]:
            alignment = ds_results["alignment"]
            mean_rra = np.mean([r["rra_score"] for r in alignment])
            mean_rma = np.mean([r["rma_score"] for r in alignment])
            logger.info(f"  Alignment: Mean RRA={mean_rra:.3f}, Mean RMA={mean_rma:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Feature Interaction, Consensus & Human-in-the-Loop Alignment"
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
        choices=["all", "train", "explain", "consensus", "interactions", "alignment", "visualize"],
        help="Experiment phase(s) to run",
    )
    parser.add_argument(
        "--num-explain-samples",
        type=int,
        default=10000,
        help="Number of test samples to generate explanations for",
    )
    parser.add_argument(
        "--interaction-samples",
        type=int,
        default=500,
        help="Number of samples for SHAP interaction values",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--exp1-dir",
        type=str,
        default=None,
        help="Experiment 1 results directory (for model reuse)",
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
    config = Experiment3Config()

    # Apply CLI overrides
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.exp1_dir:
        config.exp1_output_dir = Path(args.exp1_dir)
    if args.no_smote:
        config.data.apply_smote = False
    config.seed = args.seed
    config.consensus.num_explain_samples = args.num_explain_samples
    config.explainer.num_explain_samples = args.num_explain_samples
    config.interaction.shap_interaction_samples = args.interaction_samples

    datasets = args.datasets or config.ALL_DATASETS
    phases = args.phase

    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger.info("Experiment 3: Feature Interaction, Consensus & Human-in-the-Loop Alignment")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Phases: {phases}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Explain samples: {config.consensus.num_explain_samples}")
    logger.info(f"Interaction samples: {config.interaction.shap_interaction_samples}")

    run_experiment(config, datasets, phases)


if __name__ == "__main__":
    main()
