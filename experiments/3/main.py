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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Add experiment directories to path for local imports.
# Only exp3_dir and commons_dir go on sys.path. exp1_dir is NOT added because
# exp1's models.py/explainers.py do `from config import CNNGRUConfig` which
# would find exp3's config.py instead of exp1's. Commons has its own config.py
# (identical to exp1's) so commons modules resolve their imports correctly.
# pa_explainers is imported via importlib to avoid polluting sys.path with exp1.
_exp3_dir = os.path.dirname(os.path.abspath(__file__))
_commons_dir = os.path.join(_exp3_dir, "..", "commons")
_exp1_dir = os.path.join(_exp3_dir, "..", "1")
sys.path.insert(0, _commons_dir)
sys.path.insert(0, _exp3_dir)

from alignment import alignment_to_dict, compute_alignment_scores
from config import Experiment3Config, load_experiment3_config
from consensus import (
    compute_pairwise_consensus,
    compute_per_attack_consensus,
    consensus_to_dict,
)


def tag_pair(key_a: str, key_b: str) -> str:
    """Classify a consensus pair as within-mode or cross-mode."""
    a_is_pa = "PA-" in key_a
    b_is_pa = "PA-" in key_b
    if a_is_pa == b_is_pa:
        return "within-pa" if a_is_pa else "within-normal"
    return "cross-mode"
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
# Import pa_explainers from exp1 via importlib (exp1_dir is NOT on sys.path
# to avoid config.py conflicts between exp3 and exp1).
import importlib.util as _ilu

def _import_from_exp1(module_name: str, file_name: str):
    """Import a module from exp1 without adding exp1 to sys.path."""
    path = os.path.join(_exp1_dir, file_name)
    spec = _ilu.spec_from_file_location(f"exp1_{module_name}", path)
    mod = _ilu.module_from_spec(spec)
    # Temporarily add exp1 to sys.path for transitive imports within exp1
    sys.path.insert(0, _exp1_dir)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(_exp1_dir)
    return mod

_pa_explainers = _import_from_exp1("pa_explainers", "pa_explainers.py")
pa_explain_deeplift = _pa_explainers.pa_explain_deeplift
pa_explain_ig = _pa_explainers.pa_explain_ig
pa_explain_lime = _pa_explainers.pa_explain_lime
pa_explain_shap_dnn = _pa_explainers.pa_explain_shap_dnn
pa_explain_shap_tree = _pa_explainers.pa_explain_shap_tree

from visualizations import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("experiment3")


from config_loader import config_section_hash


def _checkpoint_path(output_dir: Path, dataset_name: str, phase_name: str) -> Path:
    """Return the path for a phase checkpoint marker."""
    return output_dir / dataset_name / f".phase_{phase_name}.done"


def _is_phase_done(output_dir: Path, dataset_name: str, phase_name: str, config_hash: str) -> bool:
    """Check if a phase has already completed with the same config."""
    marker = _checkpoint_path(output_dir, dataset_name, phase_name)
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text())
        return data.get("config_hash") == config_hash
    except (json.JSONDecodeError, KeyError):
        return False


def _mark_phase_done(output_dir: Path, dataset_name: str, phase_name: str, config_hash: str) -> None:
    """Write a phase completion marker."""
    marker = _checkpoint_path(output_dir, dataset_name, phase_name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
    }))


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
# Explanation key naming
# ============================================================================

# Normal: DNN_SHAP, DNN_LIME, DNN_IG, DNN_DeepLIFT, RF_SHAP, RF_LIME
# PA:     DNN_PA-SHAP, DNN_PA-LIME, DNN_PA-IG, DNN_PA-DeepLIFT, RF_PA-SHAP, RF_PA-LIME

NORMAL_KEYS = ["DNN_SHAP", "DNN_LIME", "DNN_IG", "DNN_DeepLIFT", "RF_SHAP", "RF_LIME"]
PA_KEYS = ["DNN_PA-SHAP", "DNN_PA-LIME", "DNN_PA-IG", "DNN_PA-DeepLIFT", "RF_PA-SHAP", "RF_PA-LIME"]

# Exp1 PA mode saves "DNN_SHAP" (not "DNN_PA-SHAP") in the protocol-aware/ dir.
_EXP1_FILENAME_MAP = {
    "DNN_SHAP": "DNN_SHAP", "DNN_LIME": "DNN_LIME",
    "DNN_IG": "DNN_IG", "DNN_DeepLIFT": "DNN_DeepLIFT",
    "RF_SHAP": "RF_SHAP", "RF_LIME": "RF_LIME",
    "DNN_PA-SHAP": "DNN_SHAP", "DNN_PA-LIME": "DNN_LIME",
    "DNN_PA-IG": "DNN_IG", "DNN_PA-DeepLIFT": "DNN_DeepLIFT",
    "RF_PA-SHAP": "RF_SHAP", "RF_PA-LIME": "RF_LIME",
}


def _exp1_dir_for_mode(exp1_output_dir: Path, mode: str) -> Path:
    """Return the exp1 output directory for a given XAI mode."""
    if mode == "pa":
        return exp1_output_dir / "protocol-aware"
    return exp1_output_dir / "normal"


def _load_explanations_for_mode(
    exp3_explain_dir: Path,
    exp1_explain_dir: Path | None,
    keys: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Try to load explanations for a set of keys.

    Load priority: exp3 cache -> exp1 output -> return what we found.
    """
    loaded: dict[str, np.ndarray] = {}
    indices = None

    # Try loading indices
    if (exp3_explain_dir / "explain_indices.npy").exists():
        indices = np.load(exp3_explain_dir / "explain_indices.npy")
    elif exp1_explain_dir and (exp1_explain_dir / "explain_indices.npy").exists():
        indices = np.load(exp1_explain_dir / "explain_indices.npy")

    for key in keys:
        # Try exp3 cache first
        exp3_path = exp3_explain_dir / f"{key}_attributions.npy"
        if exp3_path.exists():
            loaded[key] = np.load(exp3_path)
            continue

        # Try exp1 output (different filename for PA keys)
        if exp1_explain_dir:
            exp1_filename = _EXP1_FILENAME_MAP.get(key, key)
            exp1_path = exp1_explain_dir / f"{exp1_filename}_attributions.npy"
            if exp1_path.exists():
                loaded[key] = np.load(exp1_path)
                # Cache in exp3 dir for next run
                exp3_explain_dir.mkdir(parents=True, exist_ok=True)
                np.save(exp3_path, loaded[key])
                logger.info(f"    Loaded {key} from exp1 and cached in exp3")
                continue

    return loaded, indices


# ============================================================================
# Phase: Explain (unified normal + PA)
# ============================================================================
def phase_explain(
    dataset: DatasetBundle,
    config: Experiment3Config,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load or generate XAI explanations for all configured modes.

    Returns:
        Tuple of (unified_explanations_dict, explain_indices).
        Keys follow the unified naming: DNN_SHAP, DNN_PA-SHAP, etc.
    """
    logger.info(f"=== EXPLAINING on {dataset.dataset_name} ===")

    unified: dict[str, np.ndarray] = {}
    shared_indices: np.ndarray | None = None

    for mode in config.xai_modes:
        keys = PA_KEYS if mode == "pa" else NORMAL_KEYS
        mode_label = "protocol-aware" if mode == "pa" else "normal"
        logger.info(f"--- Loading/generating {mode_label} explanations ---")

        exp3_explain_dir = config.output_dir / dataset.dataset_name / "explanations" / mode_label
        exp3_explain_dir.mkdir(parents=True, exist_ok=True)

        exp1_explain_dir = _exp1_dir_for_mode(config.exp1_output_dir, mode) / dataset.dataset_name / "explanations"

        # Try loading cached explanations
        loaded, loaded_indices = _load_explanations_for_mode(
            exp3_explain_dir, exp1_explain_dir, keys,
        )

        if loaded_indices is not None:
            if shared_indices is not None and not np.array_equal(shared_indices, loaded_indices):
                logger.error(
                    f"  Index mismatch between modes for {dataset.dataset_name}! "
                    f"Normal has {len(shared_indices)} indices, {mode_label} has {len(loaded_indices)}. "
                    f"Cross-mode consensus will be unreliable."
                )
            if shared_indices is None:
                shared_indices = loaded_indices

        missing_keys = [k for k in keys if k not in loaded]

        if not missing_keys:
            logger.info(f"  All {mode_label} explanations loaded from cache ({len(loaded)} keys)")
            unified.update(loaded)
            continue

        logger.info(f"  Missing {len(missing_keys)} {mode_label} explanations: {missing_keys}")
        logger.info(f"  Generating missing explanations...")

        # Need to generate — load models
        dnn_model, rf_model = load_models(
            config.exp1_output_dir, dataset.dataset_name,
            dataset.X_train.shape[1], dataset.num_classes,
            config.dnn, device,
        )
        dnn_wrapper = DNNWrapper(dnn_model, device)
        rf_wrapper = RFWrapper(rf_model, num_classes=dataset.num_classes)

        # Determine indices (reuse loaded or generate fresh)
        if shared_indices is not None:
            indices = shared_indices
        else:
            n = min(config.consensus.num_explain_samples, len(dataset.X_test))
            rng = np.random.RandomState(config.seed)
            indices = rng.choice(len(dataset.X_test), size=n, replace=False)
            shared_indices = indices

        X_explain = dataset.X_test[indices]

        # Background data for SHAP (normal mode)
        rng_bg = np.random.RandomState(config.seed + 1)
        bg_indices = rng_bg.choice(len(dataset.X_train), size=config.explainer.shap_background_samples, replace=False)
        X_background = dataset.X_train[bg_indices]

        checkpoint_dir = config.output_dir / dataset.dataset_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Generate each missing key
        for key in missing_keys:
            try:
                result = _generate_single_explanation(
                    key, mode, dnn_model, rf_model, dnn_wrapper, rf_wrapper,
                    X_explain, X_background, dataset, device, config, checkpoint_dir,
                )
                if result is not None:
                    loaded[key] = result.attributions
                    np.save(exp3_explain_dir / f"{key}_attributions.npy", result.attributions)
                    logger.info(f"    {key}: {result.attributions.shape}, {result.time_per_sample_ms:.2f} ms/sample")
            except Exception as e:
                logger.error(f"    {key} failed: {e}", exc_info=True)

        # Save indices for this mode
        np.save(exp3_explain_dir / "explain_indices.npy", indices)

        unified.update(loaded)

        # Free GPU memory between modes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if shared_indices is None:
        raise RuntimeError(f"No explanations could be loaded or generated for {dataset.dataset_name}")

    # Save unified indices and labels at dataset level
    ds_dir = config.output_dir / dataset.dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)
    np.save(ds_dir / "explain_indices.npy", shared_indices)
    np.save(ds_dir / "explain_labels.npy", dataset.y_test[shared_indices])

    logger.info(f"  Unified pool: {len(unified)} explanation sets")
    return unified, shared_indices


def _generate_single_explanation(
    key: str,
    mode: str,
    dnn_model,
    rf_model,
    dnn_wrapper,
    rf_wrapper,
    X_explain: np.ndarray,
    X_background: np.ndarray,
    dataset: DatasetBundle,
    device: torch.device,
    config: Experiment3Config,
    checkpoint_dir: Path,
) -> ExplanationResult | None:
    """Generate a single explanation by key name."""
    ds_name = dataset.dataset_name

    if mode == "pa":
        generators = {
            "DNN_PA-SHAP": lambda: pa_explain_shap_dnn(
                dnn_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, device, config.explainer,
            ),
            "DNN_PA-LIME": lambda: pa_explain_lime(
                dnn_wrapper.predict_proba, X_explain, dataset.X_train,
                ds_name, "DNN", config.explainer, checkpoint_dir,
            ),
            "DNN_PA-IG": lambda: pa_explain_ig(
                dnn_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, device, config.explainer,
            ),
            "DNN_PA-DeepLIFT": lambda: pa_explain_deeplift(
                dnn_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, device, config.explainer,
            ),
            "RF_PA-SHAP": lambda: pa_explain_shap_tree(
                rf_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, config.explainer, "RF",
            ),
            "RF_PA-LIME": lambda: pa_explain_lime(
                rf_wrapper.predict_proba, X_explain, dataset.X_train,
                ds_name, "RF", config.explainer, checkpoint_dir,
            ),
        }
    else:
        generators = {
            "DNN_SHAP": lambda: explain_shap_dnn(
                dnn_model, X_explain, X_background, device, config.explainer,
            ),
            "DNN_LIME": lambda: explain_lime(
                dnn_wrapper.predict_proba, X_explain, dataset.X_train,
                dataset.feature_names, dataset.num_classes, "DNN", config.explainer,
            ),
            "DNN_IG": lambda: explain_ig(dnn_model, X_explain, device, config.explainer),
            "DNN_DeepLIFT": lambda: explain_deeplift(dnn_model, X_explain, device, config.explainer),
            "RF_SHAP": lambda: explain_shap_rf(rf_model, X_explain, config.explainer),
            "RF_LIME": lambda: explain_lime(
                rf_wrapper.predict_proba, X_explain, dataset.X_train,
                dataset.feature_names, dataset.num_classes, "RF", config.explainer,
            ),
        }

    gen_fn = generators.get(key)
    if gen_fn is None:
        logger.warning(f"  No generator for key: {key}")
        return None

    logger.info(f"    Generating {key}...")
    return gen_fn()


def _load_all_explanations(
    ds_output_dir: Path,
    xai_modes: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load all explanations from disk across all modes."""
    explanations = {}
    indices = None

    for mode in xai_modes:
        mode_label = "protocol-aware" if mode == "pa" else "normal"
        explain_dir = ds_output_dir / "explanations" / mode_label

        if not explain_dir.exists():
            continue

        for path in sorted(explain_dir.glob("*_attributions.npy")):
            key = path.stem.replace("_attributions", "")
            explanations[key] = np.load(path)

        idx_path = explain_dir / "explain_indices.npy"
        if idx_path.exists() and indices is None:
            indices = np.load(idx_path)

    # Also check dataset-level indices/labels
    if indices is None and (ds_output_dir / "explain_indices.npy").exists():
        indices = np.load(ds_output_dir / "explain_indices.npy")

    if (ds_output_dir / "explain_labels.npy").exists():
        y_explain = np.load(ds_output_dir / "explain_labels.npy")
    else:
        y_explain = None

    if indices is None:
        raise FileNotFoundError(f"No explain_indices.npy found in {ds_output_dir}")

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
    """Main experiment runner with checkpoint support."""
    device, num_gpus = setup_device()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    all_phases = {"train", "explain", "consensus", "interactions", "alignment", "visualize"}
    run_all = "all" in phases
    active_phases = all_phases if run_all else set(phases)

    # Remove alignment if disabled
    if not config.alignment.enabled:
        active_phases.discard("alignment")

    # Save config
    with open(config.output_dir / "config.json", "w") as f:
        json.dump({
            "datasets": datasets,
            "phases": list(active_phases),
            "xai_modes": config.xai_modes,
            "consensus": vars(config.consensus),
            "interaction": vars(config.interaction),
            "alignment": vars(config.alignment),
            "parallelism": vars(config.parallelism),
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
            train_hash = config_section_hash({"dnn": vars(config.dnn), "rf": vars(config.rf), "seed": config.seed})
            if _is_phase_done(config.output_dir, ds_name, "train", train_hash):
                logger.info("  Train phase: already done (checkpoint found)")
            else:
                try:
                    phase_train(dataset, config, device, num_gpus)
                    _mark_phase_done(config.output_dir, ds_name, "train", train_hash)
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
            explain_hash = config_section_hash({
                "explainer": vars(config.explainer),
                "consensus_samples": config.consensus.num_explain_samples,
                "seed": config.seed, "modes": config.xai_modes,
            })
            if _is_phase_done(config.output_dir, ds_name, "explain", explain_hash):
                logger.info("  Explain phase: already done (checkpoint found)")
            else:
                try:
                    explanations, explain_indices = phase_explain(dataset, config, device)
                    y_explain = dataset.y_test[explain_indices]
                    _mark_phase_done(config.output_dir, ds_name, "explain", explain_hash)
                except Exception as e:
                    logger.error(f"Explanation failed for {ds_name}: {e}", exc_info=True)
                    continue
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Load explanations if not generated in this run
        if explanations is None:
            try:
                explanations, explain_indices, y_explain = _load_all_explanations(
                    ds_output, config.xai_modes,
                )
                logger.info(f"  Loaded {len(explanations)} explanation sets from disk")
            except Exception as e:
                logger.error(f"Could not load explanations for {ds_name}: {e}")
                if any(p in active_phases for p in ["consensus", "interactions", "alignment", "visualize"]):
                    logger.error("  Cannot proceed without explanations. Skipping remaining phases.")
                    continue

        # Phase: Consensus
        consensus_results = None
        if "consensus" in active_phases and explanations:
            cons_hash = config_section_hash({"consensus": vars(config.consensus), "keys": sorted(explanations.keys())})
            if _is_phase_done(config.output_dir, ds_name, "consensus", cons_hash):
                logger.info("  Consensus phase: already done (checkpoint found)")
                try:
                    with open(ds_output / "consensus_results.json") as f:
                        consensus_results = json.load(f)
                except Exception:
                    pass
            else:
                try:
                    consensus_results = phase_consensus(dataset, config, explanations, y_explain)
                    ds_results["consensus"] = consensus_results
                    _mark_phase_done(config.output_dir, ds_name, "consensus", cons_hash)
                except Exception as e:
                    logger.error(f"Consensus failed for {ds_name}: {e}", exc_info=True)

        # Phase: Interactions
        interaction_results = None
        if "interactions" in active_phases and explanations:
            inter_hash = config_section_hash({"interaction": vars(config.interaction), "seed": config.seed})
            if _is_phase_done(config.output_dir, ds_name, "interactions", inter_hash):
                logger.info("  Interactions phase: already done (checkpoint found)")
            else:
                try:
                    interaction_results = phase_interactions(
                        dataset, config, explanations, explain_indices, device,
                    )
                    ds_results["interactions"] = interaction_results
                    _mark_phase_done(config.output_dir, ds_name, "interactions", inter_hash)
                except Exception as e:
                    logger.error(f"Interaction analysis failed for {ds_name}: {e}", exc_info=True)
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Phase: Alignment (optional)
        alignment_results = None
        if "alignment" in active_phases and explanations and config.alignment.enabled:
            align_hash = config_section_hash({"alignment": vars(config.alignment), "keys": sorted(explanations.keys())})
            if _is_phase_done(config.output_dir, ds_name, "alignment", align_hash):
                logger.info("  Alignment phase: already done (checkpoint found)")
            else:
                try:
                    alignment_results = phase_alignment(dataset, config, explanations, y_explain)
                    ds_results["alignment"] = alignment_results
                    _mark_phase_done(config.output_dir, ds_name, "alignment", align_hash)
                except Exception as e:
                    logger.error(f"Alignment failed for {ds_name}: {e}", exc_info=True)

        # Phase: Visualize (always re-run)
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
    """Print human-readable summary with within/cross-mode breakdown."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 3 SUMMARY: Feature Interaction, Consensus & Expert Alignment")
    logger.info("=" * 80)

    for ds_name, ds_results in all_results.items():
        logger.info(f"\n--- {ds_name} ---")

        if "consensus" in ds_results and ds_results["consensus"]:
            overall = ds_results["consensus"].get("overall", [])
            if overall:
                for pair_type in ["within-normal", "within-pa", "cross-mode"]:
                    typed = [r for r in overall if tag_pair(r.explainer_a, r.explainer_b) == pair_type]
                    if not typed:
                        continue
                    mean_sp = np.mean([r.spearman_mean for r in typed])
                    mean_kt = np.mean([r.kendall_mean for r in typed])
                    n_sig = sum(1 for r in typed if r.wilcoxon_reject_h0)
                    logger.info(
                        f"  Consensus [{pair_type}]: "
                        f"Mean Spearman={mean_sp:.3f}, Mean Kendall={mean_kt:.3f}, "
                        f"{n_sig}/{len(typed)} pairs show significant divergence"
                    )

        if "alignment" in ds_results and ds_results["alignment"]:
            alignment = ds_results["alignment"]
            mean_rra = np.mean([r["rra_score"] for r in alignment])
            mean_rma = np.mean([r["rma_score"] for r in alignment])
            logger.info(f"  Alignment: Mean RRA={mean_rra:.3f}, Mean RMA={mean_rma:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Feature Interaction, Consensus & Expert Alignment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: experiments/commons/config.yaml)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=["nsl-kdd", "cic-ids-2017", "unsw-nb15", "cse-cic-ids2018"],
        help="Datasets to process (default: from YAML config)",
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        default=["all"],
        choices=["all", "train", "explain", "consensus", "interactions", "alignment", "visualize"],
        help="Experiment phase(s) to run",
    )
    parser.add_argument(
        "--xai-modes",
        nargs="+",
        default=None,
        choices=["normal", "pa"],
        help="XAI modes to include (default: from YAML config)",
    )
    parser.add_argument(
        "--num-explain-samples",
        type=int,
        default=None,
        help="Override number of test samples for explanations",
    )
    parser.add_argument(
        "--interaction-samples",
        type=int,
        default=None,
        help="Override number of samples for SHAP interaction values",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--no-alignment",
        action="store_true",
        help="Disable expert alignment phase",
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
        help="Override random seed",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoints and rerun everything",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config from YAML
    config = load_experiment3_config(args.config)

    # Apply CLI overrides
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.no_smote:
        config.data.apply_smote = False
    if args.seed is not None:
        config.seed = args.seed
    if args.num_explain_samples is not None:
        config.consensus.num_explain_samples = args.num_explain_samples
        config.explainer.num_explain_samples = args.num_explain_samples
    if args.interaction_samples is not None:
        config.interaction.shap_interaction_samples = args.interaction_samples
    if args.xai_modes:
        config.xai_modes = args.xai_modes
    if args.no_alignment:
        config.alignment.enabled = False

    # Clear checkpoints if --no-resume
    if args.no_resume:
        import glob as glob_mod
        for marker in glob_mod.glob(str(config.output_dir / "*" / ".phase_*.done")):
            os.remove(marker)
        logger.info("Cleared all phase checkpoints")

    datasets = args.datasets or config.ALL_DATASETS
    phases = args.phase

    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger.info("Experiment 3: Feature Interaction, Consensus & Expert Alignment")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Phases: {phases}")
    logger.info(f"XAI modes: {config.xai_modes}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Explain samples: {config.consensus.num_explain_samples}")
    logger.info(f"Interaction samples: {config.interaction.shap_interaction_samples}")
    logger.info(f"Alignment: {'enabled' if config.alignment.enabled else 'disabled'}")

    run_experiment(config, datasets, phases)


if __name__ == "__main__":
    main()
