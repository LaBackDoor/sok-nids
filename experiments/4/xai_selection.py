"""XAI-driven feature selection pipeline for Experiment 4.

Imports Exp 1's parallelized explainer orchestrators (normal + PA-XAI)
to generate attributions, then converts to feature importance rankings
for iterative pruning.

Normal mode: SHAP, LIME, IG, DeepLIFT (vanilla, no domain constraints)
PA mode: Protocol-Aware versions with network protocol constraints
"""

import logging
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from config import ExperimentConfig, ExplainerConfig, XAISelectionConfig
from feature_selection import FeatureSelectionResult

logger = logging.getLogger(__name__)


def _attributions_to_importance(attributions: np.ndarray) -> np.ndarray:
    """Convert per-sample attributions to global feature importance.

    Args:
        attributions: shape (n_samples, n_features)

    Returns:
        Feature importance array of shape (n_features,).
    """
    return np.mean(np.abs(attributions), axis=0)


def _iterative_prune(
    feature_importance: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    config: XAISelectionConfig,
    dataset_name: str,
) -> tuple[np.ndarray, list[dict]]:
    """Iteratively prune features using importance scores.

    Removes least important features step by step, evaluating F1 at each
    step. Stops when F1 degrades beyond threshold.

    Returns:
        selected_indices: optimal feature subset indices
        pruning_history: list of dicts with step details
    """
    logger.info("    Starting iterative pruning...")

    n_features = len(feature_importance)
    ranked_indices = np.argsort(feature_importance)[::-1]  # descending importance

    # Train baseline RF on all features for F1 reference
    rf_baseline = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf_baseline.fit(X_train, y_train)
    y_pred_baseline = rf_baseline.predict(X_val)
    baseline_f1 = f1_score(y_val, y_pred_baseline, average="weighted", zero_division=0)
    logger.info(f"    Baseline F1 (all {n_features} features): {baseline_f1:.4f}")

    # Target from config
    target_n = config.target_features.get(dataset_name, config.min_features)
    min_n = max(config.min_features, target_n)

    pruning_history = [{
        "n_features": n_features,
        "f1": baseline_f1,
        "features_removed": 0,
    }]

    current_indices = ranked_indices.copy()
    best_indices = current_indices.copy()
    best_f1 = baseline_f1

    # Reuse a single RF across pruning steps with warm_start
    rf_reduced = RandomForestClassifier(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=42, warm_start=True,
    )

    while len(current_indices) > min_n:
        # Remove pruning_step_ratio of remaining features
        n_remove = max(1, int(len(current_indices) * config.pruning_step_ratio))
        n_keep = max(min_n, len(current_indices) - n_remove)

        # Keep top-ranked features
        current_importance = feature_importance[current_indices]
        reranked = np.argsort(current_importance)[::-1]
        current_indices = current_indices[reranked[:n_keep]]

        # Evaluate with reduced features (warm_start reuses prior trees)
        rf_reduced.fit(X_train[:, current_indices], y_train)
        y_pred = rf_reduced.predict(X_val[:, current_indices])
        current_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        step_info = {
            "n_features": len(current_indices),
            "f1": float(current_f1),
            "f1_drop": float(baseline_f1 - current_f1),
        }
        pruning_history.append(step_info)

        logger.info(
            f"    {len(current_indices)} features: F1={current_f1:.4f} "
            f"(drop={baseline_f1 - current_f1:.4f})"
        )

        # Check if degradation exceeds threshold
        if baseline_f1 - current_f1 > config.f1_degradation_threshold:
            logger.info(
                f"    F1 degradation ({baseline_f1 - current_f1:.4f}) exceeds "
                f"threshold ({config.f1_degradation_threshold}). "
                f"Reverting to previous step ({len(best_indices)} features)."
            )
            break

        best_indices = current_indices.copy()
        best_f1 = current_f1

    selected = np.sort(best_indices)
    logger.info(f"    Final selection: {len(selected)} features (F1={best_f1:.4f})")
    return selected, pruning_history


def _run_single_mode(
    mode: str,
    dnn_model: nn.Module,
    rf_model: RandomForestClassifier,
    dnn_wrapper,
    rf_wrapper,
    dataset,
    device: torch.device,
    config: ExperimentConfig,
    checkpoint_dir=None,
) -> list[FeatureSelectionResult]:
    """Run XAI pipeline for a single mode (normal or pa).

    Calls Exp 1's orchestrators to generate attributions, then converts
    to feature importance and runs iterative pruning per method.
    """
    mode_prefix = "PA-" if mode == "pa" else ""
    logger.info(f"  === XAI Feature Selection ({mode.upper()} mode) ===")

    # Generate attributions using Exp 1's orchestrators
    if mode == "pa":
        from pa_explainers import pa_generate_all_explanations
        explanation_results, explain_indices = pa_generate_all_explanations(
            dnn_model, rf_model, dnn_wrapper, rf_wrapper,
            dataset, device, config.explainer,
            checkpoint_dir=checkpoint_dir,
        )
    else:
        from explainers import generate_all_explanations
        explanation_results, explain_indices = generate_all_explanations(
            dnn_model, rf_model, dnn_wrapper, rf_wrapper,
            dataset, device, config.explainer,
        )

    # Convert each ExplanationResult to a FeatureSelectionResult via pruning
    selection_results: list[FeatureSelectionResult] = []

    for exp_result in explanation_results:
        method_label = f"{mode_prefix}{exp_result.method_name}-{exp_result.model_name}"
        logger.info(f"  Processing {method_label} attributions for feature selection...")

        t0 = time.time()

        importance = _attributions_to_importance(exp_result.attributions)

        selected_indices, pruning_history = _iterative_prune(
            importance,
            dataset.X_train, dataset.y_train,
            dataset.X_val, dataset.y_val,
            dataset.feature_names,
            config.xai,
            dataset.dataset_name,
        )

        elapsed = exp_result.total_time_s + (time.time() - t0)

        selection_results.append(FeatureSelectionResult(
            method_name=method_label,
            selected_indices=selected_indices,
            feature_rankings=importance,
            selected_feature_names=[dataset.feature_names[i] for i in selected_indices],
            n_original=len(dataset.feature_names),
            n_selected=len(selected_indices),
            selection_time_s=elapsed,
        ))

    return selection_results


def run_xai_pipeline(
    dnn_model: nn.Module,
    rf_model: RandomForestClassifier,
    dnn_wrapper,
    rf_wrapper,
    dataset,
    device: torch.device,
    config: ExperimentConfig,
) -> list[FeatureSelectionResult]:
    """Run XAI-driven feature selection for all configured modes.

    For each mode in config.xai_modes (e.g. ["normal", "pa"]), generates
    attributions via Exp 1's orchestrators and converts to feature rankings.

    Args:
        dnn_model: Trained DNN (PyTorch).
        rf_model: Trained RandomForest.
        dnn_wrapper: NNWrapper with predict_proba.
        rf_wrapper: SKLearnWrapper with predict_proba.
        dataset: DatasetBundle from Exp 1's data_loader.
        device: torch device.
        config: Full ExperimentConfig.

    Returns:
        List of FeatureSelectionResult (one per XAI method x mode).
    """
    logger.info("=== XAI-Driven Feature Selection Pipeline ===")
    logger.info(f"  Modes: {config.xai_modes}")

    all_results: list[FeatureSelectionResult] = []
    checkpoint_dir = config.output_dir / dataset.dataset_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for mode in config.xai_modes:
        mode_key = "normal" if mode in ("n", "normal") else "pa"
        try:
            results = _run_single_mode(
                mode_key,
                dnn_model, rf_model, dnn_wrapper, rf_wrapper,
                dataset, device, config,
                checkpoint_dir=str(checkpoint_dir) if mode_key == "pa" else None,
            )
            all_results.extend(results)
        except Exception as e:
            logger.error(f"XAI pipeline ({mode_key} mode) failed: {e}", exc_info=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for r in all_results:
        logger.info(
            f"  {r.method_name}: {r.n_original} -> {r.n_selected} features "
            f"in {r.selection_time_s:.2f}s"
        )

    return all_results
