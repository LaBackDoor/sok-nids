"""XAI-driven feature selection pipeline for Experiment 4.

Trains baseline DNN + RF on full feature space, then uses SHAP and LIME
to rank features and iteratively prune to an optimal subset.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
import shap
import torch
import torch.nn as nn
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier

from config import XAISelectionConfig
from feature_selection import FeatureSelectionResult

logger = logging.getLogger(__name__)


def _get_shap_feature_importance_dnn(
    model: nn.Module,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    device: torch.device,
    config: XAISelectionConfig,
) -> np.ndarray:
    """Compute mean absolute SHAP values for DNN using DeepExplainer."""
    logger.info("    Computing SHAP values for DNN (DeepExplainer)...")

    base_model = model.module if isinstance(model, nn.DataParallel) else model
    base_model.eval()
    base_model = base_model.to(device)

    # Background samples
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X_train), min(config.shap_background_samples, len(X_train)), replace=False)
    background = torch.tensor(X_train[bg_idx], dtype=torch.float32).to(device)

    explainer = shap.DeepExplainer(base_model, background)

    # Explain in batches to manage memory
    batch_size = 500
    all_shap_values = []
    for i in range(0, len(X_explain), batch_size):
        batch = torch.tensor(X_explain[i:i + batch_size], dtype=torch.float32).to(device)
        sv = explainer.shap_values(batch)
        # sv is list of arrays (one per class) or single array
        if isinstance(sv, list):
            # Average absolute SHAP across classes
            sv_abs = np.mean([np.abs(s) for s in sv], axis=0)
        else:
            sv_abs = np.abs(sv)
        all_shap_values.append(sv_abs)

    shap_values = np.concatenate(all_shap_values, axis=0)
    # Mean absolute SHAP value per feature across all samples
    feature_importance = np.mean(shap_values, axis=0)
    return feature_importance


def _get_shap_feature_importance_rf(
    model: RandomForestClassifier,
    X_explain: np.ndarray,
    config: XAISelectionConfig,
) -> np.ndarray:
    """Compute mean absolute SHAP values for RF using TreeExplainer."""
    logger.info("    Computing SHAP values for RF (TreeExplainer)...")

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_explain)

    if isinstance(sv, list):
        # Multi-class: average absolute SHAP across classes
        feature_importance = np.mean([np.mean(np.abs(s), axis=0) for s in sv], axis=0)
    else:
        feature_importance = np.mean(np.abs(sv), axis=0)

    return feature_importance


def _get_lime_feature_importance(
    predict_fn,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    num_classes: int,
    config: XAISelectionConfig,
) -> np.ndarray:
    """Compute global feature importance via LIME."""
    logger.info("    Computing LIME feature importance...")

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=[str(i) for i in range(num_classes)],
        discretize_continuous=True,
        random_state=42,
    )

    n_features = X_train.shape[1]
    importance_accumulator = np.zeros(n_features)

    # Subsample for LIME (it's slow per-instance)
    n_explain = min(500, len(X_explain))
    rng = np.random.RandomState(42)
    explain_idx = rng.choice(len(X_explain), n_explain, replace=False)

    for i, idx in enumerate(explain_idx):
        if (i + 1) % 100 == 0:
            logger.info(f"      LIME explanation {i + 1}/{n_explain}")
        exp = explainer.explain_instance(
            X_explain[idx],
            predict_fn,
            num_features=n_features,
            num_samples=config.lime_num_samples,
        )
        for feat_idx, weight in exp.local_exp[exp.top_labels[0]]:
            importance_accumulator[feat_idx] += abs(weight)

    feature_importance = importance_accumulator / n_explain
    return feature_importance


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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    logger.info("    Starting iterative pruning...")

    n_features = len(feature_importance)
    ranked_indices = np.argsort(feature_importance)[::-1]  # descending importance

    # Train baseline RF on all features for F1 reference
    rf_baseline = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf_baseline.fit(X_train, y_train)
    y_pred_baseline = rf_baseline.predict(X_val)
    baseline_f1 = f1_score(y_val, y_pred_baseline, average="weighted", zero_division=0)
    logger.info(f"    Baseline F1 (all {n_features} features): {baseline_f1:.4f}")

    # Target from roadmap
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

    while len(current_indices) > min_n:
        # Remove pruning_step_ratio of remaining features
        n_remove = max(1, int(len(current_indices) * config.pruning_step_ratio))
        n_keep = max(min_n, len(current_indices) - n_remove)

        # Keep top-ranked features
        # Re-rank current indices by their original importance
        current_importance = feature_importance[current_indices]
        reranked = np.argsort(current_importance)[::-1]
        current_indices = current_indices[reranked[:n_keep]]

        # Evaluate with reduced features
        rf_reduced = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
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


def xai_shap_dnn_selection(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    device: torch.device,
    config: XAISelectionConfig,
    dataset_name: str,
) -> FeatureSelectionResult:
    """Feature selection using SHAP on DNN with iterative pruning."""
    logger.info("  XAI-SHAP (DNN) feature selection")
    t0 = time.time()

    # Subsample for SHAP explanations
    rng = np.random.RandomState(42)
    n_explain = min(config.shap_explain_samples, len(X_train))
    explain_idx = rng.choice(len(X_train), n_explain, replace=False)
    X_explain = X_train[explain_idx]

    importance = _get_shap_feature_importance_dnn(model, X_train, X_explain, device, config)

    selected_indices, pruning_history = _iterative_prune(
        importance, X_train, y_train, X_val, y_val,
        feature_names, config, dataset_name,
    )

    elapsed = time.time() - t0
    return FeatureSelectionResult(
        method_name="SHAP-DNN",
        selected_indices=selected_indices,
        feature_rankings=importance,
        selected_feature_names=[feature_names[i] for i in selected_indices],
        n_original=len(feature_names),
        n_selected=len(selected_indices),
        selection_time_s=elapsed,
    )


def xai_shap_rf_selection(
    model: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    config: XAISelectionConfig,
    dataset_name: str,
) -> FeatureSelectionResult:
    """Feature selection using SHAP on RF with iterative pruning."""
    logger.info("  XAI-SHAP (RF) feature selection")
    t0 = time.time()

    rng = np.random.RandomState(42)
    n_explain = min(config.shap_explain_samples, len(X_train))
    explain_idx = rng.choice(len(X_train), n_explain, replace=False)
    X_explain = X_train[explain_idx]

    importance = _get_shap_feature_importance_rf(model, X_explain, config)

    selected_indices, pruning_history = _iterative_prune(
        importance, X_train, y_train, X_val, y_val,
        feature_names, config, dataset_name,
    )

    elapsed = time.time() - t0
    return FeatureSelectionResult(
        method_name="SHAP-RF",
        selected_indices=selected_indices,
        feature_rankings=importance,
        selected_feature_names=[feature_names[i] for i in selected_indices],
        n_original=len(feature_names),
        n_selected=len(selected_indices),
        selection_time_s=elapsed,
    )


def xai_lime_selection(
    predict_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    num_classes: int,
    config: XAISelectionConfig,
    dataset_name: str,
    model_name: str = "DNN",
) -> FeatureSelectionResult:
    """Feature selection using LIME with iterative pruning."""
    logger.info(f"  XAI-LIME ({model_name}) feature selection")
    t0 = time.time()

    rng = np.random.RandomState(42)
    n_explain = min(config.shap_explain_samples, len(X_train))
    explain_idx = rng.choice(len(X_train), n_explain, replace=False)
    X_explain = X_train[explain_idx]

    importance = _get_lime_feature_importance(
        predict_fn, X_train, X_explain, feature_names, num_classes, config,
    )

    selected_indices, pruning_history = _iterative_prune(
        importance, X_train, y_train, X_val, y_val,
        feature_names, config, dataset_name,
    )

    elapsed = time.time() - t0
    return FeatureSelectionResult(
        method_name=f"LIME-{model_name}",
        selected_indices=selected_indices,
        feature_rankings=importance,
        selected_feature_names=[feature_names[i] for i in selected_indices],
        n_original=len(feature_names),
        n_selected=len(selected_indices),
        selection_time_s=elapsed,
    )


def run_xai_pipeline(
    dnn_model: nn.Module,
    rf_model: RandomForestClassifier,
    dnn_predict_fn,
    rf_predict_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    num_classes: int,
    device: torch.device,
    config: XAISelectionConfig,
    dataset_name: str,
) -> list[FeatureSelectionResult]:
    """Run all XAI-driven feature selection methods."""
    logger.info(f"=== XAI-Driven Feature Selection Pipeline ===")

    results = []

    # SHAP on DNN
    results.append(xai_shap_dnn_selection(
        dnn_model, X_train, y_train, X_val, y_val,
        feature_names, device, config, dataset_name,
    ))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # SHAP on RF
    results.append(xai_shap_rf_selection(
        rf_model, X_train, y_train, X_val, y_val,
        feature_names, config, dataset_name,
    ))

    # LIME on DNN
    results.append(xai_lime_selection(
        dnn_predict_fn, X_train, y_train, X_val, y_val,
        feature_names, num_classes, config, dataset_name, model_name="DNN",
    ))

    # LIME on RF
    results.append(xai_lime_selection(
        rf_predict_fn, X_train, y_train, X_val, y_val,
        feature_names, num_classes, config, dataset_name, model_name="RF",
    ))

    for r in results:
        logger.info(f"  {r.method_name}: {r.n_original} -> {r.n_selected} features in {r.selection_time_s:.2f}s")

    return results
