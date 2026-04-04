"""Feature interaction analysis using SHAP interaction values.

Quantifies two-way feature dependencies that perturbation-based methods
(like LIME) miss due to their independence assumption.
"""

import logging
import time

import numpy as np
import torch

from config import InteractionConfig

logger = logging.getLogger(__name__)


def compute_shap_interaction_values_rf(
    rf_model,
    X_samples: np.ndarray,
    config: InteractionConfig,
) -> np.ndarray:
    """Compute SHAP interaction values for Random Forest using TreeExplainer.

    TreeExplainer natively supports interaction values efficiently.

    Args:
        rf_model: Trained sklearn RandomForestClassifier.
        X_samples: Input samples, shape (n_samples, n_features).
        config: Interaction configuration.

    Returns:
        Interaction matrix, shape (n_samples, n_features, n_features) for predicted class.
    """
    import shap

    n = min(config.shap_interaction_samples, len(X_samples))
    X = X_samples[:n]

    logger.info(f"  Computing SHAP interaction values (RF) on {n} samples...")
    start = time.time()

    explainer = shap.TreeExplainer(rf_model)
    interaction_values = explainer.shap_interaction_values(X)

    elapsed = time.time() - start
    logger.info(f"  RF interaction values computed in {elapsed:.1f}s")

    # interaction_values is list of (n, f, f) per class or (n, f, f) directly
    preds = rf_model.predict(X)

    if isinstance(interaction_values, list):
        # List of arrays per class — select predicted class for each sample
        stacked = np.stack(interaction_values, axis=0)  # (classes, n, f, f)
        result = np.zeros((n, X.shape[1], X.shape[1]), dtype=np.float32)
        for i, pred in enumerate(preds):
            result[i] = stacked[pred, i]
    elif isinstance(interaction_values, np.ndarray) and interaction_values.ndim == 4:
        # (n, f, f, classes)
        result = np.zeros((n, X.shape[1], X.shape[1]), dtype=np.float32)
        for i, pred in enumerate(preds):
            result[i] = interaction_values[i, :, :, pred]
    else:
        result = np.asarray(interaction_values, dtype=np.float32)

    return result


def compute_shap_interaction_values_dnn(
    dnn_model: torch.nn.Module,
    X_samples: np.ndarray,
    X_background: np.ndarray,
    device: torch.device,
    config: InteractionConfig,
) -> np.ndarray:
    """Approximate SHAP interaction values for DNN using batched perturbations.

    Instead of looping over top features one-by-one (20 separate SHAP calls),
    batch all perturbations into a single tensor and run one SHAP call.

    Args:
        dnn_model: Trained DNN model.
        X_samples: Input samples, shape (n_samples, n_features).
        X_background: Background reference samples.
        device: Torch device.
        config: Interaction configuration.

    Returns:
        Approximate interaction matrix, shape (n_samples, n_features, n_features).
    """
    import shap

    n = min(config.shap_interaction_samples, len(X_samples))
    n_features = X_samples.shape[1]
    X = X_samples[:n]

    logger.info(f"  Approximating DNN interaction values (batched) on {n} samples...")
    start = time.time()

    base_model = dnn_model.module if isinstance(dnn_model, torch.nn.DataParallel) else dnn_model
    base_model.eval()

    bg = X_background[:config.shap_background_samples]
    bg_tensor = torch.tensor(bg, dtype=torch.float32).to(device)
    explainer = shap.DeepExplainer(base_model, bg_tensor)

    # Get base SHAP values
    x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    base_shap = explainer.shap_values(x_tensor, check_additivity=False)

    with torch.no_grad():
        preds = torch.argmax(base_model(x_tensor), dim=1).cpu().numpy()

    # Extract base attributions for predicted class
    base_attrs = _extract_predicted_class_attrs(base_shap, preds, n, n_features)

    # Identify top features by mean importance
    mean_importance = np.mean(np.abs(base_attrs), axis=0)
    top_features = np.argsort(mean_importance)[::-1][:config.top_n_interactions]
    n_top = len(top_features)

    feature_means = np.mean(X, axis=0)

    # Batch all perturbations: create (n_top * n, n_features) tensor
    # Each block of n rows has feature j replaced with its mean
    X_batch = np.tile(X, (n_top, 1))  # (n_top * n, n_features)
    for i, j in enumerate(top_features):
        X_batch[i * n : (i + 1) * n, j] = feature_means[j]

    logger.info(f"  Batched perturbation tensor: {X_batch.shape} ({n_top} features x {n} samples)")

    # Single batched SHAP call
    x_batch_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
    pert_shap_all = explainer.shap_values(x_batch_tensor, check_additivity=False)

    # Tile predictions to match batch layout
    preds_tiled = np.tile(preds, n_top)
    pert_attrs_flat = _extract_predicted_class_attrs(pert_shap_all, preds_tiled, n_top * n, n_features)

    # Reshape to (n_top, n, n_features)
    pert_attrs = pert_attrs_flat.reshape(n_top, n, n_features)

    # Compute interaction matrix
    interaction_matrix = np.zeros((n, n_features, n_features), dtype=np.float32)
    for idx, j in enumerate(top_features):
        # Interaction effect: change in attribution for all features when j is perturbed
        interaction_matrix[:, :, j] = base_attrs - pert_attrs[idx]

    elapsed = time.time() - start
    logger.info(f"  DNN interaction approximation completed in {elapsed:.1f}s (batched)")

    return interaction_matrix


def _extract_predicted_class_attrs(
    shap_values,
    preds: np.ndarray,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    """Extract attributions for the predicted class from SHAP output."""
    if isinstance(shap_values, list):
        stacked = np.stack(shap_values, axis=0)  # (classes, n, features)
        attrs = np.zeros((n_samples, n_features), dtype=np.float32)
        for i, pred in enumerate(preds):
            attrs[i] = stacked[pred, i]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        attrs = np.zeros((n_samples, n_features), dtype=np.float32)
        for i, pred in enumerate(preds):
            attrs[i] = shap_values[i, :, pred]
    else:
        attrs = np.asarray(shap_values, dtype=np.float32)
    return attrs


def aggregate_interaction_matrix(
    interaction_values: np.ndarray,
) -> np.ndarray:
    """Aggregate per-sample interaction values into a single matrix.

    Args:
        interaction_values: Shape (n_samples, n_features, n_features).

    Returns:
        Mean absolute interaction matrix, shape (n_features, n_features).
    """
    return np.mean(np.abs(interaction_values), axis=0)


def get_top_interactions(
    interaction_matrix: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
) -> list[dict]:
    """Extract top feature pairs by interaction strength.

    Args:
        interaction_matrix: Mean absolute interaction, shape (n_features, n_features).
        feature_names: Feature name list.
        top_n: Number of top pairs to return.

    Returns:
        List of dicts with feature_a, feature_b, interaction_strength.
    """
    n = interaction_matrix.shape[0]

    # Zero out diagonal (self-interactions)
    mat = interaction_matrix.copy()
    np.fill_diagonal(mat, 0)

    # Get upper triangle indices (avoid duplicates)
    upper_indices = np.triu_indices(n, k=1)
    values = mat[upper_indices]

    # Sort by strength
    sorted_idx = np.argsort(values)[::-1][:top_n]

    results = []
    for idx in sorted_idx:
        i, j = upper_indices[0][idx], upper_indices[1][idx]
        results.append({
            "feature_a": feature_names[i] if i < len(feature_names) else str(i),
            "feature_b": feature_names[j] if j < len(feature_names) else str(j),
            "feature_a_idx": int(i),
            "feature_b_idx": int(j),
            "interaction_strength": float(values[idx]),
        })

    return results


def compare_interaction_vs_main_effects(
    interaction_matrix: np.ndarray,
    lime_attributions: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Compare SHAP interaction effects against LIME's isolated main effects.

    LIME assumes feature independence, so its attributions represent isolated
    main effects. SHAP interaction values capture cooperative terms that LIME misses.

    Args:
        interaction_matrix: Mean abs SHAP interaction, shape (n_features, n_features).
        lime_attributions: LIME attributions, shape (n_samples, n_features).
        feature_names: Feature name list.

    Returns:
        Dict with comparison metrics.
    """
    n_features = interaction_matrix.shape[0]

    # SHAP main effects = diagonal of interaction matrix (self-interaction)
    # But we zeroed diagonal, so use mean abs of off-diagonal as interaction strength
    off_diag_mask = ~np.eye(n_features, dtype=bool)
    total_interaction = np.sum(interaction_matrix[off_diag_mask])
    diag_sum = np.sum(np.diag(interaction_matrix))

    # LIME main effects
    lime_main = np.mean(np.abs(lime_attributions), axis=0)

    # Correlation between SHAP main effects and LIME effects
    shap_main = np.diag(interaction_matrix)
    from scipy import stats
    if np.std(shap_main) > 1e-12 and np.std(lime_main) > 1e-12:
        corr, p_val = stats.spearmanr(shap_main, lime_main)
    else:
        corr, p_val = 0.0, 1.0

    # Interaction-to-main-effect ratio (higher = more interactions LIME misses)
    ratio = total_interaction / (diag_sum + 1e-12)

    # Features with highest interaction effects that LIME underweights
    interaction_per_feature = np.sum(interaction_matrix, axis=1) - np.diag(interaction_matrix)
    lime_rank = np.argsort(lime_main)[::-1]
    interaction_rank = np.argsort(interaction_per_feature)[::-1]

    # Rank displacement: features that rank much higher by interaction than by LIME
    displacement = []
    for feat_idx in range(n_features):
        lime_pos = np.where(lime_rank == feat_idx)[0][0]
        inter_pos = np.where(interaction_rank == feat_idx)[0][0]
        if inter_pos < lime_pos:  # Higher interaction rank than LIME rank
            displacement.append({
                "feature": feature_names[feat_idx] if feat_idx < len(feature_names) else str(feat_idx),
                "lime_rank": int(lime_pos),
                "interaction_rank": int(inter_pos),
                "rank_gain": int(lime_pos - inter_pos),
            })

    displacement.sort(key=lambda x: x["rank_gain"], reverse=True)

    return {
        "main_effect_correlation": float(corr),
        "main_effect_correlation_p": float(p_val),
        "interaction_to_main_ratio": float(ratio),
        "total_interaction_strength": float(total_interaction),
        "features_underweighted_by_lime": displacement[:10],
    }
