"""Mathematical robustness metrics for XAI explanations under adversarial perturbation.

Implements three formal robustness notions from the roadmap:
1. Lipschitz Continuity (L): ||phi(x1) - phi(x2)|| <= L_c * ||x1 - x2||
2. Explanation Similarity (Delta_sim): max deviation within local neighborhood
3. Classification Equivalence (Delta_class): robustness within same-class subspaces
"""

import logging
import time

import numpy as np
from scipy import stats

import importlib.util as _ilu
import os as _os
_spec = _ilu.spec_from_file_location(
    "exp2_config", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "config.py")
)
_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
RobustnessConfig = _cfg.RobustnessConfig

logger = logging.getLogger(__name__)


def compute_lipschitz_constants(
    explain_fn,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    norm: str = "l2",
) -> dict:
    """Compute empirical Lipschitz constants between clean and adversarial explanations.

    Lipschitz Continuity: ||phi(x) - phi(x_adv)|| <= L_c * ||x - x_adv||

    The empirical Lipschitz constant is the maximum ratio of explanation
    distance to input distance across all samples.

    Args:
        explain_fn: callable(X) -> attributions array (n, features)
        X_clean: clean input samples
        X_adv: adversarial input samples
        norm: "l2" or "linf" distance metric

    Returns:
        dict with max/mean/median/std Lipschitz constants and per-sample ratios.
    """
    logger.info("    Computing explanations for clean samples...")
    attr_clean = explain_fn(X_clean)
    logger.info("    Computing explanations for adversarial samples...")
    attr_adv = explain_fn(X_adv)

    if attr_clean is None or attr_adv is None:
        return {"error": "explain_fn returned None"}

    # Compute distances
    if norm == "l2":
        input_dist = np.linalg.norm(X_clean - X_adv, axis=1)
        expl_dist = np.linalg.norm(attr_clean - attr_adv, axis=1)
    elif norm == "linf":
        input_dist = np.max(np.abs(X_clean - X_adv), axis=1)
        expl_dist = np.max(np.abs(attr_clean - attr_adv), axis=1)
    else:
        raise ValueError(f"Unknown norm: {norm}")

    # Avoid division by zero
    valid = input_dist > 1e-10
    if valid.sum() == 0:
        return {"error": "all input distances are zero"}

    ratios = np.full(len(X_clean), np.nan)
    ratios[valid] = expl_dist[valid] / input_dist[valid]
    valid_ratios = ratios[valid]

    return {
        "lipschitz_max": float(np.max(valid_ratios)),
        "lipschitz_mean": float(np.mean(valid_ratios)),
        "lipschitz_median": float(np.median(valid_ratios)),
        "lipschitz_std": float(np.std(valid_ratios)),
        "lipschitz_p95": float(np.percentile(valid_ratios, 95)),
        "lipschitz_p99": float(np.percentile(valid_ratios, 99)),
        "input_dist_mean": float(np.mean(input_dist[valid])),
        "expl_dist_mean": float(np.mean(expl_dist[valid])),
        "num_valid": int(valid.sum()),
        "num_total": len(X_clean),
        "norm": norm,
    }


def compute_explanation_similarity(
    explain_fn,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    epsilon: float = 0.1,
    norm: str = "l2",
) -> dict:
    """Evaluate Explanation Similarity (Delta_sim).

    Tests whether the maximum deviation between benign and adversarial
    explanations is bounded by epsilon within a local neighborhood.

    ExplSim: exists gamma > 0 such that d_E(h(x), gamma * h(x_adv)) <= epsilon

    Args:
        explain_fn: callable(X) -> attributions
        X_clean: clean samples
        X_adv: adversarial samples
        epsilon: similarity threshold
        norm: distance metric

    Returns:
        dict with similarity metrics and satisfaction rates.
    """
    attr_clean = explain_fn(X_clean)
    attr_adv = explain_fn(X_adv)

    if attr_clean is None or attr_adv is None:
        return {"error": "explain_fn returned None"}

    # Normalize attributions to unit norm for scale-invariant comparison
    clean_norms = np.linalg.norm(attr_clean, axis=1, keepdims=True)
    adv_norms = np.linalg.norm(attr_adv, axis=1, keepdims=True)

    # Avoid division by zero
    clean_norms = np.maximum(clean_norms, 1e-10)
    adv_norms = np.maximum(adv_norms, 1e-10)

    attr_clean_norm = attr_clean / clean_norms
    attr_adv_norm = attr_adv / adv_norms

    # Compute distances between normalized explanations
    if norm == "l2":
        distances = np.linalg.norm(attr_clean_norm - attr_adv_norm, axis=1)
    else:
        distances = np.max(np.abs(attr_clean_norm - attr_adv_norm), axis=1)

    # Check similarity satisfaction
    satisfied = distances <= epsilon
    satisfaction_rate = float(satisfied.mean())

    # Cosine similarity between explanations
    cos_sim = np.sum(attr_clean_norm * attr_adv_norm, axis=1)

    # Top-k feature overlap (Jaccard of top-5 features)
    k = min(5, attr_clean.shape[1])
    top_k_clean = np.argsort(-np.abs(attr_clean), axis=1)[:, :k]
    top_k_adv = np.argsort(-np.abs(attr_adv), axis=1)[:, :k]

    jaccard_scores = []
    for i in range(len(X_clean)):
        intersection = len(set(top_k_clean[i]) & set(top_k_adv[i]))
        union = len(set(top_k_clean[i]) | set(top_k_adv[i]))
        jaccard_scores.append(intersection / union if union > 0 else 0.0)

    # Spearman rank correlation of attributions
    rank_correlations = []
    for i in range(len(X_clean)):
        if np.std(attr_clean[i]) > 1e-10 and np.std(attr_adv[i]) > 1e-10:
            corr, _ = stats.spearmanr(attr_clean[i], attr_adv[i])
            rank_correlations.append(corr)
        else:
            rank_correlations.append(0.0)

    return {
        "similarity_satisfaction_rate": satisfaction_rate,
        "similarity_epsilon": epsilon,
        "distance_mean": float(np.mean(distances)),
        "distance_std": float(np.std(distances)),
        "distance_max": float(np.max(distances)),
        "cosine_similarity_mean": float(np.mean(cos_sim)),
        "cosine_similarity_std": float(np.std(cos_sim)),
        "top_k_jaccard_mean": float(np.mean(jaccard_scores)),
        "top_k_jaccard_std": float(np.std(jaccard_scores)),
        "rank_correlation_mean": float(np.mean(rank_correlations)),
        "rank_correlation_std": float(np.std(rank_correlations)),
        "norm": norm,
    }


def compute_classification_equivalence(
    predict_fn,
    explain_fn,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    norm: str = "l2",
) -> dict:
    """Evaluate Classification Equivalence (Delta_class).

    Assesses robustness exclusively within subspaces where the underlying
    classification outcome remains identical (Prediction-Preserving attacks).

    ClsEq: F(x) = F(x_adv) => explanations should be similar

    Only considers samples where the adversarial perturbation did NOT change
    the model's prediction. For these prediction-preserving cases, measures
    how much the explanation shifted.

    Args:
        predict_fn: callable(X) -> predictions array
        explain_fn: callable(X) -> attributions array
        X_clean: clean samples
        X_adv: adversarial samples

    Returns:
        dict with classification equivalence metrics.
    """
    # Get predictions
    if hasattr(predict_fn, '__call__'):
        preds_clean = np.argmax(predict_fn(X_clean), axis=1)
        preds_adv = np.argmax(predict_fn(X_adv), axis=1)
    else:
        preds_clean = predict_fn.predict(X_clean)
        preds_adv = predict_fn.predict(X_adv)

    # Find prediction-preserving samples
    same_pred = preds_clean == preds_adv
    num_same = int(same_pred.sum())
    num_diff = int((~same_pred).sum())

    logger.info(
        f"    Classification Equivalence: {num_same} prediction-preserving, "
        f"{num_diff} prediction-changing"
    )

    if num_same == 0:
        return {
            "num_prediction_preserving": 0,
            "num_prediction_changing": num_diff,
            "error": "no prediction-preserving samples found",
        }

    # Compute explanations only for prediction-preserving samples
    X_clean_pp = X_clean[same_pred]
    X_adv_pp = X_adv[same_pred]

    attr_clean = explain_fn(X_clean_pp)
    attr_adv = explain_fn(X_adv_pp)

    if attr_clean is None or attr_adv is None:
        return {"error": "explain_fn returned None for PP samples"}

    # Explanation distances for prediction-preserving samples
    if norm == "l2":
        expl_dist = np.linalg.norm(attr_clean - attr_adv, axis=1)
    else:
        expl_dist = np.max(np.abs(attr_clean - attr_adv), axis=1)

    # Normalize by attribution magnitude
    attr_magnitude = np.linalg.norm(attr_clean, axis=1)
    relative_dist = expl_dist / np.maximum(attr_magnitude, 1e-10)

    # Top-k feature stability
    k = min(5, attr_clean.shape[1])
    top_k_clean = np.argsort(-np.abs(attr_clean), axis=1)[:, :k]
    top_k_adv = np.argsort(-np.abs(attr_adv), axis=1)[:, :k]

    jaccard_scores = []
    for i in range(num_same):
        intersection = len(set(top_k_clean[i]) & set(top_k_adv[i]))
        union = len(set(top_k_clean[i]) | set(top_k_adv[i]))
        jaccard_scores.append(intersection / union if union > 0 else 0.0)

    # Rank correlation for PP samples
    rank_correlations = []
    for i in range(num_same):
        if np.std(attr_clean[i]) > 1e-10 and np.std(attr_adv[i]) > 1e-10:
            corr, _ = stats.spearmanr(attr_clean[i], attr_adv[i])
            rank_correlations.append(corr)
        else:
            rank_correlations.append(0.0)

    return {
        "num_prediction_preserving": num_same,
        "num_prediction_changing": num_diff,
        "prediction_preserving_rate": float(num_same / len(X_clean)),
        "expl_distance_mean": float(np.mean(expl_dist)),
        "expl_distance_std": float(np.std(expl_dist)),
        "expl_distance_max": float(np.max(expl_dist)),
        "relative_distance_mean": float(np.mean(relative_dist)),
        "top_k_jaccard_mean": float(np.mean(jaccard_scores)),
        "top_k_jaccard_std": float(np.std(jaccard_scores)),
        "rank_correlation_mean": float(np.mean(rank_correlations)),
        "rank_correlation_std": float(np.std(rank_correlations)),
        "norm": norm,
    }


def evaluate_robustness_for_method(
    method_name: str,
    explain_fn,
    predict_fn,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    attack_name: str,
    epsilon: float,
    config: RobustnessConfig,
) -> dict:
    """Evaluate all three robustness metrics for one XAI method + one attack config.

    Returns combined metrics dict.
    """
    logger.info(
        f"  Evaluating robustness: {method_name} vs {attack_name} (eps={epsilon})"
    )

    result = {
        "method": method_name,
        "attack": attack_name,
        "epsilon": epsilon,
    }

    # 1. Lipschitz Continuity
    start = time.time()
    lip = compute_lipschitz_constants(
        explain_fn, X_clean, X_adv, norm=config.distance_norm
    )
    result["lipschitz"] = lip
    logger.info(
        f"    Lipschitz: max={lip.get('lipschitz_max', 'N/A'):.4f}, "
        f"mean={lip.get('lipschitz_mean', 'N/A'):.4f} "
        f"({time.time() - start:.1f}s)"
    )

    # 2. Explanation Similarity
    start = time.time()
    sim = compute_explanation_similarity(
        explain_fn, X_clean, X_adv,
        epsilon=config.explanation_similarity_epsilon,
        norm=config.distance_norm,
    )
    result["similarity"] = sim
    logger.info(
        f"    ExplSim: satisfaction={sim.get('similarity_satisfaction_rate', 'N/A'):.4f}, "
        f"cos_sim={sim.get('cosine_similarity_mean', 'N/A'):.4f} "
        f"({time.time() - start:.1f}s)"
    )

    # 3. Classification Equivalence
    start = time.time()
    cls_eq = compute_classification_equivalence(
        predict_fn, explain_fn, X_clean, X_adv, norm=config.distance_norm
    )
    result["classification_equivalence"] = cls_eq
    logger.info(
        f"    ClsEq: PP_rate={cls_eq.get('prediction_preserving_rate', 'N/A'):.4f}, "
        f"top_k_jaccard={cls_eq.get('top_k_jaccard_mean', 'N/A'):.4f} "
        f"({time.time() - start:.1f}s)"
    )

    return result
