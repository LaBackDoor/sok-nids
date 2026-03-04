"""Evaluation metrics for XAI explanation quality.

Metrics implemented:
1. Faithfulness (Descriptive Accuracy) - Feature ablation + Pearson correlation
2. Sparsity - Fraction of near-zero attributions
3. Complexity - Shannon entropy of normalized attributions
4. Efficiency - Timing (captured during explanation generation)
5. Stability - Jaccard similarity across repeated runs
6. Completeness - Success rate on corrupted inputs
7. Robustness - Explanation variance under input perturbation
"""

import logging
import time

import numpy as np
from scipy import stats

from config import MetricConfig

logger = logging.getLogger(__name__)


def faithfulness(
    predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    attributions: np.ndarray,
    k_values: list[int],
    num_classes: int,
) -> dict:
    """Descriptive Accuracy: ablate top-k features, measure prediction degradation.

    Also computes faithfulness correlation (Pearson) between importance scores
    and the change in prediction probability when each feature is masked.
    """
    results = {}
    from sklearn.metrics import f1_score, roc_auc_score

    # Baseline performance
    y_proba_base = predict_fn(X)
    y_pred_base = np.argmax(y_proba_base, axis=1)
    base_f1 = f1_score(y, y_pred_base, average="weighted", zero_division=0)
    try:
        if num_classes == 2:
            base_auc = roc_auc_score(y, y_proba_base[:, 1])
        else:
            base_auc = roc_auc_score(y, y_proba_base, multi_class="ovr", average="weighted")
    except ValueError:
        base_auc = float("nan")

    results["baseline_f1"] = base_f1
    results["baseline_auc"] = base_auc

    # Feature importance: mean absolute attribution per feature
    mean_importance = np.mean(np.abs(attributions), axis=0)

    for k in k_values:
        k = min(k, X.shape[1])
        top_k_indices = np.argsort(mean_importance)[::-1][:k]

        # Mask top-k features with column mean (baseline reference)
        X_masked = X.copy()
        for idx in top_k_indices:
            X_masked[:, idx] = np.mean(X[:, idx])

        y_proba_masked = predict_fn(X_masked)
        y_pred_masked = np.argmax(y_proba_masked, axis=1)
        masked_f1 = f1_score(y, y_pred_masked, average="weighted", zero_division=0)
        try:
            if num_classes == 2:
                masked_auc = roc_auc_score(y, y_proba_masked[:, 1])
            else:
                masked_auc = roc_auc_score(
                    y, y_proba_masked, multi_class="ovr", average="weighted"
                )
        except ValueError:
            masked_auc = float("nan")

        results[f"f1_drop_k{k}"] = base_f1 - masked_f1
        results[f"auc_drop_k{k}"] = base_auc - masked_auc if not np.isnan(base_auc) else float("nan")
        results[f"masked_f1_k{k}"] = masked_f1
        results[f"masked_auc_k{k}"] = masked_auc

    # Faithfulness correlation: per-feature Pearson between importance and prob drop
    n_features = X.shape[1]
    importance_scores = mean_importance
    prob_drops = np.zeros(n_features)
    for j in range(n_features):
        X_single_mask = X.copy()
        X_single_mask[:, j] = np.mean(X[:, j])
        y_proba_j = predict_fn(X_single_mask)
        # Mean prediction probability drop for the true class
        base_probs = y_proba_base[np.arange(len(y)), y]
        masked_probs = y_proba_j[np.arange(len(y)), y]
        prob_drops[j] = np.mean(base_probs - masked_probs)

    corr, p_value = stats.pearsonr(importance_scores, prob_drops)
    results["faithfulness_correlation"] = corr
    results["faithfulness_p_value"] = p_value

    return results


def sparsity(attributions: np.ndarray, thresholds: list[float]) -> dict:
    """Sparsity: fraction of feature importance scores below each threshold."""
    abs_attrs = np.abs(attributions)
    # Normalize to [0, 1] per sample
    max_vals = abs_attrs.max(axis=1, keepdims=True)
    max_vals[max_vals == 0] = 1.0
    norm_attrs = abs_attrs / max_vals

    results = {}
    for tau in thresholds:
        frac_below = np.mean(norm_attrs < tau)
        results[f"sparsity_tau_{tau:.1f}"] = frac_below

    # Overall sparsity: mean fraction below 0.1
    results["sparsity_mean"] = np.mean(norm_attrs < 0.1)
    return results


def complexity(attributions: np.ndarray) -> dict:
    """Complexity: Shannon entropy of normalized attribution vector."""
    abs_attrs = np.abs(attributions)
    # Normalize each sample to a probability distribution
    sums = abs_attrs.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    probs = abs_attrs / sums

    # Shannon entropy per sample
    entropies = []
    for p in probs:
        p_nonzero = p[p > 0]
        h = -np.sum(p_nonzero * np.log2(p_nonzero))
        entropies.append(h)

    entropies = np.array(entropies)
    max_entropy = np.log2(attributions.shape[1])  # Maximum possible entropy

    return {
        "complexity_mean_entropy": float(np.mean(entropies)),
        "complexity_std_entropy": float(np.std(entropies)),
        "complexity_normalized": float(np.mean(entropies) / max_entropy) if max_entropy > 0 else 0.0,
    }


def efficiency(explanation_result) -> dict:
    """Efficiency: computation time metrics (extracted from ExplanationResult)."""
    return {
        "time_per_sample_ms": explanation_result.time_per_sample_ms,
        "total_time_s": explanation_result.total_time_s,
        "samples_per_second": len(explanation_result.attributions) / explanation_result.total_time_s
        if explanation_result.total_time_s > 0
        else 0.0,
    }


def stability(
    explain_fn,
    X_single: np.ndarray,
    n_runs: int,
    top_k: int,
) -> dict:
    """Stability: Jaccard similarity of top-k features across repeated runs.

    Args:
        explain_fn: Callable that takes X (n, features) and returns attributions (n, features).
        X_single: Single sample repeated for stability test, shape (1, features).
        n_runs: Number of repeated explanation runs.
        top_k: Number of top features to compare.
    """
    all_top_k_sets = []
    n_degenerate = 0

    for run in range(n_runs):
        attrs = explain_fn(X_single)
        if attrs.ndim == 1:
            attrs = attrs.reshape(1, -1)
        # Detect degenerate attributions (all zeros or uniform values)
        if np.all(attrs[0] == attrs[0][0]):
            n_degenerate += 1
        top_indices = set(np.argsort(np.abs(attrs[0]))[::-1][:top_k])
        all_top_k_sets.append(top_indices)

    # If all runs produced degenerate attributions, stability is meaningless
    if n_degenerate == n_runs:
        return {
            "stability_jaccard_mean": float("nan"),
            "stability_jaccard_min": float("nan"),
            "stability_degenerate": True,
        }

    # Pairwise Jaccard similarity
    jaccard_scores = []
    for i in range(len(all_top_k_sets)):
        for j in range(i + 1, len(all_top_k_sets)):
            intersection = len(all_top_k_sets[i] & all_top_k_sets[j])
            union = len(all_top_k_sets[i] | all_top_k_sets[j])
            jaccard_scores.append(intersection / union if union > 0 else 1.0)

    return {
        "stability_jaccard_mean": float(np.mean(jaccard_scores)) if jaccard_scores else 1.0,
        "stability_jaccard_min": float(np.min(jaccard_scores)) if jaccard_scores else 1.0,
        "stability_degenerate": False,
    }


def completeness(
    predict_fn,
    explain_fn,
    X_test: np.ndarray,
    num_corrupted: int,
    rng: np.random.RandomState,
) -> dict:
    """Completeness: fraction of explanations that succeed on corrupted/edge-case inputs.

    Tests: zero vectors, extreme outliers, randomly corrupted samples, and
    the local accuracy axiom (sum of attributions ≈ f(x) - f(baseline)).
    """
    n_features = X_test.shape[1]
    total = 0
    successes = 0

    # Test 1: Zero vectors
    X_zeros = np.zeros((min(50, num_corrupted // 3), n_features), dtype=np.float32)
    total += len(X_zeros)
    try:
        attrs = explain_fn(X_zeros)
        if attrs is not None and not np.any(np.isnan(attrs)):
            successes += len(X_zeros)
    except Exception:
        pass

    # Test 2: Extreme outlier values
    X_extreme = rng.uniform(-100, 100, size=(min(50, num_corrupted // 3), n_features)).astype(np.float32)
    total += len(X_extreme)
    try:
        attrs = explain_fn(X_extreme)
        if attrs is not None and not np.any(np.isnan(attrs)):
            successes += len(X_extreme)
    except Exception:
        pass

    # Test 3: Randomly corrupted samples from test set
    n_corrupt = min(num_corrupted - total, len(X_test))
    if n_corrupt > 0:
        indices = rng.choice(len(X_test), size=n_corrupt, replace=False)
        X_corrupt = X_test[indices].copy()
        # Randomly set some features to extreme values
        mask = rng.random(X_corrupt.shape) < 0.3
        X_corrupt[mask] = rng.uniform(-10, 10, size=mask.sum()).astype(np.float32)
        total += len(X_corrupt)
        try:
            attrs = explain_fn(X_corrupt)
            if attrs is not None and not np.any(np.isnan(attrs)):
                successes += len(X_corrupt)
        except Exception:
            pass

    # Test 4: Local accuracy axiom check
    # Verify that the sum of attributions approximates the model output difference
    # (output - baseline_output). Tests on a subset of clean test samples.
    axiom_samples = min(50, len(X_test))
    axiom_indices = rng.choice(len(X_test), size=axiom_samples, replace=False)
    X_axiom = X_test[axiom_indices]
    try:
        attrs_axiom = explain_fn(X_axiom)
        if attrs_axiom is not None and not np.any(np.isnan(attrs_axiom)):
            attr_sums = np.sum(attrs_axiom, axis=1)

            # Compute f(x) - f(baseline) for the predicted class
            baseline = np.zeros((1, n_features), dtype=np.float32)
            model_preds = predict_fn(X_axiom)  # (n, classes)
            baseline_preds = predict_fn(baseline)  # (1, classes)
            predicted_classes = np.argmax(model_preds, axis=1)
            pred_diffs = (
                model_preds[np.arange(len(X_axiom)), predicted_classes]
                - baseline_preds[0, predicted_classes]
            )

            # Check if attribution sums approximate the prediction difference
            abs_errors = np.abs(attr_sums - pred_diffs)
            valid_axiom = bool(np.all(np.isclose(attr_sums, pred_diffs, atol=1e-2)))
            axiom_mae = float(np.mean(abs_errors))
            axiom_pass_rate = float(np.mean(np.isclose(attr_sums, pred_diffs, atol=1e-2)))
        else:
            valid_axiom = False
            axiom_mae = float("nan")
            axiom_pass_rate = 0.0
    except Exception:
        valid_axiom = False
        axiom_mae = float("nan")
        axiom_pass_rate = 0.0

    return {
        "completeness_success_rate": successes / total if total > 0 else 0.0,
        "completeness_total_tested": total,
        "completeness_successes": successes,
        "completeness_axiom_valid": valid_axiom,
        "completeness_axiom_mae": axiom_mae,
        "completeness_axiom_pass_rate": axiom_pass_rate,
    }


def robustness(
    explain_fn,
    X_samples: np.ndarray,
    noise_std: float,
    num_perturbations: int,
    rng: np.random.RandomState,
) -> dict:
    """Robustness: explanation variance under small Gaussian noise perturbations."""
    n_samples = min(50, len(X_samples))  # Limit for efficiency
    X_subset = X_samples[:n_samples]

    # Get base explanations
    base_attrs = explain_fn(X_subset)

    # Generate perturbed explanations
    all_deviations = []
    for _ in range(num_perturbations):
        noise = rng.normal(0, noise_std, size=X_subset.shape).astype(np.float32)
        X_perturbed = np.clip(X_subset + noise, 0, 1)
        perturbed_attrs = explain_fn(X_perturbed)

        # Normalized L2 deviation
        diff = perturbed_attrs - base_attrs
        norms = np.linalg.norm(diff, axis=1)
        base_norms = np.linalg.norm(base_attrs, axis=1)
        base_norms[base_norms == 0] = 1.0
        relative_deviation = norms / base_norms
        all_deviations.append(relative_deviation)

    deviations = np.concatenate(all_deviations)

    # Rank stability under perturbation
    rank_changes = []
    for _ in range(num_perturbations):
        noise = rng.normal(0, noise_std, size=X_subset.shape).astype(np.float32)
        X_perturbed = np.clip(X_subset + noise, 0, 1)
        perturbed_attrs = explain_fn(X_perturbed)

        for i in range(n_samples):
            base_rank = np.argsort(np.abs(base_attrs[i]))[::-1]
            pert_rank = np.argsort(np.abs(perturbed_attrs[i]))[::-1]
            # Spearman correlation of ranks
            corr, _ = stats.spearmanr(base_rank, pert_rank)
            rank_changes.append(corr)

    return {
        "robustness_mean_deviation": float(np.mean(deviations)),
        "robustness_std_deviation": float(np.std(deviations)),
        "robustness_max_deviation": float(np.max(deviations)),
        "robustness_rank_correlation_mean": float(np.mean(rank_changes)),
        "robustness_rank_correlation_std": float(np.std(rank_changes)),
    }


def evaluate_all_metrics(
    predict_fn,
    explain_fn,
    explanation_result,
    X_test: np.ndarray,
    y_test: np.ndarray,
    explain_indices: np.ndarray,
    num_classes: int,
    config: MetricConfig,
) -> dict:
    """Run all evaluation metrics for a single explanation method."""
    X_explain = X_test[explain_indices]
    y_explain = y_test[explain_indices]
    rng = np.random.RandomState(42)

    metrics = {
        "method": explanation_result.method_name,
        "model": explanation_result.model_name,
    }

    # 1. Faithfulness
    logger.info(f"    Computing Faithfulness...")
    faith = faithfulness(
        predict_fn, X_explain, y_explain,
        explanation_result.attributions, config.faithfulness_k_values, num_classes,
    )
    metrics.update(faith)

    # 2. Sparsity
    logger.info(f"    Computing Sparsity...")
    spar = sparsity(explanation_result.attributions, config.sparsity_thresholds)
    metrics.update(spar)

    # 3. Complexity
    logger.info(f"    Computing Complexity...")
    comp = complexity(explanation_result.attributions)
    metrics.update(comp)

    # 4. Efficiency
    logger.info(f"    Computing Efficiency...")
    eff = efficiency(explanation_result)
    metrics.update(eff)

    # 5. Stability (on a subset of samples)
    logger.info(f"    Computing Stability...")
    n_stability = min(20, len(X_explain))
    stability_scores = []
    for i in range(n_stability):
        x_single = X_explain[i : i + 1]
        stab = stability(
            explain_fn, x_single, config.stability_runs, config.stability_top_k,
        )
        stability_scores.append(stab["stability_jaccard_mean"])
    metrics["stability_jaccard_mean"] = float(np.mean(stability_scores))
    metrics["stability_jaccard_std"] = float(np.std(stability_scores))

    # 6. Completeness
    logger.info(f"    Computing Completeness...")
    comp_result = completeness(
        predict_fn, explain_fn, X_test, config.completeness_num_corrupted, rng,
    )
    metrics.update(comp_result)

    # 7. Robustness
    logger.info(f"    Computing Robustness...")
    rob = robustness(
        explain_fn, X_explain, config.robustness_noise_std,
        config.robustness_num_perturbations, rng,
    )
    metrics.update(rob)

    return metrics
