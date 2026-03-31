"""Evaluation metrics for XAI explanation quality.

Metrics implemented:
1. Faithfulness (Descriptive Accuracy) - Feature ablation + Pearson correlation
2. Sparsity - Fraction of near-zero attributions
3. Complexity - Shannon entropy of normalized attributions
4. Efficiency - Timing (captured during explanation generation)
5. Stability - Jaccard similarity across repeated runs
6. Completeness - Success rate on corrupted inputs
"""

import logging
import time

import numpy as np
from scipy import stats

from config import MetricConfig

logger = logging.getLogger(__name__)


def _safe_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
    context: str = "",
) -> float:
    """Compute ROC AUC only over classes that have both positive and negative samples.

    sklearn's OVR mode warns when a class has only one value in y_true.
    This computes per-class AUC only for valid classes and returns the
    weighted average, avoiding the UndefinedMetricWarning entirely.
    """
    from sklearn.metrics import roc_auc_score

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        logger.warning(f"AUC requires at least 2 classes in y ({context}); skipping.")
        return float("nan")

    try:
        if num_classes == 2:
            return roc_auc_score(y_true, y_proba[:, 1])

        # Manual OVR: compute per-class AUC only for classes present in y_true
        per_class_auc = []
        per_class_weight = []
        for c in range(num_classes):
            binary_true = (y_true == c).astype(int)
            n_pos = binary_true.sum()
            n_neg = len(binary_true) - n_pos
            if n_pos == 0 or n_neg == 0:
                continue
            auc_c = roc_auc_score(binary_true, y_proba[:, c])
            per_class_auc.append(auc_c)
            per_class_weight.append(n_pos)

        if not per_class_auc:
            return float("nan")
        weights = np.array(per_class_weight, dtype=float)
        return float(np.average(per_class_auc, weights=weights))
    except ValueError as e:
        logger.warning(f"AUC computation failed ({context}): {e}")
        return float("nan")


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
    base_auc = _safe_roc_auc(y, y_proba_base, num_classes, "baseline")

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
        masked_auc = _safe_roc_auc(y, y_proba_masked, num_classes, f"masked k={k}")

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


def stability_batched(
    explain_fn,
    X_samples: np.ndarray,
    n_runs: int,
    top_k: int,
) -> list[dict]:
    """Stability: Jaccard similarity of top-k features across repeated runs.

    Batches all samples × runs into a single explain_fn call to avoid
    per-sample overhead (parallel-backend spin-up, etc.).

    Args:
        explain_fn: Callable that takes X (n, features) and returns attributions (n, features).
        X_samples: Samples to test stability on, shape (n_samples, features).
        n_runs: Number of repeated explanation runs.
        top_k: Number of top features to compare.

    Returns:
        List of per-sample stability dicts (one per row in X_samples).
    """
    n_samples = len(X_samples)
    # Build batch: each sample repeated n_runs times, grouped by run
    # Layout: [run0_sample0, run0_sample1, ..., run1_sample0, run1_sample1, ...]
    X_batch = np.tile(X_samples, (n_runs, 1))  # (n_runs * n_samples, features)

    # Single batched call
    attrs_all = explain_fn(X_batch)
    if attrs_all.ndim == 1:
        attrs_all = attrs_all.reshape(-1, X_samples.shape[-1])
    # Reshape to (n_runs, n_samples, features)
    attrs_per_run = attrs_all.reshape(n_runs, n_samples, -1)

    results = []
    for s in range(n_samples):
        all_top_k_sets = []
        n_degenerate = 0
        for r in range(n_runs):
            a = attrs_per_run[r, s]
            if np.all(a == a[0]):
                n_degenerate += 1
            top_indices = set(np.argsort(np.abs(a))[::-1][:top_k])
            all_top_k_sets.append(top_indices)

        if n_degenerate == n_runs:
            results.append({
                "stability_jaccard_mean": float("nan"),
                "stability_jaccard_min": float("nan"),
                "stability_degenerate": True,
            })
            continue

        jaccard_scores = []
        for i in range(len(all_top_k_sets)):
            for j in range(i + 1, len(all_top_k_sets)):
                intersection = len(all_top_k_sets[i] & all_top_k_sets[j])
                union = len(all_top_k_sets[i] | all_top_k_sets[j])
                jaccard_scores.append(intersection / union if union > 0 else 1.0)

        results.append({
            "stability_jaccard_mean": float(np.mean(jaccard_scores)) if jaccard_scores else 1.0,
            "stability_jaccard_min": float(np.min(jaccard_scores)) if jaccard_scores else 1.0,
            "stability_degenerate": False,
        })

    return results


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

    # 5. Stability (on a subset of samples, batched)
    logger.info(f"    Computing Stability...")
    n_stability = min(20, len(X_explain))
    stab_results = stability_batched(
        explain_fn, X_explain[:n_stability], config.stability_runs, config.stability_top_k,
    )
    stability_scores = [s["stability_jaccard_mean"] for s in stab_results]
    metrics["stability_jaccard_mean"] = float(np.nanmean(stability_scores))
    metrics["stability_jaccard_std"] = float(np.nanstd(stability_scores))
    n_degenerate = sum(1 for s in stab_results if s.get("stability_degenerate", False))
    if n_degenerate > 0:
        logger.warning(f"    {n_degenerate}/{n_stability} samples had degenerate attributions (stability=nan)")

    # 6. Completeness
    logger.info(f"    Computing Completeness...")
    comp_result = completeness(
        predict_fn, explain_fn, X_test, config.completeness_num_corrupted, rng,
    )
    metrics.update(comp_result)

    return metrics
