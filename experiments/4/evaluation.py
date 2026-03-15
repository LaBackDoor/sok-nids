"""Evaluation metrics for Experiment 4: downstream model benchmarking.

Metrics: Accuracy, F1-Score, FPR, FAR, training time, inference latency.
"""

import logging
import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_fpr_far(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """Compute False Positive Rate and False Alarm Rate.

    FPR: Rate at which benign traffic is flagged as malicious (per-class macro).
    FAR: Overall false alarm rate = FP / (FP + TN) across all classes.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Per-class FPR: FP_i / (FP_i + TN_i)
    fpr_per_class = []
    for i in range(num_classes):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fpr_i = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fpr_per_class.append(fpr_i)

    macro_fpr = np.mean(fpr_per_class)

    # Overall FAR: total FP / (total FP + total TN)
    total_fp = 0
    total_tn = 0
    for i in range(num_classes):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        total_fp += fp
        total_tn += tn
    overall_far = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0

    return {
        "fpr_per_class": [float(f) for f in fpr_per_class],
        "fpr_macro": float(macro_fpr),
        "far_overall": float(overall_far),
    }


def measure_inference_latency(
    wrapper,
    X_test: np.ndarray,
    n_runs: int = 5,
) -> dict:
    """Measure inference latency in ms per sample.

    Runs prediction multiple times and reports mean/std.
    """
    # Warmup run
    wrapper.predict(X_test[:min(100, len(X_test))])

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        wrapper.predict(X_test)
        elapsed = time.perf_counter() - t0
        ms_per_sample = (elapsed / len(X_test)) * 1000
        latencies.append(ms_per_sample)

    return {
        "inference_ms_per_sample_mean": float(np.mean(latencies)),
        "inference_ms_per_sample_std": float(np.std(latencies)),
        "total_inference_time_s": float(np.mean(latencies) * len(X_test) / 1000),
    }


def evaluate_downstream_model(
    wrapper,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    model_name: str,
    selection_method: str,
    n_features: int,
    train_time: float,
) -> dict:
    """Full evaluation of a downstream model."""
    logger.info(f"    Evaluating {model_name} ({selection_method}, {n_features} features)")

    y_pred = wrapper.predict(X_test)
    y_proba = wrapper.predict_proba(X_test)

    # Classification metrics
    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    # AUC-ROC
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except ValueError:
        auc = float("nan")

    # FPR and FAR
    fpr_far = compute_fpr_far(y_test, y_pred, num_classes)

    # Inference latency
    latency = measure_inference_latency(wrapper, X_test)

    metrics = {
        "model": model_name,
        "selection_method": selection_method,
        "n_features": n_features,
        "accuracy": float(acc),
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "f1_per_class": [float(f) for f in f1_per_class],
        "auc_roc": float(auc),
        "fpr_macro": fpr_far["fpr_macro"],
        "far_overall": fpr_far["far_overall"],
        "fpr_per_class": fpr_far["fpr_per_class"],
        "train_time_s": float(train_time),
        **latency,
    }

    logger.info(
        f"      Acc={acc:.4f} F1w={f1_weighted:.4f} F1m={f1_macro:.4f} "
        f"AUC={auc:.4f} FPR={fpr_far['fpr_macro']:.4f} "
        f"Latency={latency['inference_ms_per_sample_mean']:.4f}ms/sample"
    )

    return metrics


def compute_reduction_summary(
    full_feature_metrics: dict,
    reduced_metrics: dict,
    n_original_features: int,
) -> dict:
    """Compute percentage changes between full and reduced feature models."""
    n_reduced = reduced_metrics["n_features"]
    feature_reduction_pct = (1 - n_reduced / n_original_features) * 100

    acc_change = reduced_metrics["accuracy"] - full_feature_metrics["accuracy"]
    f1_change = reduced_metrics["f1_weighted"] - full_feature_metrics["f1_weighted"]
    fpr_change = reduced_metrics["fpr_macro"] - full_feature_metrics["fpr_macro"]

    train_speedup = (
        full_feature_metrics["train_time_s"] / reduced_metrics["train_time_s"]
        if reduced_metrics["train_time_s"] > 0 else float("inf")
    )
    latency_speedup = (
        full_feature_metrics["inference_ms_per_sample_mean"]
        / reduced_metrics["inference_ms_per_sample_mean"]
        if reduced_metrics["inference_ms_per_sample_mean"] > 0 else float("inf")
    )

    return {
        "feature_reduction_pct": float(feature_reduction_pct),
        "accuracy_change": float(acc_change),
        "f1_weighted_change": float(f1_change),
        "fpr_change": float(fpr_change),
        "train_time_speedup": float(train_speedup),
        "inference_latency_speedup": float(latency_speedup),
    }
