"""Explainer consensus analysis: pairwise agreement metrics between XAI methods.

Metrics:
- Spearman's rank correlation coefficient (monotonic relationship)
- Kendall's tau-b (concordance of ranked pairs)
- Top-k feature intersection (overlap of most important features)
- Wilcoxon signed-rank test (statistical significance of divergence)
"""

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats
from tqdm import tqdm

from config import ConsensusConfig

logger = logging.getLogger(__name__)


@dataclass
class PairwiseConsensusResult:
    explainer_a: str
    explainer_b: str
    spearman_mean: float
    spearman_std: float
    kendall_mean: float
    kendall_std: float
    top_k_intersection: dict[int, float]  # {k: mean_overlap}
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    wilcoxon_reject_h0: bool


def _compute_pairwise_rank_correlations(
    attrs_a: np.ndarray,
    attrs_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample Spearman and Kendall correlations between two attribution arrays.

    Args:
        attrs_a: Attributions from explainer A, shape (n_samples, n_features).
        attrs_b: Attributions from explainer B, shape (n_samples, n_features).

    Returns:
        Tuple of (spearman_rhos, kendall_taus) arrays, each shape (n_samples,).
    """
    n_samples = len(attrs_a)
    spearman_rhos = np.zeros(n_samples)
    kendall_taus = np.zeros(n_samples)

    abs_a = np.abs(attrs_a)
    abs_b = np.abs(attrs_b)

    for i in range(n_samples):
        # Skip constant vectors (all zeros) — correlation undefined
        if np.std(abs_a[i]) < 1e-12 or np.std(abs_b[i]) < 1e-12:
            spearman_rhos[i] = 0.0
            kendall_taus[i] = 0.0
            continue

        rho, _ = stats.spearmanr(abs_a[i], abs_b[i])
        tau, _ = stats.kendalltau(abs_a[i], abs_b[i])
        spearman_rhos[i] = rho if np.isfinite(rho) else 0.0
        kendall_taus[i] = tau if np.isfinite(tau) else 0.0

    return spearman_rhos, kendall_taus


def _compute_top_k_intersection(
    attrs_a: np.ndarray,
    attrs_b: np.ndarray,
    k_values: list[int],
) -> dict[int, float]:
    """Compute mean top-k feature overlap between two explainers.

    For each sample, get the top-k features by absolute attribution from each
    explainer and compute |intersection| / k.

    Returns:
        Dict mapping k -> mean intersection ratio across samples.
    """
    abs_a = np.abs(attrs_a)
    abs_b = np.abs(attrs_b)
    n_features = attrs_a.shape[1]

    results = {}
    for k in k_values:
        k_actual = min(k, n_features)
        overlaps = np.zeros(len(attrs_a))

        for i in range(len(attrs_a)):
            top_a = set(np.argsort(abs_a[i])[::-1][:k_actual])
            top_b = set(np.argsort(abs_b[i])[::-1][:k_actual])
            overlaps[i] = len(top_a & top_b) / k_actual

        results[k] = float(np.mean(overlaps))

    return results


def _compute_wilcoxon_test(
    attrs_a: np.ndarray,
    attrs_b: np.ndarray,
    alpha: float,
) -> tuple[float, float, bool]:
    """Wilcoxon signed-rank test on mean absolute attributions per feature.

    Tests H₀: the paired explainers yield statistically equivalent feature attributions.

    Args:
        attrs_a: Attributions from explainer A, shape (n_samples, n_features).
        attrs_b: Attributions from explainer B, shape (n_samples, n_features).
        alpha: Significance threshold.

    Returns:
        Tuple of (statistic, p_value, reject_h0).
    """
    # Aggregate: mean absolute attribution per feature across all samples
    mean_abs_a = np.mean(np.abs(attrs_a), axis=0)
    mean_abs_b = np.mean(np.abs(attrs_b), axis=0)

    # Wilcoxon requires non-zero differences
    diff = mean_abs_a - mean_abs_b
    if np.all(np.abs(diff) < 1e-12):
        return 0.0, 1.0, False

    try:
        stat, p_value = stats.wilcoxon(mean_abs_a, mean_abs_b, alternative="two-sided")
    except ValueError:
        # All differences are zero or too few samples
        return 0.0, 1.0, False

    return float(stat), float(p_value), p_value < alpha


def compute_pairwise_consensus(
    explanations: dict[str, np.ndarray],
    config: ConsensusConfig,
) -> list[PairwiseConsensusResult]:
    """Compute all pairwise consensus metrics between explainers.

    Args:
        explanations: Dict mapping explainer key (e.g., "DNN_SHAP") to
            attributions array of shape (n_samples, n_features).
        config: Consensus configuration.

    Returns:
        List of PairwiseConsensusResult for every explainer pair.
    """
    keys = sorted(explanations.keys())
    results = []

    for key_a, key_b in combinations(keys, 2):
        logger.info(f"  Consensus: {key_a} vs {key_b}")
        attrs_a = explanations[key_a]
        attrs_b = explanations[key_b]

        # Rank correlations
        spearman_rhos, kendall_taus = _compute_pairwise_rank_correlations(attrs_a, attrs_b)

        # Top-k intersection
        top_k = _compute_top_k_intersection(attrs_a, attrs_b, config.top_k_values)

        # Wilcoxon test
        w_stat, w_pval, w_reject = _compute_wilcoxon_test(attrs_a, attrs_b, config.alpha)

        results.append(PairwiseConsensusResult(
            explainer_a=key_a,
            explainer_b=key_b,
            spearman_mean=float(np.mean(spearman_rhos)),
            spearman_std=float(np.std(spearman_rhos)),
            kendall_mean=float(np.mean(kendall_taus)),
            kendall_std=float(np.std(kendall_taus)),
            top_k_intersection=top_k,
            wilcoxon_statistic=w_stat,
            wilcoxon_p_value=w_pval,
            wilcoxon_reject_h0=w_reject,
        ))

    return results


def compute_per_attack_consensus(
    explanations: dict[str, np.ndarray],
    y_labels: np.ndarray,
    label_names: list[str],
    config: ConsensusConfig,
) -> dict[str, list[PairwiseConsensusResult]]:
    """Compute consensus metrics broken down by attack type.

    Args:
        explanations: Dict mapping explainer key to attributions (n_samples, n_features).
        y_labels: Integer class labels for each sample.
        label_names: Human-readable class names.
        config: Consensus configuration.

    Returns:
        Dict mapping attack_type_name -> list of PairwiseConsensusResult.
    """
    # Skip benign class (typically class 0 or "BENIGN"/"normal")
    benign_labels = {"BENIGN", "benign", "normal", "Normal"}
    unique_labels = np.unique(y_labels)

    per_attack = {}
    for label_idx in unique_labels:
        label_name = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)
        if label_name in benign_labels:
            continue

        mask = y_labels == label_idx
        n_attack = np.sum(mask)
        if n_attack < 10:
            logger.warning(f"  Skipping {label_name}: only {n_attack} samples")
            continue

        logger.info(f"  Attack type: {label_name} ({n_attack} samples)")
        attack_explanations = {k: v[mask] for k, v in explanations.items()}
        per_attack[label_name] = compute_pairwise_consensus(attack_explanations, config)

    return per_attack


def consensus_to_dict(results: list[PairwiseConsensusResult]) -> list[dict]:
    """Convert consensus results to JSON-serializable dicts."""
    return [
        {
            "explainer_a": r.explainer_a,
            "explainer_b": r.explainer_b,
            "spearman_mean": r.spearman_mean,
            "spearman_std": r.spearman_std,
            "kendall_mean": r.kendall_mean,
            "kendall_std": r.kendall_std,
            "top_k_intersection": {str(k): v for k, v in r.top_k_intersection.items()},
            "wilcoxon_statistic": r.wilcoxon_statistic,
            "wilcoxon_p_value": r.wilcoxon_p_value,
            "wilcoxon_reject_h0": r.wilcoxon_reject_h0,
        }
        for r in results
    ]
