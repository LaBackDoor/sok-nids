"""Explainer consensus analysis: pairwise agreement metrics between XAI methods.

Metrics:
- Spearman's rank correlation coefficient (monotonic relationship)
- Kendall's tau-b (concordance of ranked pairs)
- Top-k feature intersection (overlap of most important features)
- Wilcoxon signed-rank test (statistical significance of divergence)
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

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


def _pair_checkpoint_path(checkpoint_dir: Path, key_a: str, key_b: str) -> Path:
    """Return the checkpoint file path for a consensus pair."""
    return checkpoint_dir / f"{key_a}__vs__{key_b}.json"


def _load_pair_checkpoint(checkpoint_dir: Path, key_a: str, key_b: str) -> PairwiseConsensusResult | None:
    """Load a cached pair result if it exists."""
    path = _pair_checkpoint_path(checkpoint_dir, key_a, key_b)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return PairwiseConsensusResult(
            explainer_a=data["explainer_a"],
            explainer_b=data["explainer_b"],
            spearman_mean=data["spearman_mean"],
            spearman_std=data["spearman_std"],
            kendall_mean=data["kendall_mean"],
            kendall_std=data["kendall_std"],
            top_k_intersection={int(k): v for k, v in data["top_k_intersection"].items()},
            wilcoxon_statistic=data["wilcoxon_statistic"],
            wilcoxon_p_value=data["wilcoxon_p_value"],
            wilcoxon_reject_h0=data["wilcoxon_reject_h0"],
        )
    except (json.JSONDecodeError, KeyError):
        return None


def _save_pair_checkpoint(checkpoint_dir: Path, result: PairwiseConsensusResult) -> None:
    """Save a single pair result to disk."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _pair_checkpoint_path(checkpoint_dir, result.explainer_a, result.explainer_b)
    path.write_text(json.dumps({
        "explainer_a": result.explainer_a,
        "explainer_b": result.explainer_b,
        "spearman_mean": result.spearman_mean,
        "spearman_std": result.spearman_std,
        "kendall_mean": result.kendall_mean,
        "kendall_std": result.kendall_std,
        "top_k_intersection": {str(k): v for k, v in result.top_k_intersection.items()},
        "wilcoxon_statistic": result.wilcoxon_statistic,
        "wilcoxon_p_value": result.wilcoxon_p_value,
        "wilcoxon_reject_h0": result.wilcoxon_reject_h0,
    }, indent=2))


def _compute_pairwise_rank_correlations(
    attrs_a: np.ndarray,
    attrs_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample Spearman and Kendall correlations between two attribution arrays."""
    n_samples = len(attrs_a)
    spearman_rhos = np.zeros(n_samples)
    kendall_taus = np.zeros(n_samples)

    abs_a = np.abs(attrs_a)
    abs_b = np.abs(attrs_b)

    for i in range(n_samples):
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
    """Compute mean top-k feature overlap between two explainers."""
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
    """Wilcoxon signed-rank test on mean absolute attributions per feature."""
    mean_abs_a = np.mean(np.abs(attrs_a), axis=0)
    mean_abs_b = np.mean(np.abs(attrs_b), axis=0)

    diff = mean_abs_a - mean_abs_b
    if np.all(np.abs(diff) < 1e-12):
        return 0.0, 1.0, False

    try:
        stat, p_value = stats.wilcoxon(mean_abs_a, mean_abs_b, alternative="two-sided")
    except ValueError:
        return 0.0, 1.0, False

    return float(stat), float(p_value), p_value < alpha


def _compute_single_pair(
    key_a: str,
    key_b: str,
    attrs_a: np.ndarray,
    attrs_b: np.ndarray,
    top_k_values: list[int],
    alpha: float,
) -> PairwiseConsensusResult:
    """Compute all consensus metrics for a single explainer pair.

    This is the unit of work for parallel execution.
    """
    spearman_rhos, kendall_taus = _compute_pairwise_rank_correlations(attrs_a, attrs_b)
    top_k = _compute_top_k_intersection(attrs_a, attrs_b, top_k_values)
    w_stat, w_pval, w_reject = _compute_wilcoxon_test(attrs_a, attrs_b, alpha)

    return PairwiseConsensusResult(
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
    )


def tag_pair(key_a: str, key_b: str) -> str:
    """Classify a consensus pair as within-mode or cross-mode."""
    a_is_pa = "PA-" in key_a
    b_is_pa = "PA-" in key_b
    if a_is_pa == b_is_pa:
        return "within-pa" if a_is_pa else "within-normal"
    return "cross-mode"


def compute_pairwise_consensus(
    explanations: dict[str, np.ndarray],
    config: ConsensusConfig,
    max_workers: int = 1,
    checkpoint_dir: Path | None = None,
) -> list[PairwiseConsensusResult]:
    """Compute all pairwise consensus metrics between explainers.

    Args:
        explanations: Dict mapping explainer key to attributions (n_samples, n_features).
        config: Consensus configuration.
        max_workers: Number of parallel workers for pair computation.
        checkpoint_dir: Directory for per-pair checkpoints (resume support).

    Returns:
        List of PairwiseConsensusResult for every explainer pair.
    """
    keys = sorted(explanations.keys())
    all_pairs = list(combinations(keys, 2))
    results = []

    # Load cached pairs
    uncached_pairs = []
    if checkpoint_dir:
        for key_a, key_b in all_pairs:
            cached = _load_pair_checkpoint(checkpoint_dir, key_a, key_b)
            if cached is not None:
                results.append(cached)
                logger.info(f"  Loaded cached: {key_a} vs {key_b}")
            else:
                uncached_pairs.append((key_a, key_b))
    else:
        uncached_pairs = all_pairs

    if not uncached_pairs:
        logger.info(f"  All {len(results)} pairs loaded from cache")
        return results

    logger.info(f"  Computing {len(uncached_pairs)} pairs ({len(results)} cached), workers={max_workers}")

    if max_workers <= 1:
        for key_a, key_b in uncached_pairs:
            logger.info(f"  Consensus: {key_a} vs {key_b} [{tag_pair(key_a, key_b)}]")
            result = _compute_single_pair(
                key_a, key_b, explanations[key_a], explanations[key_b],
                config.top_k_values, config.alpha,
            )
            results.append(result)
            if checkpoint_dir:
                _save_pair_checkpoint(checkpoint_dir, result)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for key_a, key_b in uncached_pairs:
                future = pool.submit(
                    _compute_single_pair,
                    key_a, key_b,
                    explanations[key_a], explanations[key_b],
                    config.top_k_values, config.alpha,
                )
                futures[future] = (key_a, key_b)

            for future in as_completed(futures):
                key_a, key_b = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if checkpoint_dir:
                        _save_pair_checkpoint(checkpoint_dir, result)
                    logger.info(
                        f"  {key_a} vs {key_b} [{tag_pair(key_a, key_b)}]: "
                        f"Spearman={result.spearman_mean:.3f}, Kendall={result.kendall_mean:.3f}"
                    )
                except Exception as e:
                    logger.error(f"  {key_a} vs {key_b} failed: {e}", exc_info=True)

    return results


def compute_per_attack_consensus(
    explanations: dict[str, np.ndarray],
    y_labels: np.ndarray,
    label_names: list[str],
    config: ConsensusConfig,
    max_workers: int = 1,
    checkpoint_dir: Path | None = None,
) -> dict[str, list[PairwiseConsensusResult]]:
    """Compute consensus metrics broken down by attack type."""
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

        attack_ckpt = checkpoint_dir / label_name if checkpoint_dir else None
        per_attack[label_name] = compute_pairwise_consensus(
            attack_explanations, config, max_workers, attack_ckpt,
        )

    return per_attack


def consensus_to_dict(results: list[PairwiseConsensusResult]) -> list[dict]:
    """Convert consensus results to JSON-serializable dicts."""
    return [
        {
            "explainer_a": r.explainer_a,
            "explainer_b": r.explainer_b,
            "pair_type": tag_pair(r.explainer_a, r.explainer_b),
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
