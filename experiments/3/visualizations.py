"""Visualization module for Experiment 3 results.

Generates:
- Consensus heatmaps (Spearman/Kendall pairwise matrices)
- Top-k intersection bar charts
- Wilcoxon p-value matrices
- SHAP dependence plots for top feature interactions
- Interaction strength heatmaps
- RRA/RMA alignment bar charts per explainer and attack type
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from consensus import PairwiseConsensusResult

logger = logging.getLogger(__name__)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved plot: {path.name}")


def plot_consensus_heatmap(
    results: list[PairwiseConsensusResult],
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot pairwise consensus as a heatmap.

    Args:
        results: List of pairwise consensus results.
        metric: One of "spearman_mean", "kendall_mean".
        title: Plot title.
        output_path: File path to save the plot.
    """
    # Extract unique explainer names
    names = sorted(set(
        [r.explainer_a for r in results] + [r.explainer_b for r in results]
    ))
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}

    matrix = np.eye(n)  # Diagonal = 1 (self-agreement)
    for r in results:
        val = getattr(r, metric)
        i, j = idx[r.explainer_a], idx[r.explainer_b]
        matrix[i, j] = val
        matrix[j, i] = val

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        xticklabels=names,
        yticklabels=names,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax,
    )
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    _save_fig(fig, output_path)


def plot_wilcoxon_pvalue_matrix(
    results: list[PairwiseConsensusResult],
    alpha: float,
    output_path: Path,
) -> None:
    """Plot Wilcoxon p-value matrix with significance markers."""
    names = sorted(set(
        [r.explainer_a for r in results] + [r.explainer_b for r in results]
    ))
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}

    matrix = np.ones((n, n))
    annot = [["-"] * n for _ in range(n)]

    for r in results:
        i, j = idx[r.explainer_a], idx[r.explainer_b]
        matrix[i, j] = r.wilcoxon_p_value
        matrix[j, i] = r.wilcoxon_p_value
        label = f"{r.wilcoxon_p_value:.2e}"
        if r.wilcoxon_reject_h0:
            label += " *"
        annot[i][j] = label
        annot[j][i] = label

    for i in range(n):
        annot[i][i] = "-"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        xticklabels=names,
        yticklabels=names,
        annot=annot,
        fmt="",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.1,
        ax=ax,
    )
    ax.set_title(f"Wilcoxon Signed-Rank Test p-values (a={alpha})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    _save_fig(fig, output_path)


def plot_top_k_intersection(
    results: list[PairwiseConsensusResult],
    k_values: list[int],
    output_path: Path,
) -> None:
    """Bar chart of top-k feature intersection for each explainer pair."""
    for k in k_values:
        labels = []
        values = []
        for r in results:
            labels.append(f"{r.explainer_a}\nvs\n{r.explainer_b}")
            values.append(r.top_k_intersection.get(k, 0.0))

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 6))
        bars = ax.bar(range(len(labels)), values, color="steelblue")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f"Top-{k} Intersection Ratio")
        ax.set_title(f"Top-{k} Feature Agreement Between Explainers")
        ax.set_ylim(0, 1)
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.3, label="Perfect agreement")
        ax.legend()
        _save_fig(fig, output_path.parent / f"{output_path.stem}_k{k}.png")


def plot_interaction_heatmap(
    interaction_matrix: np.ndarray,
    feature_names: list[str],
    top_n: int,
    title: str,
    output_path: Path,
) -> None:
    """Heatmap of top feature interactions."""
    # Select top features by total interaction strength
    total_strength = np.sum(np.abs(interaction_matrix), axis=1)
    top_indices = np.argsort(total_strength)[::-1][:top_n]

    sub_matrix = interaction_matrix[np.ix_(top_indices, top_indices)]
    sub_names = [feature_names[i] if i < len(feature_names) else str(i) for i in top_indices]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        sub_matrix,
        xticklabels=sub_names,
        yticklabels=sub_names,
        annot=True,
        fmt=".4f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    _save_fig(fig, output_path)


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_data: np.ndarray,
    feature_a_idx: int,
    feature_b_idx: int,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """SHAP dependence plot for a feature pair interaction."""
    feat_a_name = feature_names[feature_a_idx] if feature_a_idx < len(feature_names) else str(feature_a_idx)
    feat_b_name = feature_names[feature_b_idx] if feature_b_idx < len(feature_names) else str(feature_b_idx)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        X_data[:, feature_a_idx],
        shap_values[:, feature_a_idx],
        c=X_data[:, feature_b_idx],
        cmap="coolwarm",
        alpha=0.5,
        s=5,
    )
    plt.colorbar(scatter, ax=ax, label=feat_b_name)
    ax.set_xlabel(feat_a_name)
    ax.set_ylabel(f"SHAP value for {feat_a_name}")
    ax.set_title(f"SHAP Dependence: {feat_a_name} (colored by {feat_b_name})")
    _save_fig(fig, output_path)


def plot_alignment_scores(
    alignment_results: list[dict],
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    """Bar chart of RRA or RMA scores grouped by attack type."""
    if not alignment_results:
        return

    # Group by attack type
    attack_types = sorted(set(r["attack_type"] for r in alignment_results))
    explainer_keys = sorted(set(r["explainer_key"] for r in alignment_results))

    n_attacks = len(attack_types)
    n_explainers = len(explainer_keys)
    if n_attacks == 0 or n_explainers == 0:
        return

    x = np.arange(n_attacks)
    width = 0.8 / n_explainers

    fig, ax = plt.subplots(figsize=(max(12, n_attacks * 2), 6))
    colors = plt.cm.Set2(np.linspace(0, 1, n_explainers))

    for i, exp_key in enumerate(explainer_keys):
        values = []
        for attack in attack_types:
            match = [r for r in alignment_results if r["attack_type"] == attack and r["explainer_key"] == exp_key]
            values.append(match[0][metric] if match else 0.0)
        ax.bar(x + i * width, values, width, label=exp_key, color=colors[i])

    ax.set_xticks(x + width * (n_explainers - 1) / 2)
    ax.set_xticklabels(attack_types, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(0, 1)
    _save_fig(fig, output_path)


def generate_all_plots(
    consensus_results: list[PairwiseConsensusResult] | None,
    per_attack_consensus: dict | None,
    interaction_matrices: dict | None,
    top_interactions: dict | None,
    alignment_results: list[dict] | None,
    feature_names: list[str],
    dataset_name: str,
    plot_dir: Path,
    config,
    shap_values_for_dependence: np.ndarray | None = None,
    X_data_for_dependence: np.ndarray | None = None,
    max_workers: int = 8,
) -> None:
    """Generate all visualization plots for a dataset using parallel workers."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    plot_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, callable]] = []

    if consensus_results:
        tasks.append(("spearman_heatmap", lambda: plot_consensus_heatmap(
            consensus_results, "spearman_mean",
            f"Spearman Rank Correlation — {dataset_name}",
            plot_dir / f"{dataset_name}_spearman_heatmap.png",
        )))
        tasks.append(("kendall_heatmap", lambda: plot_consensus_heatmap(
            consensus_results, "kendall_mean",
            f"Kendall's Tau — {dataset_name}",
            plot_dir / f"{dataset_name}_kendall_heatmap.png",
        )))
        tasks.append(("wilcoxon_pvalues", lambda: plot_wilcoxon_pvalue_matrix(
            consensus_results, config.consensus.alpha,
            plot_dir / f"{dataset_name}_wilcoxon_pvalues.png",
        )))
        tasks.append(("top_k_intersection", lambda: plot_top_k_intersection(
            consensus_results, config.consensus.top_k_values,
            plot_dir / f"{dataset_name}_top_k_intersection",
        )))

    if interaction_matrices:
        for model_name, matrix in interaction_matrices.items():
            _mn, _mx = model_name, matrix
            tasks.append((f"interactions_{_mn}", lambda mn=_mn, mx=_mx: plot_interaction_heatmap(
                mx, feature_names,
                top_n=min(15, mx.shape[0]),
                title=f"Feature Interaction Strength ({mn}) — {dataset_name}",
                output_path=plot_dir / f"{dataset_name}_{mn}_interactions.png",
            )))

    if top_interactions and shap_values_for_dependence is not None and X_data_for_dependence is not None:
        for model_name, pairs in top_interactions.items():
            for pair_idx, pair in enumerate(pairs[:5]):
                _mn, _pi, _p = model_name, pair_idx, pair
                tasks.append((f"dependence_{_mn}_{_pi}", lambda mn=_mn, pi=_pi, p=_p: plot_shap_dependence(
                    shap_values_for_dependence,
                    X_data_for_dependence,
                    p["feature_a_idx"],
                    p["feature_b_idx"],
                    feature_names,
                    plot_dir / f"{dataset_name}_{mn}_dependence_{pi}.png",
                )))

    if alignment_results:
        tasks.append(("rra_scores", lambda: plot_alignment_scores(
            alignment_results, "rra_score",
            f"Relevance Rank Accuracy (RRA) — {dataset_name}",
            plot_dir / f"{dataset_name}_rra_scores.png",
        )))
        tasks.append(("rma_scores", lambda: plot_alignment_scores(
            alignment_results, "rma_score",
            f"Relevance Mass Accuracy (RMA) — {dataset_name}",
            plot_dir / f"{dataset_name}_rma_scores.png",
        )))

    if not tasks:
        logger.info("  No plots to generate")
        return

    logger.info(f"  Generating {len(tasks)} plots with {min(max_workers, len(tasks))} workers")

    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"  Plot '{name}' failed: {e}", exc_info=True)
