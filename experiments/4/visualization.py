"""Visualization and reporting for Experiment 4."""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def generate_all_plots(all_results: dict, output_dir: Path) -> None:
    """Generate comparison plots for Experiment 4 results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Flatten results into DataFrame
    rows = []
    for ds_name, ds_data in all_results.items():
        if "benchmarks" not in ds_data:
            continue
        for entry in ds_data["benchmarks"]:
            entry["dataset"] = ds_name
            rows.append(entry)

    if not rows:
        logger.warning("No benchmark results to plot.")
        return

    df = pd.DataFrame(rows)
    df["label"] = df["model"] + "\n" + df["selection_method"]

    # Determine pipeline type
    stat_methods = {"Chi-Squared", "PCA", "Spearman", "InfoGain"}

    def _classify_pipeline(method: str) -> str:
        if method == "Full":
            return "Baseline"
        if method in stat_methods:
            return "Statistical"
        if method.startswith("PA-"):
            return "PA-XAI"
        return "XAI"

    df["pipeline"] = df["selection_method"].apply(_classify_pipeline)

    # Color palette
    palette = {
        "Baseline": "#666666",
        "Statistical": "#2196F3",
        "XAI": "#FF5722",
        "PA-XAI": "#4CAF50",
    }

    for ds_name in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds_name]

        # --- Accuracy comparison ---
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(data=ds_df, x="label", y="accuracy", hue="pipeline",
                    palette=palette, ax=ax, dodge=False)
        ax.set_title(f"Classification Accuracy — {ds_name}")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        fig.savefig(plot_dir / f"{ds_name}_accuracy.png", dpi=150)
        plt.close(fig)

        # --- F1 comparison ---
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(data=ds_df, x="label", y="f1_weighted", hue="pipeline",
                    palette=palette, ax=ax, dodge=False)
        ax.set_title(f"Weighted F1-Score — {ds_name}")
        ax.set_ylabel("F1 (Weighted)")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        fig.savefig(plot_dir / f"{ds_name}_f1.png", dpi=150)
        plt.close(fig)

        # --- FPR comparison ---
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(data=ds_df, x="label", y="fpr_macro", hue="pipeline",
                    palette=palette, ax=ax, dodge=False)
        ax.set_title(f"False Positive Rate (Macro) — {ds_name}")
        ax.set_ylabel("FPR")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        fig.savefig(plot_dir / f"{ds_name}_fpr.png", dpi=150)
        plt.close(fig)

        # --- Inference latency ---
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(data=ds_df, x="label", y="inference_ms_per_sample_mean", hue="pipeline",
                    palette=palette, ax=ax, dodge=False)
        ax.set_title(f"Inference Latency (ms/sample) — {ds_name}")
        ax.set_ylabel("ms / sample")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        fig.savefig(plot_dir / f"{ds_name}_latency.png", dpi=150)
        plt.close(fig)

        # --- Training time ---
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(data=ds_df, x="label", y="train_time_s", hue="pipeline",
                    palette=palette, ax=ax, dodge=False)
        ax.set_title(f"Training Time (seconds) — {ds_name}")
        ax.set_ylabel("Time (s)")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        fig.savefig(plot_dir / f"{ds_name}_train_time.png", dpi=150)
        plt.close(fig)

    # --- Cross-dataset heatmaps ---
    if len(df["dataset"].unique()) > 1:
        for metric, title in [
            ("accuracy", "Accuracy"),
            ("f1_weighted", "F1 (Weighted)"),
            ("fpr_macro", "FPR (Macro)"),
            ("inference_ms_per_sample_mean", "Inference Latency (ms)"),
        ]:
            try:
                df["method_model"] = df["selection_method"] + " / " + df["model"]
                pivot = df.pivot_table(index="method_model", columns="dataset", values=metric)
                if pivot.empty:
                    continue
                fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.4)))
                sns.heatmap(pivot.astype(float), annot=True, fmt=".4f", cmap="YlOrRd", ax=ax)
                ax.set_title(f"{title} — All Datasets")
                plt.tight_layout()
                fig.savefig(plot_dir / f"heatmap_{metric}.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate heatmap for {metric}: {e}")

    # --- Feature selection comparison: selected feature counts ---
    selection_rows = []
    for ds_name, ds_data in all_results.items():
        if "feature_selections" not in ds_data:
            continue
        for sel in ds_data["feature_selections"]:
            selection_rows.append({
                "dataset": ds_name,
                "method": sel["method_name"],
                "n_selected": sel["n_selected"],
                "n_original": sel["n_original"],
                "selection_time_s": sel["selection_time_s"],
                "pipeline": _classify_pipeline(sel["method_name"]),
            })

    if selection_rows:
        sel_df = pd.DataFrame(selection_rows)
        for ds_name in sel_df["dataset"].unique():
            ds_sel = sel_df[sel_df["dataset"] == ds_name]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Feature count
            colors = [palette.get(p, "#999") for p in ds_sel["pipeline"]]
            ax1.barh(ds_sel["method"], ds_sel["n_selected"], color=colors)
            ax1.axvline(ds_sel["n_original"].iloc[0], color="red", linestyle="--",
                        label=f"Full ({ds_sel['n_original'].iloc[0]})")
            ax1.set_xlabel("Features Selected")
            ax1.set_title(f"Feature Selection — {ds_name}")
            ax1.legend()

            # Selection time
            ax2.barh(ds_sel["method"], ds_sel["selection_time_s"], color=colors)
            ax2.set_xlabel("Selection Time (s)")
            ax2.set_title(f"Selection Time — {ds_name}")

            plt.tight_layout()
            fig.savefig(plot_dir / f"{ds_name}_feature_selection.png", dpi=150)
            plt.close(fig)

    logger.info(f"Plots saved to {plot_dir}")


def generate_summary_csv(all_results: dict, output_dir: Path) -> None:
    """Generate a CSV summary of all benchmark results."""
    import csv

    rows = []
    for ds_name, ds_data in all_results.items():
        if "benchmarks" not in ds_data:
            continue
        for entry in ds_data["benchmarks"]:
            entry["dataset"] = ds_name
            rows.append(entry)

    if not rows:
        return

    cols = [
        "dataset", "model", "selection_method", "n_features",
        "accuracy", "f1_weighted", "f1_macro", "auc_roc",
        "fpr_macro", "far_overall",
        "train_time_s", "inference_ms_per_sample_mean",
    ]

    path = output_dir / "benchmark_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: f"{r.get(k, 'N/A')}" for k in cols})

    logger.info(f"Summary CSV saved to {path}")
