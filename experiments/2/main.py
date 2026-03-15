#!/usr/bin/env python3
"""Experiment 2: Adversarial Robustness and Explanation-Aware Attacks.

Run the full pipeline:
  - Phase 1 (scaffolding): Integrity Scaffolding Attack (M3 threat model)
  - Phase 2 (adversarial): Generate FGSM/PGD adversarial examples (M1 threat model)
  - Phase 3 (robustness): Evaluate Lipschitz, ExplSim, ClassEq robustness metrics

Usage:
    # Full experiment on all datasets
    python experiments/2/main.py

    # Specific dataset(s)
    python experiments/2/main.py --datasets nsl-kdd cic-ids-2017

    # Specific phase
    python experiments/2/main.py --phase scaffolding
    python experiments/2/main.py --phase adversarial
    python experiments/2/main.py --phase robustness

    # Reduced samples for faster testing
    python experiments/2/main.py --num-attack-samples 100 --num-robustness-samples 100
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add experiment directories to path for local imports.
# We must import Experiment 1 modules first (they internally do `from config import DataConfig`
# which must resolve to exp1/config.py), then import Experiment 2 modules.
exp2_dir = os.path.dirname(os.path.abspath(__file__))
exp1_dir = os.path.join(exp2_dir, "..", "1")

# Step 1: Import all Experiment 1 modules with exp1 on sys.path
sys.path.insert(0, exp1_dir)

from data_loader import DataConfig, DatasetBundle, load_dataset
from explainers import (
    ExplanationResult,
    explain_ig,
    explain_lime,
    explain_shap_dnn,
    explain_shap_rf,
)
from models import (
    DNNWrapper,
    NIDSNet,
    RFWrapper,
    SoftmaxModel,
    load_models,
    save_models,
    train_dnn,
    train_rf,
)

# Also import exp1 config classes directly
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("exp1_config", os.path.join(exp1_dir, "config.py"))
_exp1_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_exp1_cfg)
DNNConfig = _exp1_cfg.DNNConfig
RFConfig = _exp1_cfg.RFConfig
ExplainerConfig = _exp1_cfg.ExplainerConfig

# Step 2: Now import Experiment 2 modules (put exp2 dir first on path)
sys.path.insert(0, exp2_dir)

from attacks import generate_adversarial_examples  # noqa: E402
from robustness import evaluate_robustness_for_method  # noqa: E402
from scaffolding import run_scaffolding_attack  # noqa: E402

# Import Experiment 2 config via importlib to avoid collision
_spec2 = _ilu.spec_from_file_location("exp2_config", os.path.join(exp2_dir, "config.py"))
_exp2_cfg = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_exp2_cfg)
Experiment2Config = _exp2_cfg.Experiment2Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("experiment2")


def setup_device() -> tuple[torch.device, int]:
    """Detect available GPUs and set up device."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device("cuda")
        logger.info(f"CUDA available: {num_gpus} GPU(s) detected")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"  GPU {i}: {props.name} | "
                f"{props.total_memory / 1e9:.1f} GB | "
                f"Compute {props.major}.{props.minor}"
            )
    else:
        device = torch.device("cpu")
        num_gpus = 0
        logger.warning("No CUDA GPUs available. Running on CPU.")
    return device, num_gpus


def ensure_models_trained(
    dataset: DatasetBundle,
    exp1_results_dir: Path,
    device: torch.device,
    num_gpus: int,
) -> tuple[NIDSNet, DNNWrapper, RFWrapper]:
    """Load pre-trained models from Experiment 1, or train if not available."""
    model_dir = exp1_results_dir / "models" / dataset.dataset_name

    if (model_dir / "dnn.pt").exists() and (model_dir / "rf.joblib").exists():
        logger.info(f"  Loading pre-trained models from {model_dir}")
        dnn_model, rf_model = load_models(
            exp1_results_dir,
            dataset.dataset_name,
            dataset.X_train.shape[1],
            dataset.num_classes,
            DNNConfig(),
            device,
        )
        dnn_wrapper = DNNWrapper(dnn_model, device)
        rf_wrapper = RFWrapper(rf_model, num_classes=dataset.num_classes)
        return dnn_model, dnn_wrapper, rf_wrapper

    # Train models if not available
    logger.info("  Models not found. Training DNN and RF...")
    dnn_model, dnn_wrapper, _ = train_dnn(dataset, DNNConfig(), device, num_gpus)
    rf_model, rf_wrapper, _ = train_rf(dataset, RFConfig())
    save_models(dnn_model, rf_model, exp1_results_dir, dataset.dataset_name)
    return dnn_model, dnn_wrapper, rf_wrapper


def make_explain_fn(
    method: str,
    dnn_model: NIDSNet,
    dnn_wrapper: DNNWrapper,
    rf_wrapper: RFWrapper,
    dataset: DatasetBundle,
    device: torch.device,
    config: Experiment2Config,
):
    """Create a callable explain function for robustness evaluation."""
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(
        len(dataset.X_train),
        size=config.robustness.shap_background_samples,
        replace=False,
    )
    X_bg = dataset.X_train[bg_idx]

    explainer_cfg = ExplainerConfig(
        shap_background_samples=config.robustness.shap_background_samples,
        lime_num_features=config.robustness.lime_num_features,
        lime_num_samples=config.robustness.lime_num_samples,
        ig_n_steps=config.robustness.ig_n_steps,
        ig_internal_batch_size=config.robustness.ig_internal_batch_size,
    )

    if method == "SHAP":
        def fn(X):
            r = explain_shap_dnn(dnn_model, X, X_bg, device, explainer_cfg)
            return r.attributions
        return fn

    elif method == "LIME":
        def fn(X):
            r = explain_lime(
                dnn_wrapper.predict_proba, X, dataset.X_train,
                dataset.feature_names, dataset.num_classes, "DNN", explainer_cfg,
            )
            return r.attributions
        return fn

    elif method == "IG":
        def fn(X):
            r = explain_ig(dnn_model, X, device, explainer_cfg)
            return r.attributions
        return fn

    else:
        raise ValueError(f"Unknown method: {method}")


# ─── Phase: Scaffolding Attack (M3) ────────────────────────────────────────────

def phase_scaffolding(
    dataset: DatasetBundle,
    dnn_wrapper: DNNWrapper,
    config: Experiment2Config,
    device: torch.device,
) -> dict:
    """Run the Integrity Scaffolding Attack on LIME and SHAP."""
    return run_scaffolding_attack(
        legitimate_wrapper=dnn_wrapper,
        dataset_bundle=dataset,
        config=config.scaffolding,
        device=device,
    )


# ─── Phase: Adversarial Example Generation (M1) ────────────────────────────────

def phase_adversarial(
    dataset: DatasetBundle,
    dnn_model: NIDSNet,
    config: Experiment2Config,
    device: torch.device,
) -> dict:
    """Generate FGSM and PGD adversarial examples."""
    logger.info(f"=== ADVERSARIAL ATTACKS on {dataset.dataset_name} ===")

    adv_results = generate_adversarial_examples(
        model=dnn_model,
        X=dataset.X_test,
        y=dataset.y_test,
        config=config.attack,
        device=device,
    )

    # Save adversarial examples
    adv_dir = config.output_dir / dataset.dataset_name / "adversarial"
    adv_dir.mkdir(parents=True, exist_ok=True)

    np.save(adv_dir / "indices.npy", adv_results["indices"])
    np.save(adv_dir / "X_clean.npy", adv_results["X_clean"])
    np.save(adv_dir / "y_clean.npy", adv_results["y_clean"])

    for attack_type in ["fgsm", "pgd"]:
        for eps, X_adv in adv_results[attack_type].items():
            fname = f"{attack_type}_eps{eps:.4f}.npy"
            np.save(adv_dir / fname, X_adv)

    # Save attack success rates
    summary = {
        "fgsm_success_rates": {str(k): v for k, v in adv_results["fgsm_success"].items()},
        "pgd_success_rates": {str(k): v for k, v in adv_results["pgd_success"].items()},
        "num_samples": len(adv_results["indices"]),
    }
    with open(adv_dir / "attack_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"  Adversarial examples saved to {adv_dir}")
    return summary


# ─── Phase: Robustness Evaluation ──────────────────────────────────────────────

def phase_robustness(
    dataset: DatasetBundle,
    dnn_model: NIDSNet,
    dnn_wrapper: DNNWrapper,
    rf_wrapper: RFWrapper,
    config: Experiment2Config,
    device: torch.device,
    adv_results: dict | None = None,
) -> list[dict]:
    """Evaluate Lipschitz, ExplSim, ClassEq for each XAI method + attack combo."""
    logger.info(f"=== ROBUSTNESS EVALUATION on {dataset.dataset_name} ===")

    # Load or use provided adversarial examples
    adv_dir = config.output_dir / dataset.dataset_name / "adversarial"
    if adv_results is None:
        logger.info("  Loading adversarial examples from disk...")
        X_clean = np.load(adv_dir / "X_clean.npy")
        y_clean = np.load(adv_dir / "y_clean.npy")
        adv_examples = {}
        for path in sorted(adv_dir.glob("*.npy")):
            if path.stem.startswith(("fgsm_", "pgd_")):
                adv_examples[path.stem] = np.load(path)
    else:
        X_clean = adv_results["X_clean"]
        y_clean = adv_results["y_clean"]
        adv_examples = {}
        for attack_type in ["fgsm", "pgd"]:
            for eps, X_adv in adv_results[attack_type].items():
                adv_examples[f"{attack_type}_eps{eps:.4f}"] = X_adv

    # Limit samples for robustness evaluation
    n = min(config.robustness.num_samples, len(X_clean))
    X_clean = X_clean[:n]
    y_clean = y_clean[:n]
    adv_examples = {k: v[:n] for k, v in adv_examples.items()}

    all_results = []

    for method in config.robustness.explanation_methods:
        logger.info(f"  --- Method: {method} ---")
        try:
            explain_fn = make_explain_fn(
                method, dnn_model, dnn_wrapper, rf_wrapper,
                dataset, device, config,
            )
        except Exception as e:
            logger.error(f"  Failed to create explain_fn for {method}: {e}")
            continue

        for adv_name, X_adv in adv_examples.items():
            # Parse attack name and epsilon
            parts = adv_name.split("_eps")
            attack_name = parts[0]
            epsilon = float(parts[1]) if len(parts) > 1 else 0.0

            try:
                result = evaluate_robustness_for_method(
                    method_name=method,
                    explain_fn=explain_fn,
                    predict_fn=dnn_wrapper.predict_proba,
                    X_clean=X_clean,
                    X_adv=X_adv,
                    attack_name=attack_name,
                    epsilon=epsilon,
                    config=config.robustness,
                )
                result["dataset"] = dataset.dataset_name
                all_results.append(result)
            except Exception as e:
                logger.error(
                    f"  Robustness eval failed for {method}/{adv_name}: {e}",
                    exc_info=True,
                )

    # Save robustness results
    rob_dir = config.output_dir / dataset.dataset_name / "robustness"
    rob_dir.mkdir(parents=True, exist_ok=True)
    with open(rob_dir / "robustness_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=_json_serialize)
    logger.info(f"  Robustness metrics saved to {rob_dir}")

    return all_results


# ─── Summary and Plotting ──────────────────────────────────────────────────────

def generate_plots(all_results: dict, output_dir: Path) -> None:
    """Generate summary plots for Experiment 2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Attack success rates vs epsilon
    for ds_name, ds_results in all_results.items():
        if "adversarial" not in ds_results:
            continue
        adv = ds_results["adversarial"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, attack_type in enumerate(["fgsm", "pgd"]):
            key = f"{attack_type}_success_rates"
            if key not in adv:
                continue
            rates = adv[key]
            epsilons = sorted([float(e) for e in rates.keys()])
            success = [rates[str(e)] for e in epsilons]

            axes[idx].plot(epsilons, success, "o-", linewidth=2, markersize=8)
            axes[idx].set_title(f"{attack_type.upper()} Attack — {ds_name}")
            axes[idx].set_xlabel("Epsilon (L-inf)")
            axes[idx].set_ylabel("Attack Success Rate")
            axes[idx].set_ylim(-0.05, 1.05)
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(plot_dir / f"{ds_name}_attack_success.png", dpi=150)
        plt.close(fig)

    # Plot 2: Lipschitz constants per method/attack
    for ds_name, ds_results in all_results.items():
        if "robustness" not in ds_results:
            continue
        rob = ds_results["robustness"]

        methods = sorted(set(r["method"] for r in rob))
        attacks = sorted(set(f"{r['attack']}_eps{r['epsilon']}" for r in rob))

        if not methods or not attacks:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.8 / len(methods)

        for m_idx, method in enumerate(methods):
            lip_vals = []
            labels = []
            for attack in attacks:
                parts = attack.split("_eps")
                a_name, a_eps = parts[0], float(parts[1])
                matching = [
                    r for r in rob
                    if r["method"] == method
                    and r["attack"] == a_name
                    and abs(r["epsilon"] - a_eps) < 1e-6
                ]
                if matching and "lipschitz" in matching[0]:
                    lip_vals.append(matching[0]["lipschitz"].get("lipschitz_mean", 0))
                else:
                    lip_vals.append(0)
                labels.append(attack)

            x = np.arange(len(labels))
            ax.bar(x + m_idx * width, lip_vals, width, label=method)

        ax.set_title(f"Empirical Lipschitz Constants — {ds_name}")
        ax.set_ylabel("Mean Lipschitz Constant")
        ax.set_xticks(np.arange(len(attacks)) + width * len(methods) / 2)
        ax.set_xticklabels(attacks, rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / f"{ds_name}_lipschitz.png", dpi=150)
        plt.close(fig)

    # Plot 3: Scaffolding attack success
    scaffolding_data = []
    for ds_name, ds_results in all_results.items():
        if "scaffolding" not in ds_results:
            continue
        sc = ds_results["scaffolding"]
        for xai_method in ["lime", "shap"]:
            if xai_method in sc:
                scaffolding_data.append({
                    "dataset": ds_name,
                    "method": xai_method.upper(),
                    "top1": sc[xai_method]["dummy_top1_rate"],
                    "top3": sc[xai_method]["dummy_top3_rate"],
                })

    if scaffolding_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"{d['dataset']}\n{d['method']}" for d in scaffolding_data]
        top1 = [d["top1"] for d in scaffolding_data]
        top3 = [d["top3"] for d in scaffolding_data]
        x = np.arange(len(labels))
        ax.bar(x - 0.15, top1, 0.3, label="Dummy in Top-1", color="crimson")
        ax.bar(x + 0.15, top3, 0.3, label="Dummy in Top-3", color="salmon")
        ax.set_ylabel("Success Rate")
        ax.set_title("Scaffolding Attack Success Rate")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "scaffolding_success.png", dpi=150)
        plt.close(fig)

    logger.info(f"Plots saved to {plot_dir}")


def _print_final_summary(all_results: dict) -> None:
    """Print human-readable summary."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 2 SUMMARY: Adversarial Robustness & Explanation-Aware Attacks")
    logger.info("=" * 80)

    for ds_name, ds_results in all_results.items():
        logger.info(f"\n--- {ds_name} ---")

        if "adversarial" in ds_results:
            adv = ds_results["adversarial"]
            for attack in ["fgsm", "pgd"]:
                key = f"{attack}_success_rates"
                if key in adv:
                    rates = adv[key]
                    logger.info(f"  {attack.upper()} success rates: {rates}")

        if "scaffolding" in ds_results:
            sc = ds_results["scaffolding"]
            for method in ["lime", "shap"]:
                if method in sc:
                    logger.info(
                        f"  Scaffolding ({method.upper()}): "
                        f"top1={sc[method]['dummy_top1_rate']:.3f}, "
                        f"top3={sc[method]['dummy_top3_rate']:.3f}, "
                        f"mean_rank={sc[method]['dummy_mean_rank']:.1f}"
                    )

        if "robustness" in ds_results:
            for r in ds_results["robustness"]:
                lip = r.get("lipschitz", {})
                sim = r.get("similarity", {})
                ceq = r.get("classification_equivalence", {})
                logger.info(
                    f"  {r['method']} vs {r['attack']}(eps={r['epsilon']}): "
                    f"Lip_max={lip.get('lipschitz_max', 'N/A'):.2f} "
                    f"Lip_mean={lip.get('lipschitz_mean', 'N/A'):.2f} "
                    f"CosSim={sim.get('cosine_similarity_mean', 'N/A'):.3f} "
                    f"ClsEq_Jacc={ceq.get('top_k_jaccard_mean', 'N/A'):.3f}"
                )


def _json_serialize(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ─── Main Orchestrator ─────────────────────────────────────────────────────────

def run_experiment(config: Experiment2Config, datasets: list[str], phases: list[str]):
    """Main experiment runner."""
    device, num_gpus = setup_device()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(config.output_dir / "config.json", "w") as f:
        json.dump(
            {
                "datasets": datasets,
                "phases": phases,
                "attack": vars(config.attack),
                "scaffolding": vars(config.scaffolding),
                "robustness": vars(config.robustness),
                "exp1_results_dir": str(config.exp1_results_dir),
                "seed": config.seed,
            },
            f,
            indent=2,
            default=_json_serialize,
        )

    data_config = DataConfig()
    all_results = {}

    for ds_name in datasets:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"DATASET: {ds_name}")
        logger.info(f"{'=' * 60}")

        ds_start = time.time()

        try:
            dataset = load_dataset(ds_name, data_config)
        except Exception as e:
            logger.error(f"Failed to load {ds_name}: {e}", exc_info=True)
            continue

        # Load or train models
        try:
            dnn_model, dnn_wrapper, rf_wrapper = ensure_models_trained(
                dataset, config.exp1_results_dir, device, num_gpus
            )
        except Exception as e:
            logger.error(f"Failed to load/train models for {ds_name}: {e}", exc_info=True)
            continue

        ds_results = {}

        # Phase: Scaffolding Attack (M3)
        if ("scaffolding" in phases or "all" in phases) and \
                ds_name in config.scaffolding.scaffolding_datasets:
            try:
                logger.info(f"\n--- Phase: Scaffolding Attack ---")
                sc_results = phase_scaffolding(dataset, dnn_wrapper, config, device)
                ds_results["scaffolding"] = sc_results
            except Exception as e:
                logger.error(f"Scaffolding failed for {ds_name}: {e}", exc_info=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Phase: Adversarial Attack Generation (M1)
        adv_results_raw = None
        if "adversarial" in phases or "all" in phases:
            try:
                logger.info(f"\n--- Phase: Adversarial Attacks ---")
                adv_summary = phase_adversarial(dataset, dnn_model, config, device)
                ds_results["adversarial"] = adv_summary
                # Keep raw results for robustness phase
                adv_results_raw = generate_adversarial_examples(
                    dnn_model, dataset.X_test, dataset.y_test, config.attack, device
                ) if "robustness" in phases or "all" in phases else None
                # Actually re-load from saved files to avoid double computation
                adv_results_raw = None
            except Exception as e:
                logger.error(f"Adversarial attacks failed for {ds_name}: {e}", exc_info=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Phase: Robustness Evaluation
        if "robustness" in phases or "all" in phases:
            try:
                logger.info(f"\n--- Phase: Robustness Evaluation ---")
                rob_results = phase_robustness(
                    dataset, dnn_model, dnn_wrapper, rf_wrapper,
                    config, device,
                )
                ds_results["robustness"] = rob_results
            except Exception as e:
                logger.error(f"Robustness eval failed for {ds_name}: {e}", exc_info=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        ds_elapsed = time.time() - ds_start
        logger.info(f"\nDataset {ds_name} completed in {ds_elapsed:.1f}s")
        all_results[ds_name] = ds_results

    # Save combined results
    combined_path = config.output_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_serialize)
    logger.info(f"\nAll results saved to {combined_path}")

    # Generate plots
    try:
        generate_plots(all_results, config.output_dir)
    except Exception as e:
        logger.error(f"Plot generation failed: {e}", exc_info=True)

    _print_final_summary(all_results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 2: Adversarial Robustness and Explanation-Aware Attacks"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=["nsl-kdd", "cic-ids-2017", "unsw-nb15", "cse-cic-ids2018"],
        help="Datasets to process (default: all)",
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        default=["all"],
        choices=["all", "scaffolding", "adversarial", "robustness"],
        help="Experiment phase(s) to run",
    )
    parser.add_argument(
        "--num-attack-samples",
        type=int,
        default=None,
        help="Number of samples for adversarial attacks (default: 1000)",
    )
    parser.add_argument(
        "--num-robustness-samples",
        type=int,
        default=None,
        help="Number of samples for robustness evaluation (default: 1000)",
    )
    parser.add_argument(
        "--num-scaffolding-samples",
        type=int,
        default=None,
        help="Number of samples for scaffolding attack (default: 500)",
    )
    parser.add_argument(
        "--exp1-dir",
        type=str,
        default=None,
        help="Experiment 1 results directory (for pre-trained models)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Experiment2Config()

    # Apply CLI overrides
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.exp1_dir:
        config.exp1_results_dir = Path(args.exp1_dir)
    if args.num_attack_samples:
        config.attack.num_attack_samples = args.num_attack_samples
    if args.num_robustness_samples:
        config.robustness.num_samples = args.num_robustness_samples
    if args.num_scaffolding_samples:
        config.scaffolding.num_eval_samples = args.num_scaffolding_samples
    config.seed = args.seed

    datasets = args.datasets or config.datasets
    phases = args.phase

    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger.info("Experiment 2: Adversarial Robustness and Explanation-Aware Attacks")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Phases: {phases}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Exp1 models: {config.exp1_results_dir}")
    logger.info(f"Attack samples: {config.attack.num_attack_samples}")
    logger.info(f"Robustness samples: {config.robustness.num_samples}")

    run_experiment(config, datasets, phases)


if __name__ == "__main__":
    main()
