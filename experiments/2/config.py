"""Configuration for Experiment 2: Adversarial Robustness and Explanation-Aware Attacks."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AttackConfig:
    """Parameters for FGSM and PGD adversarial attack generation."""

    fgsm_epsilons: list[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.3]
    )
    pgd_epsilons: list[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1]
    )
    pgd_num_steps: int = 20
    pgd_step_size_factor: float = 2.5  # step_size = factor * epsilon / num_steps
    num_attack_samples: int = 1000
    attack_batch_size: int = 4096
    run_constrained: bool = True
    run_unconstrained: bool = True


@dataclass
class ScaffoldingConfig:
    """Parameters for the Integrity Scaffolding Attack (M3 threat model)."""

    biased_feature: str | None = None  # auto-select via mutual information if None
    dummy_feature_name: str = "dummy_metadata_tag"
    ood_z_threshold: float = 3.0  # z-score threshold for OOD detection
    ood_ratio_threshold: float = 0.3  # fraction of OOD features to trigger
    num_eval_samples: int = 500
    biased_model_hidden: list[int] = field(default_factory=lambda: [64, 32])
    biased_model_epochs: int = 50
    biased_model_lr: float = 0.01
    lime_num_samples: int = 5000
    lime_num_features: int = 10
    shap_background_samples: int = 100
    scaffolding_datasets: list[str] = field(
        default_factory=lambda: ["nsl-kdd", "cic-ids-2017"]
    )


@dataclass
class RobustnessConfig:
    """Parameters for mathematical robustness metric evaluation."""

    num_samples: int = 1000
    explanation_methods: list[str] = field(
        default_factory=lambda: ["SHAP", "LIME", "IG", "DeepLIFT"]
    )
    pa_explanation_methods: list[str] = field(
        default_factory=lambda: ["PA-SHAP", "PA-LIME", "PA-IG", "PA-DeepLIFT"]
    )
    distance_norm: str = "l2"  # l2 or linf for Lipschitz computation
    explanation_similarity_epsilon: float = 0.1
    shap_background_samples: int = 100
    lime_num_samples: int = 5000
    lime_num_features: int = 10
    ig_n_steps: int = 50
    ig_internal_batch_size: int = 4096


@dataclass
class Experiment2Config:
    """Top-level configuration for Experiment 2."""

    attack: AttackConfig = field(default_factory=AttackConfig)
    scaffolding: ScaffoldingConfig = field(default_factory=ScaffoldingConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    exp1_results_dir: Path = Path("experiments/1/results")
    output_dir: Path = Path("experiments/2/results")
    seed: int = 42
    datasets: list[str] = field(
        default_factory=lambda: ["nsl-kdd", "cic-ids-2017", "unsw-nb15", "cse-cic-ids2018", "cic-iov-2024"]
    )
