"""Configuration for Experiment 3: Feature Interaction, Consensus, and Human-in-the-Loop Alignment."""

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

# Import shared configs from experiment 1 via importlib to avoid circular import
# (both exp1 and exp3 have a module named "config")
_exp1_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1", "config.py")
_spec = importlib.util.spec_from_file_location("exp1_config", _exp1_config_path)
_exp1_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp1_config)

DataConfig = _exp1_config.DataConfig
DNNConfig = _exp1_config.DNNConfig
RFConfig = _exp1_config.RFConfig
ExplainerConfig = _exp1_config.ExplainerConfig


@dataclass
class ConsensusConfig:
    """Configuration for explainer consensus analysis."""

    top_k_values: list[int] = field(default_factory=lambda: [5, 10])
    num_explain_samples: int = 10000
    alpha: float = 0.05  # Wilcoxon significance threshold


@dataclass
class InteractionConfig:
    """Configuration for feature interaction analysis."""

    top_n_interactions: int = 20
    shap_interaction_samples: int = 500  # SHAP interaction values are O(n * f^2)
    shap_background_samples: int = 100


@dataclass
class AlignmentConfig:
    """Configuration for human-in-the-loop alignment scoring."""

    top_k_rra: list[int] = field(default_factory=lambda: [5, 10, 15])


@dataclass
class Experiment3Config:
    data: DataConfig = field(default_factory=DataConfig)
    dnn: DNNConfig = field(default_factory=DNNConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    output_dir: Path = Path("experiments/3/results")
    exp1_output_dir: Path = Path("experiments/1/results")
    seed: int = 42

    ALL_DATASETS: list[str] = field(
        default_factory=lambda: ["nsl-kdd", "cic-ids-2017", "unsw-nb15", "cse-cic-ids2018"]
    )
