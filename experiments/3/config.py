"""Configuration for Experiment 3: Feature Interaction, Consensus, and Expert Alignment."""

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
    """Configuration for expert alignment scoring."""

    enabled: bool = True
    top_k_rra: list[int] = field(default_factory=lambda: [5, 10, 15])


@dataclass
class ParallelismConfig:
    """Configuration for parallel execution."""

    max_dataset_workers: int = 4
    max_consensus_workers: int = 16
    max_plot_workers: int = 8
    num_gpu_devices: int = 2


@dataclass
class Experiment3Config:
    data: DataConfig = field(default_factory=DataConfig)
    dnn: DNNConfig = field(default_factory=DNNConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    output_dir: Path = Path("experiments/3/results")
    exp1_output_dir: Path = Path("experiments/1/results")
    xai_modes: list[str] = field(default_factory=lambda: ["normal", "pa"])
    seed: int = 42

    ALL_DATASETS: list[str] = field(
        default_factory=lambda: ["nsl-kdd", "cic-ids-2017", "unsw-nb15", "cse-cic-ids2018"]
    )


def load_experiment3_config(yaml_path: str | Path | None = None) -> Experiment3Config:
    """Load Experiment3Config from YAML, falling back to dataclass defaults.

    Args:
        yaml_path: Path to shared YAML config. None uses default location.

    Returns:
        Populated Experiment3Config.
    """
    import sys
    loader_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "commons")
    if loader_dir not in sys.path:
        sys.path.insert(0, loader_dir)
    from config_loader import load_yaml_config

    try:
        raw = load_yaml_config(yaml_path)
    except FileNotFoundError:
        return Experiment3Config()

    config = Experiment3Config()
    config.seed = raw.get("seed", config.seed)

    # Data config
    data_raw = raw.get("data", {})
    if "data_root" in data_raw:
        config.data.data_root = Path(data_raw["data_root"])
    if "test_size" in data_raw:
        config.data.test_size = data_raw["test_size"]
    if "val_split" in data_raw:
        config.data.val_split = data_raw["val_split"]
    if "apply_smote" in data_raw:
        config.data.apply_smote = data_raw["apply_smote"]
    config.data.random_state = config.seed

    # Model configs
    models_raw = raw.get("models", {})
    dnn_raw = models_raw.get("dnn", {})
    for k, v in dnn_raw.items():
        if hasattr(config.dnn, k):
            setattr(config.dnn, k, v)
    rf_raw = models_raw.get("rf", {})
    for k, v in rf_raw.items():
        if hasattr(config.rf, k):
            setattr(config.rf, k, v)

    # Explainer config
    exp_raw = raw.get("explainer", {})
    for k, v in exp_raw.items():
        if hasattr(config.explainer, k):
            setattr(config.explainer, k, v)

    # Experiment 3 specific
    exp3_raw = raw.get("experiment_3", {})
    if "output_dir" in exp3_raw:
        config.output_dir = Path(exp3_raw["output_dir"])
    if "xai_modes" in exp3_raw:
        config.xai_modes = exp3_raw["xai_modes"]

    # Exp1 output dir (for reuse)
    exp1_raw = raw.get("experiment_1", {})
    if "output_dir" in exp1_raw:
        config.exp1_output_dir = Path(exp1_raw["output_dir"])

    # Consensus
    cons_raw = exp3_raw.get("consensus", {})
    for k, v in cons_raw.items():
        if hasattr(config.consensus, k):
            setattr(config.consensus, k, v)

    # Interactions
    inter_raw = exp3_raw.get("interactions", {})
    for k, v in inter_raw.items():
        if hasattr(config.interaction, k):
            setattr(config.interaction, k, v)

    # Alignment
    align_raw = exp3_raw.get("alignment", {})
    for k, v in align_raw.items():
        if hasattr(config.alignment, k):
            setattr(config.alignment, k, v)

    # Parallelism
    par_raw = exp3_raw.get("parallelism", {})
    for k, v in par_raw.items():
        if hasattr(config.parallelism, k):
            setattr(config.parallelism, k, v)

    # Datasets
    if "datasets" in raw:
        config.ALL_DATASETS = raw["datasets"]

    return config
