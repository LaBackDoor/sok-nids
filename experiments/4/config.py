"""Configuration for Experiment 4: XAI-Driven Dimensionality Reduction vs Statistical Baselines."""

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Import shared configs from experiment 1 via importlib to avoid circular import
# (both exp1 and exp4 have a module named "config")
_exp1_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1", "config.py")
_spec = importlib.util.spec_from_file_location("exp1_config", _exp1_config_path)
_exp1_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp1_config)

DataConfig = _exp1_config.DataConfig
DNNConfig = _exp1_config.DNNConfig
RFConfig = _exp1_config.RFConfig
ExplainerConfig = _exp1_config.ExplainerConfig


@dataclass
class CNNConfig:
    """1D-CNN for tabular NIDS data."""
    channels: list[int] = field(default_factory=lambda: [64, 128, 64])
    kernel_size: int = 3
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 8192
    epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class SVMConfig:
    kernel: str = "rbf"
    C: float = 1.0
    gamma: str = "scale"
    max_iter: int = 5000
    probability: bool = True
    max_train_samples: int | None = None


@dataclass
class StatisticalSelectionConfig:
    """Config for statistical feature selection methods."""
    pca_variance_threshold: float = 0.95
    spearman_threshold: float = 0.8
    info_gain_top_k: int = 20


@dataclass
class XAISelectionConfig:
    """Config for XAI-driven feature selection (iterative pruning parameters)."""
    f1_degradation_threshold: float = 0.01
    min_features: int = 5
    pruning_step_ratio: float = 0.1
    subsample_fraction: float = 0.2  # fraction of training data for pruning loop
    subsample_max: int = 50000  # cap subsampled rows


@dataclass
class ParallelismConfig:
    num_gpu_devices: int = 2


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    dnn: DNNConfig = field(default_factory=DNNConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    statistical: StatisticalSelectionConfig = field(default_factory=StatisticalSelectionConfig)
    xai: XAISelectionConfig = field(default_factory=XAISelectionConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    output_dir: Path = Path("experiments/4/results")
    exp1_output_dir: Path = Path("experiments/1/results")
    xai_modes: list[str] = field(default_factory=lambda: ["normal", "pa"])
    seed: int = 42

    ALL_DATASETS: list[str] = field(
        default_factory=lambda: [
            "nsl-kdd",
            "cic-ids-2017",
            "unsw-nb15",
            "cse-cic-ids2018",
        ]
    )


def load_experiment_config(yaml_path: str | Path | None = None) -> ExperimentConfig:
    """Load ExperimentConfig from YAML, falling back to dataclass defaults.

    Args:
        yaml_path: Path to shared YAML config. None uses default location.

    Returns:
        Populated ExperimentConfig.
    """
    loader_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "commons")
    if loader_dir not in sys.path:
        sys.path.insert(0, loader_dir)
    from config_loader import load_yaml_config

    try:
        raw = load_yaml_config(yaml_path)
    except FileNotFoundError:
        return ExperimentConfig()

    config = ExperimentConfig()
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

    # Shared model configs (DNN, RF)
    models_raw = raw.get("models", {})
    dnn_raw = models_raw.get("dnn", {})
    for k, v in dnn_raw.items():
        if hasattr(config.dnn, k):
            setattr(config.dnn, k, v)
    rf_raw = models_raw.get("rf", {})
    for k, v in rf_raw.items():
        if hasattr(config.rf, k):
            setattr(config.rf, k, v)

    # Explainer config (shared SHAP/LIME/IG/DeepLIFT params)
    exp_raw = raw.get("explainer", {})
    for k, v in exp_raw.items():
        if hasattr(config.explainer, k):
            setattr(config.explainer, k, v)

    # Experiment 4 specific
    exp4_raw = raw.get("experiment_4", {})
    if "output_dir" in exp4_raw:
        config.output_dir = Path(exp4_raw["output_dir"])
    if "xai_modes" in exp4_raw:
        config.xai_modes = exp4_raw["xai_modes"]

    # Exp1 output dir
    exp1_raw = raw.get("experiment_1", {})
    if "output_dir" in exp1_raw:
        config.exp1_output_dir = Path(exp1_raw["output_dir"])
    if "exp1_output_dir" in exp4_raw:
        config.exp1_output_dir = Path(exp4_raw["exp1_output_dir"])

    # CNN config (exp4-specific)
    cnn_raw = exp4_raw.get("cnn", {})
    for k, v in cnn_raw.items():
        if hasattr(config.cnn, k):
            setattr(config.cnn, k, v)

    # SVM config (exp4-specific)
    svm_raw = exp4_raw.get("svm", {})
    for k, v in svm_raw.items():
        if hasattr(config.svm, k):
            setattr(config.svm, k, v)

    # Statistical selection config
    stat_raw = exp4_raw.get("statistical", {})
    for k, v in stat_raw.items():
        if hasattr(config.statistical, k):
            setattr(config.statistical, k, v)

    # XAI selection config (pruning params)
    xai_raw = exp4_raw.get("xai", {})
    for k, v in xai_raw.items():
        if hasattr(config.xai, k):
            setattr(config.xai, k, v)

    # Parallelism
    par_raw = exp4_raw.get("parallelism", {})
    for k, v in par_raw.items():
        if hasattr(config.parallelism, k):
            setattr(config.parallelism, k, v)

    # Datasets
    if "datasets" in raw:
        config.ALL_DATASETS = raw["datasets"]

    return config
