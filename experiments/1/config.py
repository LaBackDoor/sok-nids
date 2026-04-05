"""Configuration for Experiment 1: Quantitative Benchmarking of Explanation Quality.

Dataclass defaults serve as the canonical schema.  If ``config.yaml`` exists
next to this file the values in it **override** the defaults — so you can
tweak parameters without touching Python.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Dataclass definitions (unchanged — used as types everywhere)
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    data_root: Path = Path("/home/resbears/Downloads/data")
    nsl_kdd_dir: str = "nsl-kdd"
    cic_ids_2017_dir: str = "cic-ids-2017"
    unsw_nb15_dir: str = "cic_unsw-nb15_augmented_dataset"
    cse_cic_ids2018_dir: str = "cse-cic-ids2018"

    test_size: float = 0.2
    val_split: float = 0.1  # fraction of training data reserved for validation
    random_state: int = 42
    apply_smote: bool = True


@dataclass
class DNNConfig:
    hidden_layers: list[int] = field(default_factory=lambda: [1024, 768, 512])
    dropout_rate: float = 0.01
    learning_rate: float = 0.01
    batch_size: int = 8192
    epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class RFConfig:
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 2
    criterion: str = "gini"
    n_jobs: int = -1


@dataclass
class XGBConfig:
    n_estimators: int = 300
    max_depth: int = 8
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    n_jobs: int = -1
    tree_method: str = "hist"
    device: str = "cuda"


@dataclass
class CNNLSTMConfig:
    grid_size: int = 11
    hidden_size: int = 128
    num_lstm_layers: int = 1
    dropout: float = 0.5
    learning_rate: float = 0.001
    batch_size: int = 4096
    epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class CNNGRUConfig:
    input_channels: int = 1
    cnn_filters: int = 64
    cnn_kernel_size: int = 3
    pool_kernel_size: int = 2
    cnn_dropout: float = 0.5
    gru_hidden_size: int = 75
    gru_num_layers: int = 1
    gru_dropout: float = 0.5
    gru_input_size: int = 1
    fc_hidden_size: int = 128
    input_spatial_size: int = 11
    seq_len: int = 121
    learning_rate: float = 0.001
    batch_size: int = 4096
    epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class ExplainerConfig:
    shap_background_samples: int = 100
    lime_num_features: int = 10
    lime_num_samples: int = 2000
    ig_n_steps: int = 50
    ig_internal_batch_size: int = 8192
    num_explain_samples: int = 10000
    cpu_fraction: float = 0.9
    # LIME stability tuning
    lime_auto_tune: bool = False
    lime_tune_candidates: list[int] = field(default_factory=lambda: [200, 500, 1000, 2000, 5000])
    lime_tune_n_repeats: int = 5
    lime_tune_stability_threshold: float = 0.85
    lime_tune_probe_samples: int = 10


@dataclass
class MetricConfig:
    faithfulness_k_values: list[int] = field(default_factory=lambda: [5, 10, 15])
    sparsity_thresholds: list[float] = field(
        default_factory=lambda: [i / 10 for i in range(11)]
    )
    efficiency_batch_size: int = 10000
    stability_runs: int = 3
    stability_top_k: int = 10
    completeness_num_corrupted: int = 500


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    dnn: DNNConfig = field(default_factory=DNNConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)
    cnn_lstm: CNNLSTMConfig = field(default_factory=CNNLSTMConfig)
    cnn_gru: CNNGRUConfig = field(default_factory=CNNGRUConfig)
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    output_dir: Path = Path("experiments/1/results/normal")
    models_dir: Path = Path("experiments/1/results")
    xai_mode: str = "n"
    seed: int = 42
    cpu_fraction: float = 0.9

    ALL_DATASETS: list[str] = field(
        default_factory=lambda: [
            "nsl-kdd",
            "cic-ids-2017",
            "unsw-nb15",
            "cse-cic-ids2018",
        ]
    )


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

_SECTION_TO_CLASS = {
    "data": DataConfig,
    "dnn": DNNConfig,
    "rf": RFConfig,
    "xgb": XGBConfig,
    "cnn_lstm": CNNLSTMConfig,
    "cnn_gru": CNNGRUConfig,
    "explainer": ExplainerConfig,
    "metric": MetricConfig,
}


def _apply_dict(obj, overrides: dict):
    """Set attributes on *obj* from *overrides*, converting Path fields."""
    for key, value in overrides.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if isinstance(current, Path):
            value = Path(value)
        setattr(obj, key, value)


def load_config(yaml_path: str | Path | None = None) -> ExperimentConfig:
    """Build an ExperimentConfig, optionally overlaying values from YAML.

    Resolution order:
      1. Dataclass defaults (this file).
      2. ``config.yaml`` next to this file (if it exists).
      3. Explicit *yaml_path* (if provided).
    """
    cfg = ExperimentConfig()

    # Auto-discover shared config.yaml in experiments/commons/
    default_yaml = Path(__file__).parent.parent / "commons" / "config.yaml"
    paths_to_try = [default_yaml]
    if yaml_path is not None:
        paths_to_try.append(Path(yaml_path))

    for path in paths_to_try:
        if not path.exists():
            continue
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Top-level scalars
        if "seed" in raw:
            cfg.seed = raw["seed"]
        if "datasets" in raw:
            cfg.ALL_DATASETS = raw["datasets"]
        if "output_dir" in raw:
            cfg.output_dir = Path(raw["output_dir"])
        if "models_dir" in raw:
            cfg.models_dir = Path(raw["models_dir"])
        if "cpu_fraction" in raw:
            cfg.cpu_fraction = float(raw["cpu_fraction"])

        # Nested sections → matching dataclass attribute
        for section, cls in _SECTION_TO_CLASS.items():
            if section in raw and isinstance(raw[section], dict):
                _apply_dict(getattr(cfg, section), raw[section])

        # Models are nested under "models:" in the shared YAML
        if "models" in raw and isinstance(raw["models"], dict):
            model_map = {"dnn": "dnn", "rf": "rf", "xgb": "xgb",
                         "cnn_lstm": "cnn_lstm", "cnn_gru": "cnn_gru"}
            for yaml_key, attr_name in model_map.items():
                if yaml_key in raw["models"] and isinstance(raw["models"][yaml_key], dict):
                    _apply_dict(getattr(cfg, attr_name), raw["models"][yaml_key])

        # Experiment-1-specific overrides (output_dir, models_dir)
        if "experiment_1" in raw and isinstance(raw["experiment_1"], dict):
            exp1 = raw["experiment_1"]
            if "output_dir" in exp1:
                cfg.output_dir = Path(exp1["output_dir"])
            if "models_dir" in exp1:
                cfg.models_dir = Path(exp1["models_dir"])

    # Propagate top-level cpu_fraction into ExplainerConfig so that
    # explainer code can read it without needing the full ExperimentConfig.
    cfg.explainer.cpu_fraction = cfg.cpu_fraction

    return cfg
