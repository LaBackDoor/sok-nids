"""Configuration for Experiment 1: Quantitative Benchmarking of Explanation Quality."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_root: Path = Path("/mnt/c/Users/abanisenioluwa_oroj1/Downloads/data/data")
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
    lime_num_samples: int = 5000
    ig_n_steps: int = 50
    ig_internal_batch_size: int = 8192
    num_explain_samples: int = 10000


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

    ALL_DATASETS: list[str] = field(
        default_factory=lambda: [
            "nsl-kdd",
            "cic-ids-2017",
            "unsw-nb15",
            "cse-cic-ids2018",
        ]
    )
