"""Configuration for Experiment 4: XAI-Driven Dimensionality Reduction vs Statistical Baselines."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_root: Path = Path("./data")
    nsl_kdd_dir: str = "nsl-kdd"
    cic_ids_2017_dir: str = "cic-ids-2017"
    unsw_nb15_dir: str = "cic_unsw-nb15_augmented_dataset"
    cse_cic_ids2018_dir: str = "cse-cic-ids2018"
    test_size: float = 0.2
    val_split: float = 0.1
    random_state: int = 42
    apply_smote: bool = True


@dataclass
class DNNConfig:
    hidden_layers: list[int] = field(default_factory=lambda: [1024, 768, 512])
    dropout_rate: float = 0.01
    learning_rate: float = 0.01
    batch_size: int = 1024
    epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class CNNConfig:
    """1D-CNN for tabular NIDS data."""

    channels: list[int] = field(default_factory=lambda: [64, 128, 64])
    kernel_size: int = 3
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 1024
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
class SVMConfig:
    kernel: str = "rbf"
    C: float = 1.0
    gamma: str = "scale"
    max_iter: int = 5000
    probability: bool = True
    # SVM can't handle huge datasets efficiently; subsample for training
    max_train_samples: int = 50000


@dataclass
class StatisticalSelectionConfig:
    """Config for statistical feature selection methods."""

    pca_variance_threshold: float = 0.95  # cumulative variance to retain
    spearman_threshold: float = 0.8  # correlation threshold for redundancy removal
    info_gain_top_k: int = 20  # top features by information gain


@dataclass
class XAISelectionConfig:
    """Config for XAI-driven feature selection."""

    shap_background_samples: int = 100
    shap_explain_samples: int = 5000
    lime_num_samples: int = 5000
    lime_num_features: int = 10
    # Iterative pruning
    f1_degradation_threshold: float = 0.01  # stop if F1 drops > 1%
    min_features: int = 5  # never go below this
    pruning_step_ratio: float = 0.1  # remove 10% of remaining features each step
    # Target feature counts from roadmap (used as minimum bounds)
    target_features: dict = field(
        default_factory=lambda: {
            "nsl-kdd": 19,
            "cic-ids-2017": 15,
            "unsw-nb15": 15,
            "cse-cic-ids2018": 15,
        }
    )


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    dnn: DNNConfig = field(default_factory=DNNConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)
    statistical: StatisticalSelectionConfig = field(
        default_factory=StatisticalSelectionConfig
    )
    xai: XAISelectionConfig = field(default_factory=XAISelectionConfig)
    output_dir: Path = Path("experiments/4/results")
    exp1_output_dir: Path = Path("experiments/1/results")
    seed: int = 42

    ALL_DATASETS: list[str] = field(
        default_factory=lambda: [
            "nsl-kdd",
            "cic-ids-2017",
            "unsw-nb15",
            "cse-cic-ids2018",
        ]
    )
