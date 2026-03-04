"""Data loading and preprocessing for all 4 NIDS benchmark datasets."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from config import DataConfig

logger = logging.getLogger(__name__)

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]

NSL_KDD_CATEGORICAL = ["protocol_type", "service", "flag"]


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    label_encoder: LabelEncoder
    scaler: MinMaxScaler
    dataset_name: str
    num_classes: int


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute NaN/Inf in numeric columns, drop duplicates."""
    df = df.replace([np.inf, -np.inf], np.nan)

    # Median imputation for numeric columns (per roadmap specification)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_missing = df[numeric_cols].isna().sum().sum()
    if n_missing > 0:
        logger.info(f"  Median-imputing {n_missing} missing numeric values")
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Drop any remaining rows with NaN in non-numeric columns
    n_before = len(df)
    df = df.dropna()
    df = df.drop_duplicates()
    n_after = len(df)
    if n_before != n_after:
        logger.info(f"  Cleaned {n_before - n_after} rows (remaining NaN/duplicates)")
    return df.reset_index(drop=True)


def _load_nsl_kdd(config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load NSL-KDD train/test splits with predefined column names."""
    base = config.data_root / config.nsl_kdd_dir
    train_path = base / "KDDTrain+.txt"
    test_path = base / "KDDTest+.txt"

    logger.info(f"Loading NSL-KDD from {base}")
    df_train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    df_test = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)

    # Drop difficulty_level column
    df_train = df_train.drop(columns=["difficulty_level"])
    df_test = df_test.drop(columns=["difficulty_level"])

    return df_train, df_test


def _load_cic_ids_2017(config: DataConfig) -> pd.DataFrame:
    """Load and combine all CIC-IDS-2017 CSV files."""
    base = config.data_root / config.cic_ids_2017_dir / "csv" / "MachineLearningCVE"
    logger.info(f"Loading CIC-IDS-2017 from {base}")

    csv_files = sorted(base.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {base}")

    frames = []
    for f in csv_files:
        logger.info(f"  Reading {f.name}")
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        # Strip leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"  Combined shape: {df.shape}")
    return df


def _load_unsw_nb15(config: DataConfig) -> pd.DataFrame:
    """Load UNSW-NB15 augmented dataset (Data.csv + Label.csv)."""
    base = config.data_root / config.unsw_nb15_dir
    logger.info(f"Loading UNSW-NB15 from {base}")

    df_data = pd.read_csv(base / "Data.csv", low_memory=False)
    df_labels = pd.read_csv(base / "Label.csv")

    df_data["Label"] = df_labels["Label"]
    logger.info(f"  Shape: {df_data.shape}")
    return df_data


def _load_cse_cic_ids2018(config: DataConfig) -> pd.DataFrame:
    """Load all CSE-CIC-IDS2018 processed CSV files."""
    base = config.data_root / config.cse_cic_ids2018_dir / "Processed Traffic Data for ML Algorithms"
    logger.info(f"Loading CSE-CIC-IDS2018 from {base}")

    csv_files = sorted(base.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {base}")

    frames = []
    for f in csv_files:
        logger.info(f"  Reading {f.name}")
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"  Combined shape: {df.shape}")
    return df


def _preprocess_nsl_kdd(
    df_train: pd.DataFrame, df_test: pd.DataFrame, config: DataConfig
) -> DatasetBundle:
    """Preprocess NSL-KDD: one-hot encode categoricals, scale, SMOTE."""
    # One-hot encode categorical features
    df_train = pd.get_dummies(df_train, columns=NSL_KDD_CATEGORICAL)
    df_test = pd.get_dummies(df_test, columns=NSL_KDD_CATEGORICAL)

    # Align columns (test might have categories not in train and vice versa)
    train_cols = set(df_train.columns) - {"label"}
    test_cols = set(df_test.columns) - {"label"}
    all_cols = sorted(train_cols | test_cols)

    for col in all_cols:
        if col not in df_train.columns:
            df_train[col] = 0
        if col not in df_test.columns:
            df_test[col] = 0

    feature_names = [c for c in all_cols if c != "label"]

    X_train = df_train[feature_names].values.astype(np.float32)
    X_test = df_test[feature_names].values.astype(np.float32)
    y_train_raw = df_train["label"].values
    y_test_raw = df_test["label"].values

    # Encode labels
    le = LabelEncoder()
    le.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    # Split off validation set from training data (avoid test set leakage)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config.val_split,
        random_state=config.random_state, stratify=y_train,
    )
    logger.info(f"  Train/Val split: {X_train.shape[0]} train, {X_val.shape[0]} val")

    # Scale features (fit on train only)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # SMOTE on training data only
    if config.apply_smote:
        logger.info("  Applying SMOTE...")
        min_count = min(np.bincount(y_train))
        k = min(5, min_count - 1) if min_count > 1 else 0
        if k < 1:
            logger.warning(f"  Skipping SMOTE: min class has {min_count} sample(s)")
        else:
            try:
                smote = SMOTE(random_state=config.random_state, k_neighbors=k)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logger.info(f"  Post-SMOTE train shape: {X_train.shape}")
            except ValueError as e:
                logger.warning(f"  SMOTE failed: {e}. Proceeding without SMOTE.")

    return DatasetBundle(
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        label_encoder=le,
        scaler=scaler,
        dataset_name="nsl-kdd",
        num_classes=len(le.classes_),
    )


def _preprocess_flow_dataset(
    df: pd.DataFrame, dataset_name: str, config: DataConfig
) -> DatasetBundle:
    """Generic preprocessing for CICFlowMeter-style datasets."""
    # Identify label column
    label_col = None
    for candidate in ["Label", "label", " Label"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError(f"Could not find label column in {dataset_name}. Columns: {df.columns.tolist()}")

    # Drop non-feature columns
    drop_cols = [label_col]
    for col in ["Timestamp", "timestamp", "Flow ID", "Src IP", "Dst IP", "Src Port"]:
        if col in df.columns:
            drop_cols.append(col)

    y_raw = df[label_col].astype(str).str.strip()

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[feature_cols].copy()

    # Coerce all feature columns to numeric
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    # Combine X and y for cleaning
    X_df["__label__"] = y_raw.values
    X_df = _clean_dataframe(X_df)
    y_raw = X_df["__label__"].values
    X_df = X_df.drop(columns=["__label__"])
    feature_names = X_df.columns.tolist()

    X = X_df.values.astype(np.float32)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    logger.info(f"  Classes ({len(le.classes_)}): {le.classes_[:10]}...")

    # Train/test split, then train/val split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=config.val_split,
        random_state=config.random_state, stratify=y_trainval,
    )
    logger.info(f"  Train/Val/Test split: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    # Scale (fit on train only)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # SMOTE on training data only
    if config.apply_smote:
        logger.info("  Applying SMOTE...")
        min_class_count = min(np.bincount(y_train))
        k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
        if k_neighbors < 1:
            logger.warning("  Cannot apply SMOTE (class with single sample). Skipping.")
        else:
            try:
                smote = SMOTE(random_state=config.random_state, k_neighbors=k_neighbors)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logger.info(f"  Post-SMOTE train shape: {X_train.shape}")
            except ValueError as e:
                logger.warning(f"  SMOTE failed: {e}. Proceeding without SMOTE.")

    return DatasetBundle(
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        label_encoder=le,
        scaler=scaler,
        dataset_name=dataset_name,
        num_classes=len(le.classes_),
    )


def load_dataset(name: str, config: DataConfig) -> DatasetBundle:
    """Load and preprocess a single dataset by name."""
    logger.info(f"=== Loading dataset: {name} ===")

    if name == "nsl-kdd":
        df_train, df_test = _load_nsl_kdd(config)
        df_train = _clean_dataframe(df_train)
        df_test = _clean_dataframe(df_test)
        bundle = _preprocess_nsl_kdd(df_train, df_test, config)

    elif name == "cic-ids-2017":
        df = _load_cic_ids_2017(config)
        bundle = _preprocess_flow_dataset(df, name, config)

    elif name == "unsw-nb15":
        df = _load_unsw_nb15(config)
        bundle = _preprocess_flow_dataset(df, name, config)

    elif name == "cse-cic-ids2018":
        df = _load_cse_cic_ids2018(config)
        bundle = _preprocess_flow_dataset(df, name, config)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    logger.info(
        f"  Train: {bundle.X_train.shape}, Test: {bundle.X_test.shape}, "
        f"Classes: {bundle.num_classes}"
    )
    return bundle
