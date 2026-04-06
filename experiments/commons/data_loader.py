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


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """No-op: keep full float64/int64 precision (1TB RAM available)."""
    return df


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

# Standard 5-class mapping: attack type -> category
NSL_KDD_ATTACK_CATEGORY = {
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "mailbomb": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "named": "R2L", "sendmail": "R2L", "snmpgetattack": "R2L",
    "snmpguess": "R2L", "worm": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "httptunnel": "U2R", "ps": "U2R",
    "sqlattack": "U2R", "xterm": "U2R",
}


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
    # Replace inf in-place to avoid full copy
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mask = ~np.isfinite(df[col].values)
        if mask.any():
            df.loc[mask, col] = np.nan

    # Median imputation for numeric columns (per roadmap specification)
    n_missing = df[numeric_cols].isna().sum().sum()
    if n_missing > 0:
        logger.info(f"  Median-imputing {n_missing} missing numeric values")
        medians = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(medians)

    # Drop any remaining rows with NaN in non-numeric columns, then duplicates
    n_before = len(df)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    n_after = len(df)
    if n_before != n_after:
        logger.info(f"  Cleaned {n_before - n_after} rows (remaining NaN/duplicates)")
    df.reset_index(drop=True, inplace=True)
    return df


def _apply_smote_batched(
    X: np.ndarray, y: np.ndarray, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE one minority class at a time.

    Caps each class target at the *median* class count.  Classes already
    above the median are left alone.

    For each minority class we build a working set (minority samples +
    all other classes) and run SMOTE only on that subset.  Synthetic samples
    are collected and appended to the original data.
    """
    classes, counts = np.unique(y, return_counts=True)
    median_count = int(np.median(counts))
    target_count = median_count
    logger.info(
        f"  SMOTE target per class: {target_count} "
        f"(median of class counts; majority={int(counts.max())})"
    )

    synthetic_X_parts: list[np.ndarray] = []
    synthetic_y_parts: list[np.ndarray] = []

    for cls, cnt in zip(classes, counts):
        if cnt >= target_count:
            continue  # already at or above the target
        target = target_count

        # Minority samples for this class
        cls_mask = y == cls
        X_cls = X[cls_mask]

        # Background: all other classes (no subsampling)
        bg_mask = ~cls_mask
        X_bg = X[bg_mask]
        y_bg = y[bg_mask]

        # Combine into working set
        X_work = np.concatenate([X_cls, X_bg])
        y_work = np.concatenate([np.full(len(X_cls), cls), y_bg])

        k = min(5, cnt - 1) if cnt > 1 else 0
        if k < 1:
            logger.warning(
                f"  SMOTE: class {cls} has {cnt} sample(s), skipping"
            )
            continue

        try:
            smote = SMOTE(
                sampling_strategy={cls: target},
                random_state=random_state,
                k_neighbors=k,
            )
            X_res, y_res = smote.fit_resample(X_work, y_work)
            # Extract only the newly generated synthetic samples
            new_mask = np.arange(len(X_work), len(X_res))
            synthetic_X_parts.append(X_res[new_mask])
            synthetic_y_parts.append(y_res[new_mask])
            logger.info(
                f"  SMOTE: class {cls} ({cnt} -> {target}, "
                f"+{len(new_mask)} synthetic)"
            )
        except ValueError as e:
            logger.warning(f"  SMOTE failed for class {cls}: {e}")

    if synthetic_X_parts:
        X_out = np.concatenate([X] + synthetic_X_parts)
        y_out = np.concatenate([y] + synthetic_y_parts)
    else:
        X_out, y_out = X, y

    logger.info(f"  Post-SMOTE train shape: {X_out.shape}")
    return X_out, y_out


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

    df = None
    for f in csv_files:
        logger.info(f"  Reading {f.name}")
        chunk = pd.read_csv(f, encoding="utf-8", low_memory=False)
        chunk.columns = chunk.columns.str.strip()
        chunk = _downcast_numeric(chunk)
        if df is None:
            df = chunk
        else:
            df = pd.concat([df, chunk], ignore_index=True)
    logger.info(f"  Combined shape: {df.shape}")

    # Derive Protocol column (not present in MachineLearningCVE CSVs).
    # TCP=6: any TCP flag > 0 or Init_Win_bytes_forward >= 0.
    # ICMP=1: Destination Port == 0 and not TCP.
    # UDP=17: everything else.
    if "Protocol" not in df.columns:
        tcp_flag_cols = [
            "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
            "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
            "CWE Flag Count", "ECE Flag Count",
        ]
        is_tcp = (df[tcp_flag_cols] > 0).any(axis=1) | (df["Init_Win_bytes_forward"] >= 0)
        is_icmp = ~is_tcp & (df["Destination Port"] == 0)
        protocol = pd.Series(17, index=df.index, dtype="int8")  # default UDP
        protocol[is_tcp] = 6
        protocol[is_icmp] = 1
        # Insert after "Destination Port" to match schema ordering
        dst_pos = df.columns.get_loc("Destination Port") + 1
        df.insert(dst_pos, "Protocol", protocol)
        logger.info(f"  Derived Protocol column (TCP={is_tcp.sum()}, UDP={(~is_tcp & ~is_icmp).sum()}, ICMP={is_icmp.sum()})")

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
    """Load all CSE-CIC-IDS2018 processed CSV files.

    Returns None — this dataset is too large for a single DataFrame.
    Use _load_cse_cic_ids2018_streaming() instead.
    """
    raise NotImplementedError("Use streaming loader for CSE-CIC-IDS2018")


def _load_and_preprocess_cse_cic_ids2018_streaming(
    config: DataConfig, dataset_name: str
) -> DatasetBundle:
    """Load + preprocess CSE-CIC-IDS2018 in a streaming fashion.

    Each CSV is read, cleaned, and converted to numpy individually so we
    never hold the full 16M-row DataFrame in memory at once.
    """
    base = config.data_root / config.cse_cic_ids2018_dir / "Processed Traffic Data for ML Algorithms"
    logger.info(f"Loading CSE-CIC-IDS2018 (streaming) from {base}")

    csv_files = sorted(base.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {base}")

    # Metadata columns to drop
    meta_cols = {"Timestamp", "timestamp", "Flow ID", "Src IP", "Dst IP", "Src Port"}

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    feature_names = None

    for f in csv_files:
        logger.info(f"  Reading {f.name}")
        for ci, chunk in enumerate(
            [pd.read_csv(f, encoding="utf-8", low_memory=False)]
        ):
            chunk.columns = chunk.columns.str.strip()

            # Identify label column (first chunk only determines it)
            label_col = None
            for candidate in ["Label", "label", " Label"]:
                if candidate in chunk.columns:
                    label_col = candidate
                    break
            if label_col is None:
                raise ValueError(f"No label column in {f.name}")

            # Extract labels
            y_chunk = chunk[label_col].astype(str).str.strip().values

            # Drop label + metadata columns
            drop = [c for c in chunk.columns if c == label_col or c in meta_cols]
            chunk.drop(columns=drop, inplace=True)

            # Coerce to numeric and downcast
            for col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            chunk = _downcast_numeric(chunk)

            # Replace inf with NaN, then median-impute
            numeric_cols = chunk.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                vals = chunk[col].values
                inf_mask = ~np.isfinite(vals)
                if inf_mask.any():
                    chunk.loc[inf_mask, col] = np.nan

            n_missing = chunk.isna().sum().sum()
            if n_missing > 0:
                chunk.fillna(chunk.median(), inplace=True)

            # Drop rows still NaN (non-numeric remnants)
            valid_mask = chunk.notna().all(axis=1)
            chunk = chunk[valid_mask]
            y_chunk = y_chunk[valid_mask.values]

            if feature_names is None:
                feature_names = chunk.columns.tolist()

            X_parts.append(chunk.values.astype(np.float32))
            y_parts.append(y_chunk)

        logger.info(f"    Done: {sum(p.shape[0] for p in X_parts)} total rows so far")

    X = np.concatenate(X_parts)
    y_raw = np.concatenate(y_parts)
    logger.info(f"  Combined: {X.shape[0]} rows, {X.shape[1]} features")

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

    if config.apply_smote:
        logger.info("  Applying SMOTE (batched per-class)...")
        X_train, y_train = _apply_smote_batched(X_train, y_train, config.random_state)

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

    # Map 39 attack types to 5 classes (Normal, DoS, Probe, R2L, U2R)
    def _map_to_category(labels):
        mapped = []
        for lbl in labels:
            lbl_str = str(lbl).strip().lower()
            cat = NSL_KDD_ATTACK_CATEGORY.get(lbl_str)
            if cat is None:
                logger.warning(f"  Unknown NSL-KDD label '{lbl}', mapping to 'Normal'")
                cat = "Normal"
            mapped.append(cat)
        return np.array(mapped)

    y_train_raw = _map_to_category(y_train_raw)
    y_test_raw = _map_to_category(y_test_raw)
    logger.info(f"  Mapped to 5-class: {dict(zip(*np.unique(y_train_raw, return_counts=True)))}")
    logger.info(f"  Test distribution: {dict(zip(*np.unique(y_test_raw, return_counts=True)))}")

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

    if config.apply_smote:
        logger.info("  Applying SMOTE (batched per-class)...")
        X_train, y_train = _apply_smote_batched(X_train, y_train, config.random_state)

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
    X_df = df.drop(columns=drop_cols)

    # Coerce all feature columns to numeric
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    X_df = _downcast_numeric(X_df)

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

    if config.apply_smote:
        logger.info("  Applying SMOTE (batched per-class)...")
        X_train, y_train = _apply_smote_batched(X_train, y_train, config.random_state)

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
        bundle = _load_and_preprocess_cse_cic_ids2018_streaming(config, name)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    logger.info(
        f"  Train: {bundle.X_train.shape}, Test: {bundle.X_test.shape}, "
        f"Classes: {bundle.num_classes}"
    )
    return bundle
