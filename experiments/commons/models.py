"""DNN, RF, XGBoost, CNN-LSTM, and CNN-GRU model training for NIDS classification."""

import json
import logging
import math
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

# Import config from the same directory as this file (commons/), not from
# whatever config.py happens to be on sys.path. This avoids conflicts when
# exp3's config.py shadows exp1's on sys.path.
import importlib.util as _ilu
import os as _os
_config_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "1", "config.py")
_spec = _ilu.spec_from_file_location("_exp1_config_for_models", _config_path)
_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
CNNGRUConfig = _cfg.CNNGRUConfig
CNNLSTMConfig = _cfg.CNNLSTMConfig
DNNConfig = _cfg.DNNConfig
RFConfig = _cfg.RFConfig
XGBConfig = _cfg.XGBConfig

from data_loader import DatasetBundle

logger = logging.getLogger(__name__)

# Enable cuDNN auto-tuner for optimal convolution algorithms
torch.backends.cudnn.benchmark = True


class LazyGridDataset(Dataset):
    """Reshapes flat features to a grid on-the-fly to avoid pre-materialising
    the full reshaped array.  Keeps only the original numpy array in memory."""

    def __init__(self, X: np.ndarray, y: np.ndarray, grid_size: int):
        total = grid_size * grid_size
        if X.shape[1] < total:
            X = np.pad(X, ((0, 0), (0, total - X.shape[1])))
        elif X.shape[1] > total:
            X = X[:, :total]
        self.X = X
        self.y = y
        self.grid_size = grid_size

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(
            self.X[idx].reshape(1, self.grid_size, self.grid_size)
        ).float()
        return x, torch.tensor(self.y[idx], dtype=torch.long)


class LazyDualGridDataset(Dataset):
    """Lazy dataset for CNN-GRU: returns (spatial, temporal, label) on-the-fly."""

    def __init__(self, X: np.ndarray, y: np.ndarray, grid_size: int):
        total = grid_size * grid_size
        if X.shape[1] < total:
            X = np.pad(X, ((0, 0), (0, total - X.shape[1])))
        elif X.shape[1] > total:
            X = X[:, :total]
        self.X = X
        self.y = y
        self.grid_size = grid_size
        self.total = total

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row = self.X[idx]
        spatial = torch.from_numpy(
            row.reshape(1, self.grid_size, self.grid_size)
        ).float()
        temporal = torch.from_numpy(
            row.reshape(self.total, 1)
        ).float()
        return spatial, temporal, torch.tensor(self.y[idx], dtype=torch.long)

# Suppress cuBLAS context initialization warning (context is set successfully, warning is cosmetic)
warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS, but there was no current CUDA context.*")


class NIDSNet(nn.Module):
    """Fully connected DNN for network intrusion detection."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.01,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [1024, 768, 512]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SoftmaxModel(nn.Module):
    """Wraps a logit-outputting model to output softmax probabilities.

    Used for SHAP/Captum explainers so attributions are on probability space
    per the roadmap's Softmax output specification.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)


class DNNWrapper:
    """Wrapper to provide a unified predict/predict_proba interface for the DNN."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        # Unwrap DataParallel if needed for explainers
        self.base_model = model.module if isinstance(model, nn.DataParallel) else model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class RFWrapper:
    """Wrapper to provide a unified interface for the Random Forest."""

    def __init__(self, model: RandomForestClassifier, num_classes: int | None = None):
        self.model = model
        self.num_classes = num_classes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(X)
        # Pad to full num_classes if the model only saw a subset of classes during training
        if self.num_classes is not None and proba.shape[1] < self.num_classes:
            full_proba = np.zeros((proba.shape[0], self.num_classes), dtype=proba.dtype)
            full_proba[:, self.model.classes_] = proba
            return full_proba
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class XGBWrapper:
    """Wrapper to provide a unified interface for XGBoost.

    Handles label remapping: XGBoost requires contiguous [0, N) labels,
    but datasets may have non-contiguous labels after encoding/SMOTE.
    """

    def __init__(self, model, num_classes: int | None = None, label_map: np.ndarray | None = None):
        self.model = model
        self.num_classes = num_classes
        # label_map[compact_label] = original_label
        self.label_map = label_map

    def _gpu_dmatrix(self, X):
        """Create an xgb.DMatrix on GPU from numpy/cupy input."""
        import cupy as cp
        import xgboost as xgb
        X_gpu = cp.asarray(X) if isinstance(X, np.ndarray) else X
        return xgb.DMatrix(X_gpu)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dm = self._gpu_dmatrix(X)
        raw = self.model.get_booster().predict(dm)
        # binary:logistic → (n,) with P(class 1); multi:softprob → (n, K)
        if raw.ndim == 1:
            proba = np.column_stack([1.0 - raw, raw])
        else:
            proba = raw
        if self.num_classes is not None and proba.shape[1] < self.num_classes:
            full_proba = np.zeros((proba.shape[0], self.num_classes), dtype=proba.dtype)
            if self.label_map is not None:
                for compact_idx, orig_idx in enumerate(self.label_map):
                    full_proba[:, orig_idx] = proba[:, compact_idx]
            else:
                full_proba[:, :proba.shape[1]] = proba
            return full_proba
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        dm = self._gpu_dmatrix(X)
        raw = self.model.get_booster().predict(dm)
        # binary:logistic → (n,) floats; multi:softprob → (n, K)
        if raw.ndim == 1:
            preds = (raw > 0.5).astype(np.intp)
        else:
            preds = np.argmax(raw, axis=1).astype(np.intp)
        if self.label_map is not None:
            return self.label_map[preds]
        return preds


def _evaluate_model(wrapper, X_test: np.ndarray, y_test: np.ndarray, num_classes: int) -> dict:
    """Compute classification metrics."""
    y_pred = wrapper.predict(X_test)
    y_proba = wrapper.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    try:
        unique_classes = np.unique(y_test)
        if len(unique_classes) < 2:
            auc = float("nan")
        elif num_classes == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            # Manual OVR: only compute AUC for classes with both pos and neg samples
            per_class_auc = []
            per_class_weight = []
            for c in range(num_classes):
                binary_true = (y_test == c).astype(int)
                n_pos = binary_true.sum()
                n_neg = len(binary_true) - n_pos
                if n_pos == 0 or n_neg == 0:
                    continue
                auc_c = roc_auc_score(binary_true, y_proba[:, c])
                per_class_auc.append(auc_c)
                per_class_weight.append(n_pos)
            if per_class_auc:
                weights = np.array(per_class_weight, dtype=float)
                auc = float(np.average(per_class_auc, weights=weights))
            else:
                auc = float("nan")
    except ValueError:
        auc = float("nan")

    return {"accuracy": acc, "f1_weighted": f1, "auc_roc": auc}


def train_dnn(
    dataset: DatasetBundle,
    config: DNNConfig,
    device: torch.device,
    num_gpus: int = 1,
) -> tuple[NIDSNet, DNNWrapper, dict]:
    """Train DNN with multi-GPU DataParallel and AMP (mixed precision) support."""
    logger.info(f"Training DNN on {dataset.dataset_name} ({dataset.X_train.shape})")

    input_dim = dataset.X_train.shape[1]
    model = NIDSNet(
        input_dim=input_dim,
        num_classes=dataset.num_classes,
        hidden_layers=config.hidden_layers,
        dropout_rate=config.dropout_rate,
    )

    # Multi-GPU
    if num_gpus > 1 and torch.cuda.device_count() > 1:
        logger.info(f"  Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Warmup CUDA context to avoid cuBLAS initialization warning on first forward pass
    if device.type == "cuda":
        with torch.no_grad():
            model(torch.zeros(1, input_dim, device=device))

    use_amp = device.type == "cuda"
    if use_amp:
        logger.info("  Using Automatic Mixed Precision (AMP) training")

    # Dataloaders - use validation set (not test set) for early stopping
    train_ds = TensorDataset(
        torch.tensor(dataset.X_train, dtype=torch.float32),
        torch.tensor(dataset.y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(dataset.X_val, dtype=torch.float32),
        torch.tensor(dataset.y_val, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    train_start = time.time()

    for epoch in range(config.epochs):
        # Training with AMP
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast("cuda", enabled=use_amp):
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(val_ds)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch + 1}/{config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"  Early stopping at epoch {epoch + 1}")
                break

    train_time = time.time() - train_start
    logger.info(f"  DNN training completed in {train_time:.1f}s")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    # Re-warmup CUDA context after restoring best state (saved on CPU)
    if device.type == "cuda":
        with torch.no_grad():
            base = model.module if isinstance(model, nn.DataParallel) else model
            base(torch.zeros(1, input_dim, device=device))

    wrapper = DNNWrapper(model, device)
    metrics = _evaluate_model(wrapper, dataset.X_test, dataset.y_test, dataset.num_classes)
    metrics["train_time_seconds"] = train_time
    logger.info(f"  DNN Metrics: {metrics}")

    return model, wrapper, metrics


def train_rf(
    dataset: DatasetBundle, config: RFConfig
) -> tuple[RandomForestClassifier, RFWrapper, dict]:
    """Train Random Forest classifier."""
    logger.info(f"Training RF on {dataset.dataset_name} ({dataset.X_train.shape})")

    # For large datasets, limit parallelism to avoid OOM from per-worker
    # sorting buffers, and subsample rows per tree via max_samples.
    n_train = dataset.X_train.shape[0]
    n_jobs = config.n_jobs
    kwargs = {}
    if n_train > 5_000_000:
        n_jobs = min(n_jobs, 2) if n_jobs > 0 else 2
        kwargs["max_samples"] = min(2_000_000, n_train)
        logger.info(
            f"  Large dataset ({n_train} rows): n_jobs={n_jobs}, "
            f"max_samples={kwargs['max_samples']}"
        )

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        criterion=config.criterion,
        n_jobs=n_jobs,
        random_state=42,
        **kwargs,
    )

    train_start = time.time()
    model.fit(dataset.X_train, dataset.y_train)
    train_time = time.time() - train_start
    logger.info(f"  RF training completed in {train_time:.1f}s")

    wrapper = RFWrapper(model, num_classes=dataset.num_classes)
    metrics = _evaluate_model(wrapper, dataset.X_test, dataset.y_test, dataset.num_classes)
    metrics["train_time_seconds"] = train_time
    logger.info(f"  RF Metrics: {metrics}")

    return model, wrapper, metrics


def train_xgb(
    dataset: DatasetBundle, config: XGBConfig
) -> tuple:
    """Train XGBoost classifier."""
    import xgboost as xgb

    logger.info(f"Training XGBoost on {dataset.dataset_name} ({dataset.X_train.shape})")

    # XGBoost requires y_train labels to be contiguous [0, N).
    # Some classes may be absent from y_train after splitting, causing gaps.
    train_labels = np.unique(dataset.y_train)
    expected = np.arange(len(train_labels))
    needs_remap = not np.array_equal(train_labels, expected)

    if needs_remap:
        # Remap based on classes present in training data
        label_map = train_labels  # label_map[compact] = original
        inv_map = {int(orig): compact for compact, orig in enumerate(train_labels)}
        y_train = np.array([inv_map[y] for y in dataset.y_train])
        # For val/test, map unknown classes to -1 (will be filtered or ignored)
        y_val = np.array([inv_map.get(int(y), 0) for y in dataset.y_val])
        y_test_remapped = np.array([inv_map.get(int(y), -1) for y in dataset.y_test])
        logger.info(
            f"  Remapped {len(train_labels)} train labels to [0, {len(train_labels)})"
            f" (dataset has {dataset.num_classes} total classes)"
        )
    else:
        label_map = None
        y_train = dataset.y_train
        y_val = dataset.y_val
        y_test_remapped = None

    n_classes_train = len(np.unique(y_train))
    if n_classes_train == 2:
        objective = "binary:logistic"
        eval_metric = "logloss"
    else:
        objective = "multi:softprob"
        eval_metric = "mlogloss"

    model = xgb.XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        n_jobs=config.n_jobs,
        tree_method=config.tree_method,
        device=config.device,
        random_state=42,
        objective=objective,
        eval_metric=eval_metric,
        # XGBoost 3.x defaults to multi_output_tree which stores per-class
        # vector leaf values.  SHAP TreeExplainer cannot parse these string
        # vectors and raises "could not convert string to float".  Using the
        # classic one-output-per-tree format keeps SHAP compatibility.
        multi_strategy="one_output_per_tree",
    )

    train_start = time.time()
    model.fit(
        dataset.X_train, y_train,
        eval_set=[(dataset.X_val, y_val)],
        verbose=False,
    )
    train_time = time.time() - train_start
    logger.info(f"  XGBoost training completed in {train_time:.1f}s")

    # Keep model on GPU — XGBWrapper moves inputs to GPU via cupy.
    # set_params(device="cpu") is NOT called so inference stays on GPU.

    wrapper = XGBWrapper(model, num_classes=dataset.num_classes, label_map=label_map)

    # Evaluate only on test samples whose classes were seen during training
    if y_test_remapped is not None:
        known_mask = y_test_remapped >= 0
        n_unknown = (~known_mask).sum()
        if n_unknown > 0:
            logger.info(f"  {n_unknown} test samples have classes unseen in training, excluded from eval")
        X_test_eval = dataset.X_test[known_mask]
        y_test_eval = dataset.y_test[known_mask]
    else:
        X_test_eval = dataset.X_test
        y_test_eval = dataset.y_test

    metrics = _evaluate_model(wrapper, X_test_eval, y_test_eval, dataset.num_classes)
    metrics["train_time_seconds"] = train_time
    logger.info(f"  XGBoost Metrics: {metrics}")

    return model, wrapper, metrics


def save_models(
    dnn_model: nn.Module | None,
    rf_model: RandomForestClassifier | None,
    output_dir: Path,
    dataset_name: str,
    xgb_model=None,
    xgb_label_map: np.ndarray | None = None,
) -> None:
    """Save trained models to disk."""
    import joblib

    model_dir = output_dir / "models" / dataset_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save DNN
    if dnn_model is not None:
        base = dnn_model.module if isinstance(dnn_model, nn.DataParallel) else dnn_model
        torch.save(base.state_dict(), model_dir / "dnn.pt")

    # Save RF
    if rf_model is not None:
        joblib.dump(rf_model, model_dir / "rf.joblib")

    # Save XGBoost
    if xgb_model is not None:
        joblib.dump(xgb_model, model_dir / "xgb.joblib")
        if xgb_label_map is not None:
            np.save(model_dir / "xgb_label_map.npy", xgb_label_map)

    logger.info(f"  Models saved to {model_dir}")


def load_models(
    output_dir: Path,
    dataset_name: str,
    input_dim: int,
    num_classes: int,
    dnn_config: DNNConfig,
    device: torch.device,
) -> tuple:
    """Load previously trained models from disk."""
    import joblib

    model_dir = output_dir / "models" / dataset_name

    # Load DNN
    dnn = NIDSNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=dnn_config.hidden_layers,
        dropout_rate=dnn_config.dropout_rate,
    )
    dnn.load_state_dict(torch.load(model_dir / "dnn.pt", map_location=device, weights_only=True))
    dnn = dnn.to(device)
    dnn.eval()

    # Warmup CUDA context to avoid cuBLAS initialization warning
    if device.type == "cuda":
        with torch.no_grad():
            dnn(torch.zeros(1, input_dim, device=device))

    # Load RF
    rf = joblib.load(model_dir / "rf.joblib")

    # Load XGBoost (if available)
    xgb_path = model_dir / "xgb.joblib"
    xgb_model = joblib.load(xgb_path) if xgb_path.exists() else None
    xgb_label_map_path = model_dir / "xgb_label_map.npy"
    xgb_label_map = np.load(xgb_label_map_path) if xgb_label_map_path.exists() else None

    return dnn, rf, xgb_model, xgb_label_map


# ---------------------------------------------------------------------------
# Grid reshaping utilities for CNN models
# ---------------------------------------------------------------------------

def _compute_grid_size(n_features: int) -> int:
    """Compute the smallest grid size such that grid_size² >= n_features."""
    return math.ceil(math.sqrt(n_features))


def _features_to_grid(X: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Reshape flat features (B, F) to a 2D grid (B, 1, grid_size, grid_size).

    Zero-pads if F < grid_size².
    """
    B, F = X.shape
    total = grid_size * grid_size
    if F < total:
        X = F_pad(X, (0, total - F))
    elif F > total:
        X = X[:, :total]
    return X.view(B, 1, grid_size, grid_size)


def F_pad(tensor: torch.Tensor, pad: tuple) -> torch.Tensor:
    """Functional pad wrapper (avoids shadowing torch.nn.functional)."""
    return F.pad(tensor, pad)


# ---------------------------------------------------------------------------
# CNN-LSTM Model
# ---------------------------------------------------------------------------

class CNNLSTM(nn.Module):
    """CNN-LSTM hybrid: spatial feature extraction (CNN) followed by temporal
    modelling (LSTM) for network intrusion detection."""

    def __init__(self, num_classes: int, grid_size: int = 11,
                 hidden_size: int = 128, num_lstm_layers: int = 1,
                 dropout: float = 0.5):
        super().__init__()
        self.grid_size = grid_size

        self.cnn = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

        self.conv_out_size = 64 * (grid_size // 2) * (grid_size // 2)

        self.lstm = nn.LSTM(
            input_size=self.conv_out_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, grid_size, grid_size)
        x = self.cnn(x)
        x = x.view(x.size(0), 1, -1)  # (B, 1, conv_out_size)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        return self.fc(x)


class FlatCNNLSTM(nn.Module):
    """Wraps CNNLSTM to accept flat input (B, n_features) for use with
    explainers (SHAP, IG, DeepLIFT) that require an nn.Module interface."""

    def __init__(self, cnn_lstm: CNNLSTM, grid_size: int):
        super().__init__()
        self.cnn_lstm = cnn_lstm
        self.grid_size = grid_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _features_to_grid(x, self.grid_size)
        return self.cnn_lstm(x)


class CNNLSTMWrapper:
    """Sklearn-style predict/predict_proba interface for CNNLSTM.
    Accepts flat numpy arrays and handles grid reshaping internally."""

    def __init__(self, model: nn.Module, device: torch.device, grid_size: int):
        self.model = model
        self.device = device
        self.grid_size = grid_size
        self.base_model = model.module if isinstance(model, nn.DataParallel) else model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            tensor = _features_to_grid(tensor, self.grid_size)
            logits = self.model(tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------------------------------------------------------------------
# CNN-GRU Model
# ---------------------------------------------------------------------------

class CNNGRU(nn.Module):
    """CNN-GRU hybrid: parallel spatial (CNN) and temporal (GRU) feature
    extraction with late fusion for network intrusion detection."""

    def __init__(self, num_classes: int, input_channels: int = 1,
                 cnn_filters: int = 64, cnn_kernel_size: int = 3,
                 pool_kernel_size: int = 2, cnn_dropout: float = 0.5,
                 gru_hidden_size: int = 75, gru_num_layers: int = 1,
                 gru_dropout: float = 0.5, fc_hidden_size: int = 128,
                 input_spatial_size: int = 11):
        super().__init__()
        self.input_spatial_size = input_spatial_size

        self.cnn_branch = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, cnn_filters, kernel_size=cnn_kernel_size,
                      padding=cnn_kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size),
            nn.Dropout2d(cnn_dropout),
        )

        self.gru_bn = nn.BatchNorm1d(input_channels)
        self.gru = nn.GRU(
            input_size=input_channels, hidden_size=gru_hidden_size,
            num_layers=gru_num_layers, batch_first=True,
            dropout=gru_dropout if gru_num_layers > 1 else 0,
        )

        cnn_output_size = input_spatial_size // pool_kernel_size
        self.fc1_input_size = gru_hidden_size + cnn_filters * cnn_output_size * cnn_output_size

        self.fc = nn.Sequential(
            nn.Linear(self.fc1_input_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(gru_dropout),
            nn.Linear(fc_hidden_size, num_classes),
        )

    def forward(self, x_spatial: torch.Tensor, x_temporal: torch.Tensor) -> torch.Tensor:
        # x_spatial:  (B, channels, H, W)
        # x_temporal: (B, seq_len, channels)
        cnn_out = self.cnn_branch(x_spatial)
        cnn_out = torch.flatten(cnn_out, 1)

        x_temporal = self.gru_bn(x_temporal.transpose(1, 2)).transpose(1, 2)
        _, h_n = self.gru(x_temporal)
        gru_out = h_n[-1]

        combined = torch.cat((cnn_out, gru_out), dim=1)
        return self.fc(combined)


class FlatCNNGRU(nn.Module):
    """Wraps CNNGRU to accept flat input (B, n_features) for explainers.

    Internally creates spatial (2D grid) and temporal (sequence) views from
    the flat feature vector.
    """

    def __init__(self, cnn_gru: CNNGRU, grid_size: int):
        super().__init__()
        self.cnn_gru = cnn_gru
        self.grid_size = grid_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_spatial = _features_to_grid(x, self.grid_size)
        # Temporal: (B, grid_size², 1) — each padded feature as a timestep
        B, F = x.shape
        total = self.grid_size * self.grid_size
        if F < total:
            x_pad = F_pad(x, (0, total - F))
        elif F > total:
            x_pad = x[:, :total]
        else:
            x_pad = x
        x_temporal = x_pad.unsqueeze(2)  # (B, seq_len, 1)
        return self.cnn_gru(x_spatial, x_temporal)


class CNNGRUWrapper:
    """Sklearn-style predict/predict_proba interface for CNNGRU.
    Accepts flat numpy arrays and handles grid reshaping internally."""

    def __init__(self, model: nn.Module, device: torch.device, grid_size: int):
        self.model = model
        self.device = device
        self.grid_size = grid_size
        self.base_model = model.module if isinstance(model, nn.DataParallel) else model

    def _prepare_inputs(self, tensor: torch.Tensor):
        """Convert flat tensor to spatial + temporal views."""
        B, F = tensor.shape
        total = self.grid_size * self.grid_size
        if F < total:
            tensor_pad = F_pad(tensor, (0, total - F))
        elif F > total:
            tensor_pad = tensor[:, :total]
        else:
            tensor_pad = tensor
        x_spatial = tensor_pad.view(B, 1, self.grid_size, self.grid_size)
        x_temporal = tensor_pad.unsqueeze(2)  # (B, seq_len, 1)
        return x_spatial, x_temporal

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            x_spatial, x_temporal = self._prepare_inputs(tensor)
            logits = self.model(x_spatial, x_temporal)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.weight)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# ---------------------------------------------------------------------------
# CNN-LSTM training
# ---------------------------------------------------------------------------

def train_cnn_lstm(
    dataset: DatasetBundle,
    config: CNNLSTMConfig,
    device: torch.device,
    num_gpus: int = 1,
) -> tuple[CNNLSTM, CNNLSTMWrapper, dict]:
    """Train CNN-LSTM with multi-GPU DataParallel and AMP support."""
    n_features = dataset.X_train.shape[1]
    grid_size = config.grid_size or _compute_grid_size(n_features)
    logger.info(
        f"Training CNN-LSTM on {dataset.dataset_name} "
        f"({dataset.X_train.shape}, grid={grid_size}x{grid_size})"
    )

    model = CNNLSTM(
        num_classes=dataset.num_classes,
        grid_size=grid_size,
        hidden_size=config.hidden_size,
        num_lstm_layers=config.num_lstm_layers,
        dropout=config.dropout,
    )

    if num_gpus > 1 and torch.cuda.device_count() > 1:
        logger.info(f"  Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    if device.type == "cuda":
        with torch.no_grad():
            dummy = torch.zeros(1, 1, grid_size, grid_size, device=device)
            base = model.module if isinstance(model, nn.DataParallel) else model
            base(dummy)

    use_amp = device.type == "cuda"
    if use_amp:
        logger.info("  Using Automatic Mixed Precision (AMP) training")

    # For large datasets, use lazy reshaping to avoid duplicating the full array
    n_train = dataset.X_train.shape[0]
    if n_train > 5_000_000:
        logger.info(f"  Using lazy grid dataset ({n_train} rows)")
        train_ds = LazyGridDataset(dataset.X_train, dataset.y_train, grid_size)
        val_ds = LazyGridDataset(dataset.X_val, dataset.y_val, grid_size)
        num_workers = 2
    else:
        def _to_grid(X: np.ndarray) -> np.ndarray:
            total = grid_size * grid_size
            if X.shape[1] < total:
                X = np.pad(X, ((0, 0), (0, total - X.shape[1])))
            elif X.shape[1] > total:
                X = X[:, :total]
            return X.reshape(-1, 1, grid_size, grid_size)

        train_ds = TensorDataset(
            torch.tensor(_to_grid(dataset.X_train), dtype=torch.float32),
            torch.tensor(dataset.y_train, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(_to_grid(dataset.X_val), dtype=torch.float32),
            torch.tensor(dataset.y_val, dtype=torch.long),
        )
        num_workers = 4
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    train_start = time.time()

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast("cuda", enabled=use_amp):
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(val_ds)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch + 1}/{config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"  Early stopping at epoch {epoch + 1}")
                break

    train_time = time.time() - train_start
    logger.info(f"  CNN-LSTM training completed in {train_time:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    if device.type == "cuda":
        with torch.no_grad():
            base = model.module if isinstance(model, nn.DataParallel) else model
            base(torch.zeros(1, 1, grid_size, grid_size, device=device))

    wrapper = CNNLSTMWrapper(model, device, grid_size)
    metrics = _evaluate_model(wrapper, dataset.X_test, dataset.y_test, dataset.num_classes)
    metrics["train_time_seconds"] = train_time
    metrics["grid_size"] = grid_size
    logger.info(f"  CNN-LSTM Metrics: {metrics}")

    return model, wrapper, metrics


# ---------------------------------------------------------------------------
# CNN-GRU training
# ---------------------------------------------------------------------------

def train_cnn_gru(
    dataset: DatasetBundle,
    config: CNNGRUConfig,
    device: torch.device,
    num_gpus: int = 1,
) -> tuple[CNNGRU, CNNGRUWrapper, dict]:
    """Train CNN-GRU with multi-GPU DataParallel and AMP support."""
    n_features = dataset.X_train.shape[1]
    grid_size = config.input_spatial_size or _compute_grid_size(n_features)
    total = grid_size * grid_size
    logger.info(
        f"Training CNN-GRU on {dataset.dataset_name} "
        f"({dataset.X_train.shape}, grid={grid_size}x{grid_size})"
    )

    model = CNNGRU(
        num_classes=dataset.num_classes,
        input_channels=config.input_channels,
        cnn_filters=config.cnn_filters,
        cnn_kernel_size=config.cnn_kernel_size,
        pool_kernel_size=config.pool_kernel_size,
        cnn_dropout=config.cnn_dropout,
        gru_hidden_size=config.gru_hidden_size,
        gru_num_layers=config.gru_num_layers,
        gru_dropout=config.gru_dropout,
        fc_hidden_size=config.fc_hidden_size,
        input_spatial_size=grid_size,
    )

    if num_gpus > 1 and torch.cuda.device_count() > 1:
        logger.info(f"  Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    if device.type == "cuda":
        with torch.no_grad():
            base = model.module if isinstance(model, nn.DataParallel) else model
            dummy_s = torch.zeros(1, config.input_channels, grid_size, grid_size, device=device)
            dummy_t = torch.zeros(1, total, config.gru_input_size, device=device)
            base(dummy_s, dummy_t)

    use_amp = device.type == "cuda"
    if use_amp:
        logger.info("  Using Automatic Mixed Precision (AMP) training")

    n_train = dataset.X_train.shape[0]
    if n_train > 5_000_000:
        logger.info(f"  Using lazy dual-grid dataset ({n_train} rows)")
        train_ds = LazyDualGridDataset(dataset.X_train, dataset.y_train, grid_size)
        val_ds = LazyDualGridDataset(dataset.X_val, dataset.y_val, grid_size)
        num_workers = 2
    else:
        def _prepare(X: np.ndarray):
            """Convert flat features to spatial + temporal arrays."""
            if X.shape[1] < total:
                X = np.pad(X, ((0, 0), (0, total - X.shape[1])))
            elif X.shape[1] > total:
                X = X[:, :total]
            spatial = X.reshape(-1, 1, grid_size, grid_size)
            temporal = X.reshape(-1, total, 1)
            return spatial, temporal

        X_train_s, X_train_t = _prepare(dataset.X_train)
        X_val_s, X_val_t = _prepare(dataset.X_val)

        train_ds = TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(X_train_t, dtype=torch.float32),
            torch.tensor(dataset.y_train, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val_s, dtype=torch.float32),
            torch.tensor(X_val_t, dtype=torch.float32),
            torch.tensor(dataset.y_val, dtype=torch.long),
        )
        num_workers = 4
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    train_start = time.time()

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for X_s, X_t, y_batch in train_loader:
            X_s = X_s.to(device, non_blocking=True)
            X_t = X_t.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                outputs = model(X_s, X_t)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast("cuda", enabled=use_amp):
            for X_s, X_t, y_batch in val_loader:
                X_s = X_s.to(device, non_blocking=True)
                X_t = X_t.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                outputs = model(X_s, X_t)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * len(y_batch)
        val_loss /= len(val_ds)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch + 1}/{config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"  Early stopping at epoch {epoch + 1}")
                break

    train_time = time.time() - train_start
    logger.info(f"  CNN-GRU training completed in {train_time:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    if device.type == "cuda":
        with torch.no_grad():
            base = model.module if isinstance(model, nn.DataParallel) else model
            dummy_s = torch.zeros(1, config.input_channels, grid_size, grid_size, device=device)
            dummy_t = torch.zeros(1, total, config.gru_input_size, device=device)
            base(dummy_s, dummy_t)

    wrapper = CNNGRUWrapper(model, device, grid_size)
    metrics = _evaluate_model(wrapper, dataset.X_test, dataset.y_test, dataset.num_classes)
    metrics["train_time_seconds"] = train_time
    metrics["grid_size"] = grid_size
    logger.info(f"  CNN-GRU Metrics: {metrics}")

    return model, wrapper, metrics


# ---------------------------------------------------------------------------
# CNN model save/load
# ---------------------------------------------------------------------------

def save_cnn_models(
    output_dir: Path,
    dataset_name: str,
    cnn_lstm_model: nn.Module | None = None,
    cnn_lstm_config: CNNLSTMConfig | None = None,
    cnn_gru_model: nn.Module | None = None,
    cnn_gru_config: CNNGRUConfig | None = None,
) -> None:
    """Save trained CNN models and their configs to disk."""
    model_dir = output_dir / "models" / dataset_name
    model_dir.mkdir(parents=True, exist_ok=True)

    if cnn_lstm_model is not None:
        base = cnn_lstm_model.module if isinstance(cnn_lstm_model, nn.DataParallel) else cnn_lstm_model
        torch.save(base.state_dict(), model_dir / "cnn_lstm.pt")
        if cnn_lstm_config is not None:
            with open(model_dir / "cnn_lstm_config.json", "w") as f:
                json.dump(asdict(cnn_lstm_config), f, indent=2)
        logger.info(f"  CNN-LSTM model saved to {model_dir}")

    if cnn_gru_model is not None:
        base = cnn_gru_model.module if isinstance(cnn_gru_model, nn.DataParallel) else cnn_gru_model
        torch.save(base.state_dict(), model_dir / "cnn_gru.pt")
        if cnn_gru_config is not None:
            with open(model_dir / "cnn_gru_config.json", "w") as f:
                json.dump(asdict(cnn_gru_config), f, indent=2)
        logger.info(f"  CNN-GRU model saved to {model_dir}")


def load_cnn_models(
    output_dir: Path,
    dataset_name: str,
    num_classes: int,
    device: torch.device,
) -> tuple[CNNLSTM | None, CNNLSTMConfig | None, CNNGRU | None, CNNGRUConfig | None]:
    """Load previously trained CNN models from disk."""
    model_dir = output_dir / "models" / dataset_name
    cnn_lstm = None
    cnn_lstm_config = None
    cnn_gru = None
    cnn_gru_config = None

    # Load CNN-LSTM
    cnn_lstm_path = model_dir / "cnn_lstm.pt"
    if cnn_lstm_path.exists():
        with open(model_dir / "cnn_lstm_config.json") as f:
            cnn_lstm_config = CNNLSTMConfig(**json.load(f))
        cnn_lstm = CNNLSTM(
            num_classes=num_classes,
            grid_size=cnn_lstm_config.grid_size,
            hidden_size=cnn_lstm_config.hidden_size,
            num_lstm_layers=cnn_lstm_config.num_lstm_layers,
            dropout=cnn_lstm_config.dropout,
        )
        cnn_lstm.load_state_dict(torch.load(cnn_lstm_path, map_location=device, weights_only=True))
        cnn_lstm = cnn_lstm.to(device)
        cnn_lstm.eval()
        if device.type == "cuda":
            with torch.no_grad():
                cnn_lstm(torch.zeros(1, 1, cnn_lstm_config.grid_size, cnn_lstm_config.grid_size, device=device))
        logger.info(f"  CNN-LSTM model loaded from {model_dir}")

    # Load CNN-GRU
    cnn_gru_path = model_dir / "cnn_gru.pt"
    if cnn_gru_path.exists():
        with open(model_dir / "cnn_gru_config.json") as f:
            cnn_gru_config = CNNGRUConfig(**json.load(f))
        gs = cnn_gru_config.input_spatial_size
        cnn_gru = CNNGRU(
            num_classes=num_classes,
            input_channels=cnn_gru_config.input_channels,
            cnn_filters=cnn_gru_config.cnn_filters,
            cnn_kernel_size=cnn_gru_config.cnn_kernel_size,
            pool_kernel_size=cnn_gru_config.pool_kernel_size,
            cnn_dropout=cnn_gru_config.cnn_dropout,
            gru_hidden_size=cnn_gru_config.gru_hidden_size,
            gru_num_layers=cnn_gru_config.gru_num_layers,
            gru_dropout=cnn_gru_config.gru_dropout,
            fc_hidden_size=cnn_gru_config.fc_hidden_size,
            input_spatial_size=gs,
        )
        cnn_gru.load_state_dict(torch.load(cnn_gru_path, map_location=device, weights_only=True))
        cnn_gru = cnn_gru.to(device)
        cnn_gru.eval()
        if device.type == "cuda":
            with torch.no_grad():
                total = gs * gs
                dummy_s = torch.zeros(1, cnn_gru_config.input_channels, gs, gs, device=device)
                dummy_t = torch.zeros(1, total, cnn_gru_config.gru_input_size, device=device)
                cnn_gru(dummy_s, dummy_t)
        logger.info(f"  CNN-GRU model loaded from {model_dir}")

    return cnn_lstm, cnn_lstm_config, cnn_gru, cnn_gru_config
