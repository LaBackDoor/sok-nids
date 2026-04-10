"""Model architectures for Experiment 4: DNN, 1D-CNN, RF, SVM.

All neural network models support multi-GPU via DataParallel and AMP training.
"""

import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier as SKLearnRandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from cuml.svm import SVC
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Re-export SoftmaxModel from commons so exp1's explainers.py can find it
import importlib.util as _ilu
_commons_models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "commons", "models.py")
_spec = _ilu.spec_from_file_location("commons_models", _commons_models_path)
_commons_models = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_commons_models)
SoftmaxModel = _commons_models.SoftmaxModel

from config import CNNConfig, DNNConfig, RFConfig, SVMConfig

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# DNN (reused architecture from Experiment 1)
# ---------------------------------------------------------------------------

class NIDSNet(nn.Module):
    """Fully connected DNN for network intrusion detection."""

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_layers: list[int] | None = None, dropout_rate: float = 0.01):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [1024, 768, 512]
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.Dropout(dropout_rate)])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# 1D-CNN for tabular data
# ---------------------------------------------------------------------------

class NIDS1DCNN(nn.Module):
    """1D Convolutional Neural Network for tabular NIDS data.

    Reshapes flat feature vector into (1, n_features) and applies 1D convolutions.
    """

    def __init__(self, input_dim: int, num_classes: int,
                 channels: list[int] | None = None,
                 kernel_size: int = 3, dropout_rate: float = 0.1):
        super().__init__()
        if channels is None:
            channels = [64, 128, 64]

        conv_layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            # Padding to preserve sequence length
            padding = kernel_size // 2
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Wrapper classes
# ---------------------------------------------------------------------------

class NNWrapper:
    """Unified predict/predict_proba interface for PyTorch models."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.base_model = model.module if isinstance(model, nn.DataParallel) else model

    def predict_proba(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        self.model.eval()
        parts = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                tensor = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(self.device)
                logits = self.model(tensor)
                parts.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(parts, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class SKLearnWrapper:
    """Unified predict/predict_proba interface for sklearn models."""

    def __init__(self, model, num_classes: int):
        self.model = model
        self.num_classes = num_classes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = np.asarray(self.model.predict_proba(X.astype(np.float32)))
        if proba.shape[1] < self.num_classes:
            full = np.zeros((proba.shape[0], self.num_classes), dtype=proba.dtype)
            full[:, self.model.classes_] = proba
            return full
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X.astype(np.float32)))


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def _train_nn(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    device: torch.device, num_gpus: int,
    batch_size: int, epochs: int, lr: float,
    patience: int, model_name: str,
) -> tuple[nn.Module, float]:
    """Generic PyTorch model training with AMP and multi-GPU."""
    input_dim = X_train.shape[1]

    if num_gpus > 1 and torch.cuda.device_count() > 1:
        logger.info(f"  {model_name}: Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Warmup
    if device.type == "cuda":
        with torch.no_grad():
            model(torch.zeros(2, input_dim, device=device))

    use_amp = device.type == "cuda"
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    t0 = time.time()

    for epoch in range(epochs):
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
                f"  {model_name} Epoch {epoch+1}/{epochs} - "
                f"Train: {train_loss:.4f}, Val: {val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  {model_name}: Early stopping at epoch {epoch+1}")
                break

    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    logger.info(f"  {model_name} training completed in {train_time:.1f}s")
    return model, train_time


def train_dnn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    num_classes: int,
    config: DNNConfig, device: torch.device, num_gpus: int,
) -> tuple[nn.Module, NNWrapper, float]:
    """Train DNN on given features."""
    input_dim = X_train.shape[1]
    model = NIDSNet(input_dim, num_classes, config.hidden_layers, config.dropout_rate)
    model, train_time = _train_nn(
        model, X_train, y_train, X_val, y_val,
        device, num_gpus, config.batch_size, config.epochs,
        config.learning_rate, config.early_stopping_patience, "DNN",
    )
    wrapper = NNWrapper(model, device)
    return model, wrapper, train_time


def train_cnn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    num_classes: int,
    config: CNNConfig, device: torch.device, num_gpus: int,
) -> tuple[nn.Module, NNWrapper, float]:
    """Train 1D-CNN on given features."""
    input_dim = X_train.shape[1]
    model = NIDS1DCNN(input_dim, num_classes, config.channels, config.kernel_size, config.dropout_rate)
    model, train_time = _train_nn(
        model, X_train, y_train, X_val, y_val,
        device, num_gpus, config.batch_size, config.epochs,
        config.learning_rate, config.early_stopping_patience, "CNN",
    )
    wrapper = NNWrapper(model, device)
    return model, wrapper, train_time


def train_rf(
    X_train: np.ndarray, y_train: np.ndarray,
    num_classes: int, config: RFConfig,
) -> tuple:
    """Train Random Forest on GPU via cuML."""
    model = CuMLRandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        split_criterion=config.criterion if config.criterion != "gini" else "gini",
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train.astype(np.float32), y_train.astype(np.int32))
    train_time = time.time() - t0
    logger.info(f"  RF (GPU) training completed in {train_time:.1f}s")
    wrapper = SKLearnWrapper(model, num_classes)
    return model, wrapper, train_time


def train_svm(
    X_train: np.ndarray, y_train: np.ndarray,
    num_classes: int, config: SVMConfig,
) -> tuple[SVC, SKLearnWrapper, float]:
    """Train SVM on given features. Subsamples if dataset is too large."""
    if config.max_train_samples is not None and len(X_train) > config.max_train_samples:
        logger.info(f"  SVM: Subsampling {len(X_train)} -> {config.max_train_samples} samples")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), config.max_train_samples, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    model = SVC(
        kernel=config.kernel,
        C=config.C,
        gamma=config.gamma,
        max_iter=config.max_iter,
        probability=config.probability,
        random_state=42,
        cache_size=2000,
    )
    t0 = time.time()
    model.fit(X_sub.astype(np.float32), y_sub.astype(np.int32))
    train_time = time.time() - t0
    logger.info(f"  SVM (GPU) training completed in {train_time:.1f}s")
    wrapper = SKLearnWrapper(model, num_classes)
    return model, wrapper, train_time


def train_all_downstream(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    num_classes: int,
    dnn_config: DNNConfig, cnn_config: CNNConfig,
    rf_config: RFConfig, svm_config: SVMConfig,
    device: torch.device, num_gpus: int,
) -> dict[str, tuple]:
    """Train all 4 downstream models. Returns dict of (model, wrapper, train_time)."""
    results = {}

    logger.info(f"  Training downstream models on {X_train.shape[1]} features...")

    _, dnn_wrap, dnn_time = train_dnn(
        X_train, y_train, X_val, y_val, num_classes, dnn_config, device, num_gpus)
    results["DNN"] = (dnn_wrap, dnn_time)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _, cnn_wrap, cnn_time = train_cnn(
        X_train, y_train, X_val, y_val, num_classes, cnn_config, device, num_gpus)
    results["CNN"] = (cnn_wrap, cnn_time)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _, rf_wrap, rf_time = train_rf(X_train, y_train, num_classes, rf_config)
    results["RF"] = (rf_wrap, rf_time)

    _, svm_wrap, svm_time = train_svm(X_train, y_train, num_classes, svm_config)
    results["SVM"] = (svm_wrap, svm_time)

    return results
