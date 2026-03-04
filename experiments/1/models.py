"""DNN (multi-GPU) and Random Forest model training for NIDS classification."""

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import DNNConfig, RFConfig, XGBConfig
from data_loader import DatasetBundle

logger = logging.getLogger(__name__)

# Enable cuDNN auto-tuner for optimal convolution algorithms
torch.backends.cudnn.benchmark = True

# Suppress XGBoost device mismatch warning (prediction still uses GPU via DMatrix fallback)
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")


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
            # Process in batches to avoid OOM
            batch_size = 4096
            if len(tensor) <= batch_size:
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy()
            probs_list = []
            for i in range(0, len(tensor), batch_size):
                batch = tensor[i : i + batch_size]
                logits = self.model(batch)
                probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            return np.concatenate(probs_list, axis=0)

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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(X)
        if self.label_map is not None and self.num_classes is not None:
            full_proba = np.zeros((proba.shape[0], self.num_classes), dtype=proba.dtype)
            for compact_idx, orig_idx in enumerate(self.label_map):
                full_proba[:, orig_idx] = proba[:, compact_idx]
            return full_proba
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
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
        if num_classes == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
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

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        criterion=config.criterion,
        n_jobs=config.n_jobs,
        random_state=42,
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
    )

    train_start = time.time()
    model.fit(
        dataset.X_train, y_train,
        eval_set=[(dataset.X_val, y_val)],
        verbose=False,
    )
    train_time = time.time() - train_start
    logger.info(f"  XGBoost training completed in {train_time:.1f}s")

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
    dnn_model: nn.Module,
    rf_model: RandomForestClassifier,
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
    base = dnn_model.module if isinstance(dnn_model, nn.DataParallel) else dnn_model
    torch.save(base.state_dict(), model_dir / "dnn.pt")

    # Save RF
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
