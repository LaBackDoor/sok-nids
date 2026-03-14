import json
from dataclasses import asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import time
from pathlib import Path 

from config import CNNGRUConfig
from data_loader import DatasetBundle
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
from models import _evaluate_model

logger = logging.getLogger(__name__)


class CNNGRUWrapper:
    """
    Wrapper for CNN-GRU models to provide a unified Scikit-Learn style interface.
    Handles sequence reshaping and recurrent state outputs.
    """

    def __init__(self, model: nn.Module, device: torch.device, input_shape: tuple):
        """
        Args:
            model: The CNN-GRU model.
            device: torch.device (cpu or cuda).
            input_shape: The shape the model expects per sample.
                         e.g., (timesteps, features) or (channels, length).
        """
        self.model = model
        self.device = device
        self.input_shape = input_shape
        self.base_model = model.module if isinstance(model, nn.DataParallel) else model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        
        # 1. Convert to Tensor
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # 2. Reshape flat input to (Batch, *input_shape)
        # If X is already shaped correctly, this won't change anything.
        if len(tensor.shape) == 2:
            tensor = tensor.view(-1, *self.input_shape)

        # 3. Batching (Reduced size for memory-heavy GRU/CNN layers)
        batch_size = 128 
        probs_list = []

        with torch.no_grad():
            for i in range(0, len(tensor), batch_size):
                batch = tensor[i : i + batch_size]
                
                # 4. Handle RNN Tuple Outputs
                # GRUs often return (output, hidden_state). We only want the output/logits.
                output = self.model(batch)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output

                # 5. Apply Softmax and move to CPU
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)

        return np.concatenate(probs_list, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

class CNNGRU(nn.Module):
    def __init__(self, num_classes, input_channels=1, cnn_filters=64,
                 cnn_kernel_size=3, pool_kernel_size=2, cnn_dropout=0.5,
                 gru_hidden_size=75, gru_num_layers=1, gru_dropout=0.5,
                 fc_hidden_size=128, input_spatial_size=11):
        super(CNNGRU, self).__init__()

        # --- CNN Branch (Spatial Feature Extraction) ---
        self.cnn_branch = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, cnn_filters, kernel_size=cnn_kernel_size,
                      padding=cnn_kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size),
            nn.Dropout2d(cnn_dropout)  # Fix 1: Dropout -> Dropout2d for conv feature maps
        )

        # --- GRU Branch (Temporal Feature Extraction) ---
        self.gru_bn = nn.BatchNorm1d(input_channels)
        self.gru = nn.GRU(input_size=input_channels, hidden_size=gru_hidden_size,  # Fix 2: hardcoded 1 -> input_channels
                          num_layers=gru_num_layers, batch_first=True,
                          dropout=gru_dropout if gru_num_layers > 1 else 0)

        # --- Dynamically compute flattened CNN output size ---
        cnn_output_size = input_spatial_size // pool_kernel_size
        self.fc1_input_size = gru_hidden_size + cnn_filters * cnn_output_size * cnn_output_size

        # --- Feature Fusion & Classification Layer ---
        self.fc = nn.Sequential(  # Fix 3: consolidate into Sequential to match CNN/LSTM style
            nn.Linear(self.fc1_input_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(gru_dropout),
            nn.Linear(fc_hidden_size, num_classes)
        )

    def forward(self, x_spatial, x_temporal):
        # x_spatial:  (Batch, input_channels, H, W)
        # x_temporal: (Batch, seq_len, input_channels)

        # CNN Processing
        cnn_out = self.cnn_branch(x_spatial)
        cnn_out = torch.flatten(cnn_out, 1)

        # GRU Processing
        x_temporal = self.gru_bn(x_temporal.transpose(1, 2)).transpose(1, 2)
        _, h_n = self.gru(x_temporal)
        gru_out = h_n[-1]

        # Fusion & Classification
        combined = torch.cat((cnn_out, gru_out), dim=1)
        logits = self.fc(combined)  # Fix 3: simplified forward pass
        return logits

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class CNNLSTM(nn.Module):
    def __init__(self, num_classes, grid_size=11, hidden_size=128, num_lstm_layers=1, dropout=0.5):
        super(CNNLSTM, self).__init__()

        # Spatial Feature Extraction (CNN)
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout)
        )

        # Calculate flattened feature size for LSTM input
        self.conv_out_size = 64 * (grid_size // 2) * (grid_size // 2)

        # Temporal Feature Extraction (LSTM)
        self.lstm = nn.LSTM(
            input_size=self.conv_out_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )

        # Classification Layer (MLP)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 1, grid_size, grid_size)

        # 1. Spatial Phase
        x = self.cnn(x)

        # 2. Reshape for LSTM: (Batch, Seq_Len=1, Features)
        x = x.view(x.size(0), 1, -1)

        # 3. Temporal Phase
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]  # Support multi-layer LSTM: take last layer's hidden state

        # 4. Classification Phase
        logits = self.fc(x)
        return logits

def train_cnngru(
    dataset: DatasetBundle,
    config: CNNGRUConfig,
    device: torch.device,
    num_gpus: int = 1,
) -> tuple[CNNGRU, CNNGRUWrapper, dict]:
    """Train CNNGRU with multi-GPU DataParallel and AMP (mixed precision) support."""
    logger.info(f"Training CNNGRU on {dataset.dataset_name} ({dataset.X_train.shape})")

    model = CNNGRU(dataset.num_classes)

    # Multi-GPU
    if num_gpus > 1 and torch.cuda.device_count() > 1:
        logger.info(f"  Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Warmup CUDA context
    if device.type == "cuda":
        with torch.no_grad():
            dummy_spatial = torch.zeros(1, config.input_channels, config.input_spatial_size, config.input_spatial_size, device=device)
            dummy_temporal = torch.zeros(1, config.seq_len, config.gru_input_size, device=device)
            model(dummy_spatial, dummy_temporal)

    use_amp = device.type == "cuda"
    if use_amp:
        logger.info("  Using Automatic Mixed Precision (AMP) training")

    # Dataloaders - expects dataset to have spatial and temporal components
    train_ds = TensorDataset(
        torch.tensor(dataset.X_train_spatial, dtype=torch.float32),
        torch.tensor(dataset.X_train_temporal, dtype=torch.float32),
        torch.tensor(dataset.y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(dataset.X_val_spatial, dtype=torch.float32),
        torch.tensor(dataset.X_val_temporal, dtype=torch.float32),
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
        for X_spatial, X_temporal, y_batch in train_loader:
            X_spatial = X_spatial.to(device, non_blocking=True)
            X_temporal = X_temporal.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                outputs = model(X_spatial, X_temporal)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast("cuda", enabled=use_amp):
            for X_spatial, X_temporal, y_batch in val_loader:
                X_spatial = X_spatial.to(device, non_blocking=True)
                X_temporal = X_temporal.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                outputs = model(X_spatial, X_temporal)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * len(y_batch)
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
    logger.info(f"  CNNGRU training completed in {train_time:.1f}s")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    # Re-warmup CUDA context after restoring best state
    if device.type == "cuda":
        with torch.no_grad():
            base = model.module if isinstance(model, nn.DataParallel) else model
            dummy_spatial = torch.zeros(1, config.input_channels, config.input_spatial_size, config.input_spatial_size, device=device)
            dummy_temporal = torch.zeros(1, config.seq_len, config.gru_input_size, device=device)
            base(dummy_spatial, dummy_temporal)

    wrapper = CNNGRUWrapper(model, device)
    metrics =  None
    metrics = _evaluate_model(wrapper, dataset.X_test_spatial, dataset.X_test_temporal, dataset.y_test, dataset.num_classes)
    metrics["train_time_seconds"] = train_time
    logger.info(f"CNNGRU Metrics: {metrics}")

    return model, wrapper, metrics

def save_cnngru(
    cnngru_model: nn.Module,
    config: CNNGRUConfig,
    output_dir: Path,
    dataset_name: str,
) -> None:
    """Save trained CNNGRU model and config to disk."""
    model_dir = output_dir / "models" / dataset_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    base = cnngru_model.module if isinstance(cnngru_model, nn.DataParallel) else cnngru_model
    torch.save(base.state_dict(), model_dir / "cnngru.pt")
    
    with open(model_dir / "cnngru_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    logger.info(f"CNNGRU model saved to {model_dir}")

def load_cnngru(
    output_dir: Path,
    dataset_name: str,
    device: torch.device,
) -> tuple[CNNGRU, CNNGRUConfig]:
    """Load previously trained CNNGRU model from disk."""
    import json

    model_dir = output_dir / "models" / dataset_name

    # Load config
    with open(model_dir / "cnngru_config.json", "r") as f:
        config_dict = json.load(f)
    config = CNNGRUConfig(**config_dict)

    # Reconstruct model from config
    model = CNNGRU(config)
    model.load_state_dict(
        torch.load(model_dir / "cnngru.pt", map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()

    # Warmup CUDA context to avoid cuBLAS initialization warning
    if device.type == "cuda":
        with torch.no_grad():
            dummy_spatial = torch.zeros(
                1, config.input_channels, 
                config.input_spatial_size, config.input_spatial_size, 
                device=device
            )
            dummy_temporal = torch.zeros(
                1, config.seq_len, config.gru_input_size, 
                device=device
            )
            model(dummy_spatial, dummy_temporal)

    logger.info(f"CNNGRU model loaded from {model_dir}")
    return model, config