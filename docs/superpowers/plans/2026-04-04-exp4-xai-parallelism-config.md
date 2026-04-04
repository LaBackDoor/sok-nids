# Experiment 4: XAI Parallelism, Config Centralization, and Normal/PA Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite Experiment 4's XAI pipeline to import Exp 1's parallelized explainer orchestrators (normal + PA-XAI modes), add IG/DeepLIFT methods, and centralize config into `experiments/commons/config.yaml`.

**Architecture:** Exp 4's `xai_selection.py` calls Exp 1's `generate_all_explanations` (normal) and `pa_generate_all_explanations` (PA) to get per-sample attributions, converts them to feature importance rankings, then feeds those into the existing iterative pruning logic. Config is loaded from the shared YAML via `config_loader.load_yaml_config()`, following the same pattern as Exp 3.

**Tech Stack:** Python, PyTorch, SHAP, LIME, Captum (IG/DeepLIFT), pa_xai, scikit-learn, PyYAML

---

### Task 1: Add `experiment_4` Section to `experiments/commons/config.yaml`

**Files:**
- Modify: `experiments/commons/config.yaml:68` (append after experiment_3 section)
- Modify: `experiments/config.yaml:68` (keep in sync — these files mirror each other)

- [ ] **Step 1: Append experiment_4 section to commons/config.yaml**

Add the following after the `experiment_3` block (after line 68):

```yaml
experiment_4:
  output_dir: experiments/4/results
  exp1_output_dir: experiments/1/results
  xai_modes:
    - normal
    - pa

  cnn:
    channels: [64, 128, 64]
    kernel_size: 3
    dropout_rate: 0.1
    learning_rate: 0.001
    batch_size: 8192
    epochs: 100
    early_stopping_patience: 10
  svm:
    kernel: rbf
    C: 1.0
    gamma: scale
    max_iter: 5000
    probability: true
    max_train_samples: null

  statistical:
    pca_variance_threshold: 0.95
    spearman_threshold: 0.8
    info_gain_top_k: 20

  xai:
    f1_degradation_threshold: 0.01
    min_features: 5
    pruning_step_ratio: 0.1
    target_features:
      nsl-kdd: 19
      cic-ids-2017: 15
      unsw-nb15: 15
      cse-cic-ids2018: 15

  parallelism:
    num_gpu_devices: 2
```

- [ ] **Step 2: Mirror the same addition to experiments/config.yaml**

Copy the same `experiment_4:` block to `experiments/config.yaml` after line 68 to keep both files in sync.

- [ ] **Step 3: Verify YAML is valid**

Run:
```bash
uv run python -c "import yaml; yaml.safe_load(open('experiments/commons/config.yaml')); print('OK')"
uv run python -c "import yaml; yaml.safe_load(open('experiments/config.yaml')); print('OK')"
```
Expected: Both print `OK`.

- [ ] **Step 4: Commit**

```bash
git add experiments/commons/config.yaml experiments/config.yaml
git commit -m "config: add experiment_4 section to shared config.yaml"
```

---

### Task 2: Rewrite `experiments/4/config.py` to Load from YAML

**Files:**
- Rewrite: `experiments/4/config.py`

This follows the same pattern as `experiments/3/config.py:77-169` (`load_experiment3_config`).

- [ ] **Step 1: Rewrite config.py**

Replace the entire file with:

```python
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
    target_features: dict = field(
        default_factory=lambda: {
            "nsl-kdd": 19,
            "cic-ids-2017": 15,
            "unsw-nb15": 15,
            "cse-cic-ids2018": 15,
        }
    )


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
```

- [ ] **Step 2: Verify config loads correctly**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids
uv run python -c "
import sys; sys.path.insert(0, 'experiments/4')
from config import load_experiment_config
c = load_experiment_config()
print('output_dir:', c.output_dir)
print('xai_modes:', c.xai_modes)
print('explainer.lime_num_samples:', c.explainer.lime_num_samples)
print('xai.target_features:', c.xai.target_features)
print('dnn.hidden_layers:', c.dnn.hidden_layers)
print('cnn.channels:', c.cnn.channels)
print('datasets:', c.ALL_DATASETS)
"
```
Expected: All values match YAML. `xai_modes` should be `['normal', 'pa']`, `explainer.lime_num_samples` should be `5000`.

- [ ] **Step 3: Commit**

```bash
git add experiments/4/config.py
git commit -m "refactor(exp4): rewrite config.py to load from shared config.yaml"
```

---

### Task 3: Rewrite `experiments/4/xai_selection.py` — Import Exp 1 Orchestrators

**Files:**
- Rewrite: `experiments/4/xai_selection.py`

This is the core change. The new file imports Exp 1's `generate_all_explanations` and `pa_generate_all_explanations`, converts their per-sample attributions to feature importance rankings, then feeds those into the existing `_iterative_prune()` logic.

- [ ] **Step 1: Rewrite xai_selection.py**

Replace the entire file with:

```python
"""XAI-driven feature selection pipeline for Experiment 4.

Imports Exp 1's parallelized explainer orchestrators (normal + PA-XAI)
to generate attributions, then converts to feature importance rankings
for iterative pruning.

Normal mode: SHAP, LIME, IG, DeepLIFT (vanilla, no domain constraints)
PA mode: Protocol-Aware versions with network protocol constraints
"""

import logging
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from config import ExperimentConfig, ExplainerConfig, XAISelectionConfig
from feature_selection import FeatureSelectionResult

logger = logging.getLogger(__name__)


def _attributions_to_importance(attributions: np.ndarray) -> np.ndarray:
    """Convert per-sample attributions to global feature importance.

    Args:
        attributions: shape (n_samples, n_features)

    Returns:
        Feature importance array of shape (n_features,).
    """
    return np.mean(np.abs(attributions), axis=0)


def _iterative_prune(
    feature_importance: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    config: XAISelectionConfig,
    dataset_name: str,
) -> tuple[np.ndarray, list[dict]]:
    """Iteratively prune features using importance scores.

    Removes least important features step by step, evaluating F1 at each
    step. Stops when F1 degrades beyond threshold.

    Returns:
        selected_indices: optimal feature subset indices
        pruning_history: list of dicts with step details
    """
    logger.info("    Starting iterative pruning...")

    n_features = len(feature_importance)
    ranked_indices = np.argsort(feature_importance)[::-1]  # descending importance

    # Train baseline RF on all features for F1 reference
    rf_baseline = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf_baseline.fit(X_train, y_train)
    y_pred_baseline = rf_baseline.predict(X_val)
    baseline_f1 = f1_score(y_val, y_pred_baseline, average="weighted", zero_division=0)
    logger.info(f"    Baseline F1 (all {n_features} features): {baseline_f1:.4f}")

    # Target from config
    target_n = config.target_features.get(dataset_name, config.min_features)
    min_n = max(config.min_features, target_n)

    pruning_history = [{
        "n_features": n_features,
        "f1": baseline_f1,
        "features_removed": 0,
    }]

    current_indices = ranked_indices.copy()
    best_indices = current_indices.copy()
    best_f1 = baseline_f1

    while len(current_indices) > min_n:
        # Remove pruning_step_ratio of remaining features
        n_remove = max(1, int(len(current_indices) * config.pruning_step_ratio))
        n_keep = max(min_n, len(current_indices) - n_remove)

        # Keep top-ranked features
        current_importance = feature_importance[current_indices]
        reranked = np.argsort(current_importance)[::-1]
        current_indices = current_indices[reranked[:n_keep]]

        # Evaluate with reduced features
        rf_reduced = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        rf_reduced.fit(X_train[:, current_indices], y_train)
        y_pred = rf_reduced.predict(X_val[:, current_indices])
        current_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        step_info = {
            "n_features": len(current_indices),
            "f1": float(current_f1),
            "f1_drop": float(baseline_f1 - current_f1),
        }
        pruning_history.append(step_info)

        logger.info(
            f"    {len(current_indices)} features: F1={current_f1:.4f} "
            f"(drop={baseline_f1 - current_f1:.4f})"
        )

        # Check if degradation exceeds threshold
        if baseline_f1 - current_f1 > config.f1_degradation_threshold:
            logger.info(
                f"    F1 degradation ({baseline_f1 - current_f1:.4f}) exceeds "
                f"threshold ({config.f1_degradation_threshold}). "
                f"Reverting to previous step ({len(best_indices)} features)."
            )
            break

        best_indices = current_indices.copy()
        best_f1 = current_f1

    selected = np.sort(best_indices)
    logger.info(f"    Final selection: {len(selected)} features (F1={best_f1:.4f})")
    return selected, pruning_history


def _run_single_mode(
    mode: str,
    dnn_model: nn.Module,
    rf_model: RandomForestClassifier,
    dnn_wrapper,
    rf_wrapper,
    dataset,
    device: torch.device,
    config: ExperimentConfig,
    checkpoint_dir=None,
) -> list[FeatureSelectionResult]:
    """Run XAI pipeline for a single mode (normal or pa).

    Calls Exp 1's orchestrators to generate attributions, then converts
    to feature importance and runs iterative pruning per method.
    """
    from explainers import ExplanationResult

    mode_prefix = "PA-" if mode == "pa" else ""
    logger.info(f"  === XAI Feature Selection ({mode.upper()} mode) ===")

    # Generate attributions using Exp 1's orchestrators
    if mode == "pa":
        from pa_explainers import pa_generate_all_explanations
        explanation_results, explain_indices = pa_generate_all_explanations(
            dnn_model, rf_model, dnn_wrapper, rf_wrapper,
            dataset, device, config.explainer,
            checkpoint_dir=checkpoint_dir,
        )
    else:
        from explainers import generate_all_explanations
        explanation_results, explain_indices = generate_all_explanations(
            dnn_model, rf_model, dnn_wrapper, rf_wrapper,
            dataset, device, config.explainer,
        )

    # Convert each ExplanationResult to a FeatureSelectionResult via pruning
    selection_results: list[FeatureSelectionResult] = []

    for exp_result in explanation_results:
        method_label = f"{mode_prefix}{exp_result.method_name}-{exp_result.model_name}"
        logger.info(f"  Processing {method_label} attributions for feature selection...")

        t0 = time.time()

        importance = _attributions_to_importance(exp_result.attributions)

        selected_indices, pruning_history = _iterative_prune(
            importance,
            dataset.X_train, dataset.y_train,
            dataset.X_val, dataset.y_val,
            dataset.feature_names,
            config.xai,
            dataset.dataset_name,
        )

        elapsed = exp_result.total_time_s + (time.time() - t0)

        selection_results.append(FeatureSelectionResult(
            method_name=method_label,
            selected_indices=selected_indices,
            feature_rankings=importance,
            selected_feature_names=[dataset.feature_names[i] for i in selected_indices],
            n_original=len(dataset.feature_names),
            n_selected=len(selected_indices),
            selection_time_s=elapsed,
        ))

    return selection_results


def run_xai_pipeline(
    dnn_model: nn.Module,
    rf_model: RandomForestClassifier,
    dnn_wrapper,
    rf_wrapper,
    dataset,
    device: torch.device,
    config: ExperimentConfig,
) -> list[FeatureSelectionResult]:
    """Run XAI-driven feature selection for all configured modes.

    For each mode in config.xai_modes (e.g. ["normal", "pa"]), generates
    attributions via Exp 1's orchestrators and converts to feature rankings.

    Args:
        dnn_model: Trained DNN (PyTorch).
        rf_model: Trained RandomForest.
        dnn_wrapper: NNWrapper with predict_proba.
        rf_wrapper: SKLearnWrapper with predict_proba.
        dataset: DatasetBundle from Exp 1's data_loader.
        device: torch device.
        config: Full ExperimentConfig.

    Returns:
        List of FeatureSelectionResult (one per XAI method × mode).
    """
    logger.info("=== XAI-Driven Feature Selection Pipeline ===")
    logger.info(f"  Modes: {config.xai_modes}")

    all_results: list[FeatureSelectionResult] = []
    checkpoint_dir = config.output_dir / dataset.dataset_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for mode in config.xai_modes:
        mode_key = "normal" if mode in ("n", "normal") else "pa"
        try:
            results = _run_single_mode(
                mode_key,
                dnn_model, rf_model, dnn_wrapper, rf_wrapper,
                dataset, device, config,
                checkpoint_dir=str(checkpoint_dir) if mode_key == "pa" else None,
            )
            all_results.extend(results)
        except Exception as e:
            logger.error(f"XAI pipeline ({mode_key} mode) failed: {e}", exc_info=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for r in all_results:
        logger.info(
            f"  {r.method_name}: {r.n_original} -> {r.n_selected} features "
            f"in {r.selection_time_s:.2f}s"
        )

    return all_results
```

- [ ] **Step 2: Verify the module imports correctly**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids
uv run python -c "
import sys
sys.path.insert(0, 'experiments/commons')
sys.path.insert(0, 'experiments/1')
sys.path.insert(0, 'experiments/4')
from xai_selection import run_xai_pipeline, _attributions_to_importance, _iterative_prune
import numpy as np
# Quick smoke test for _attributions_to_importance
attrs = np.random.randn(100, 10)
imp = _attributions_to_importance(attrs)
assert imp.shape == (10,), f'Expected (10,), got {imp.shape}'
assert np.all(imp >= 0), 'Importance should be non-negative'
print('xai_selection imports and smoke test OK')
"
```
Expected: `xai_selection imports and smoke test OK`

- [ ] **Step 3: Commit**

```bash
git add experiments/4/xai_selection.py
git commit -m "refactor(exp4): rewrite xai_selection to use exp1 orchestrators with parallelism"
```

---

### Task 4: Update `experiments/4/main.py` — CLI Flag, Config Loader, New XAI Pipeline

**Files:**
- Modify: `experiments/4/main.py`

Changes:
1. Use `load_experiment_config()` instead of `ExperimentConfig()`
2. Add `--xai-mode` CLI argument
3. Update `phase_select` to pass full config + dataset to new `run_xai_pipeline`
4. Update `load_or_train_baseline_models` — import `DatasetBundle` from data_loader at top
5. Remove old `from data_loader import DatasetBundle, load_dataset` from inside `run_experiment` (move to module level)

- [ ] **Step 1: Update imports and module docstring**

Replace lines 1-55 of `main.py` with:

```python
#!/usr/bin/env python3
"""Experiment 4: XAI-Driven Dimensionality Reduction vs Statistical Baselines.

Full pipeline: data loading -> feature selection (statistical + XAI) ->
downstream model training -> inference benchmarking -> metric evaluation.

Usage:
    # Full experiment on all datasets (both normal + PA-XAI)
    python experiments/4/main.py

    # Specific dataset(s)
    python experiments/4/main.py --datasets nsl-kdd cic-ids-2017

    # Specific phase
    python experiments/4/main.py --phase select       # Feature selection only
    python experiments/4/main.py --phase benchmark    # Downstream training + eval only
    python experiments/4/main.py --phase all          # Full pipeline

    # XAI mode selection
    python experiments/4/main.py --xai-mode n         # Normal XAI only
    python experiments/4/main.py --xai-mode p         # Protocol-Aware XAI only
    python experiments/4/main.py --xai-mode both      # Both modes (default from YAML)

    # Skip specific pipelines
    python experiments/4/main.py --skip-statistical
    python experiments/4/main.py --skip-xai
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

# Add experiment directories to path for local imports
# Commons and Exp1 must be added first (lower priority), then exp4 (higher priority)
_exp4_dir = os.path.dirname(os.path.abspath(__file__))
_commons_dir = os.path.join(_exp4_dir, "..", "commons")
exp1_dir = os.path.join(_exp4_dir, "..", "1")
sys.path.insert(0, _commons_dir)
sys.path.insert(0, exp1_dir)
sys.path.insert(0, _exp4_dir)

from config import ExperimentConfig, load_experiment_config
from data_loader import DatasetBundle, load_dataset as load_dataset_exp1
from evaluation import compute_reduction_summary, evaluate_downstream_model
from feature_selection import FeatureSelectionResult, run_statistical_pipeline
from models import (
    NIDSNet,
    NNWrapper,
    SKLearnWrapper,
    train_all_downstream,
)
from visualization import generate_all_plots, generate_summary_csv
from xai_selection import run_xai_pipeline
```

- [ ] **Step 2: Update `load_or_train_baseline_models` signature**

Replace the `load_or_train_baseline_models` function (lines 108-162) — remove the `DatasetBundle` type annotation import (it's now at module level) and update the type hint:

```python
def load_or_train_baseline_models(
    dataset: DatasetBundle,
    config: ExperimentConfig,
    device: torch.device,
    num_gpus: int,
) -> tuple:
    """Load Exp1 models if available, otherwise train fresh baseline DNN + RF."""
    input_dim = dataset.X_train.shape[1]
    exp1_model_dir = config.exp1_output_dir / "models" / dataset.dataset_name

    dnn_path = exp1_model_dir / "dnn.pt"
    rf_path = exp1_model_dir / "rf.joblib"

    if dnn_path.exists() and rf_path.exists():
        logger.info(f"  Loading Experiment 1 models from {exp1_model_dir}")
        dnn = NIDSNet(
            input_dim=input_dim,
            num_classes=dataset.num_classes,
            hidden_layers=config.dnn.hidden_layers,
            dropout_rate=config.dnn.dropout_rate,
        )
        dnn.load_state_dict(torch.load(dnn_path, map_location=device, weights_only=True))
        dnn = dnn.to(device)
        dnn.eval()

        if device.type == "cuda":
            with torch.no_grad():
                dnn(torch.zeros(1, input_dim, device=device))

        rf = joblib.load(rf_path)
        dnn_wrapper = NNWrapper(dnn, device)
        rf_wrapper = SKLearnWrapper(rf, dataset.num_classes)
        logger.info("  Loaded pre-trained baseline models from Experiment 1")
    else:
        logger.info("  Experiment 1 models not found. Training fresh baselines...")
        from models import train_dnn as _train_dnn, train_rf as _train_rf

        dnn, dnn_wrapper, _ = _train_dnn(
            dataset.X_train, dataset.y_train, dataset.X_val, dataset.y_val,
            dataset.num_classes, config.dnn, device, num_gpus,
        )
        rf_model, rf_wrapper, _ = _train_rf(
            dataset.X_train, dataset.y_train, dataset.num_classes, config.rf,
        )
        rf = rf_model
        # Save for reuse
        save_dir = config.output_dir / "baseline_models" / dataset.dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        base = dnn.module if isinstance(dnn, torch.nn.DataParallel) else dnn
        torch.save(base.state_dict(), save_dir / "dnn.pt")
        joblib.dump(rf, save_dir / "rf.joblib")

    return dnn, rf, dnn_wrapper, rf_wrapper
```

- [ ] **Step 3: Update `phase_select` to use new XAI pipeline signature**

Replace the `phase_select` function (lines 169-240) with:

```python
def phase_select(
    dataset: DatasetBundle,
    config: ExperimentConfig,
    device: torch.device,
    num_gpus: int,
    skip_statistical: bool = False,
    skip_xai: bool = False,
) -> list[FeatureSelectionResult]:
    """Run both feature selection pipelines."""
    logger.info(f"=== FEATURE SELECTION on {dataset.dataset_name} ===")
    log_gpu_memory("pre-select")

    output_dir = config.output_dir / dataset.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine target feature count for this dataset
    n_features = dataset.X_train.shape[1]
    target = config.xai.target_features.get(dataset.dataset_name, max(15, n_features // 4))
    logger.info(f"  Original features: {n_features}, target: ~{target}")

    all_selections: list[FeatureSelectionResult] = []

    # --- Statistical pipeline ---
    if not skip_statistical:
        stat_results = run_statistical_pipeline(
            dataset.X_train, dataset.y_train,
            dataset.feature_names, target, config.statistical,
        )
        all_selections.extend(stat_results)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- XAI pipeline (normal + PA modes via Exp 1 orchestrators) ---
    if not skip_xai:
        dnn, rf, dnn_wrapper, rf_wrapper = load_or_train_baseline_models(
            dataset, config, device, num_gpus,
        )

        xai_results = run_xai_pipeline(
            dnn, rf, dnn_wrapper, rf_wrapper,
            dataset, device, config,
        )
        all_selections.extend(xai_results)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save selection results
    sel_dir = output_dir / "selections"
    sel_dir.mkdir(parents=True, exist_ok=True)
    for sel in all_selections:
        np.save(sel_dir / f"{sel.method_name}_indices.npy", sel.selected_indices)
        np.save(sel_dir / f"{sel.method_name}_rankings.npy", sel.feature_rankings)

    sel_summary = [{
        "method_name": s.method_name,
        "n_original": s.n_original,
        "n_selected": s.n_selected,
        "selection_time_s": s.selection_time_s,
        "selected_features": s.selected_feature_names,
    } for s in all_selections]

    with open(sel_dir / "selection_summary.json", "w") as f:
        json.dump(sel_summary, f, indent=2, default=_json_serialize)

    logger.info(f"  Feature selection results saved to {sel_dir}")
    return all_selections
```

- [ ] **Step 4: Remove inline `from data_loader import ...` in `run_experiment`**

In the `run_experiment` function, remove lines 405-406:
```python
        # DELETE these two lines:
        from data_loader import DatasetBundle, load_dataset as load_dataset_exp1
```

These imports are now at module level (Step 1).

- [ ] **Step 5: Update `parse_args` to add --xai-mode flag**

Replace the `parse_args` function (lines 522-560) with:

```python
def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 4: XAI-Driven Dimensionality Reduction vs Statistical Baselines"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        choices=["nsl-kdd", "cic-ids-2017", "unsw-nb15", "cse-cic-ids2018"],
        help="Datasets to process (default: all)",
    )
    parser.add_argument(
        "--phase", nargs="+", default=["all"],
        choices=["all", "select", "benchmark"],
        help="Experiment phase(s) to run",
    )
    parser.add_argument(
        "--xai-mode", type=str, default=None,
        choices=["n", "p", "both"],
        help="XAI mode: n (normal), p (protocol-aware), both (default from config.yaml)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-statistical", action="store_true",
        help="Skip statistical feature selection pipeline",
    )
    parser.add_argument(
        "--skip-xai", action="store_true",
        help="Skip XAI-driven feature selection pipeline",
    )
    parser.add_argument(
        "--no-smote", action="store_true",
        help="Disable SMOTE oversampling",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default from config.yaml)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: experiments/commons/config.yaml)",
    )
    return parser.parse_args()
```

- [ ] **Step 6: Update `main` to use load_experiment_config and --xai-mode**

Replace the `main` function (lines 563-595) with:

```python
def main():
    args = parse_args()
    config = load_experiment_config(args.config)

    # CLI overrides
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.no_smote:
        config.data.apply_smote = False
    if args.seed is not None:
        config.seed = args.seed
    if args.xai_mode:
        if args.xai_mode == "both":
            config.xai_modes = ["normal", "pa"]
        elif args.xai_mode == "n":
            config.xai_modes = ["normal"]
        elif args.xai_mode == "p":
            config.xai_modes = ["pa"]

    datasets = args.datasets or config.ALL_DATASETS
    phases = args.phase

    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger.info("Experiment 4: XAI-Driven Dimensionality Reduction vs Statistical Baselines")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Phases: {phases}")
    logger.info(f"XAI modes: {config.xai_modes}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Skip statistical: {args.skip_statistical}")
    logger.info(f"Skip XAI: {args.skip_xai}")

    run_experiment(config, datasets, phases, args.skip_statistical, args.skip_xai)
```

- [ ] **Step 7: Update `run_experiment` to log xai_modes in saved config**

In the `run_experiment` function, update the config save block (around line 383-393) to include `xai_modes`:

Replace:
```python
    with open(config.output_dir / "config.json", "w") as f:
        json.dump({
            "datasets": datasets,
            "phases": phases,
            "dnn": vars(config.dnn),
            "cnn": vars(config.cnn),
            "rf": vars(config.rf),
            "svm": vars(config.svm),
            "statistical": vars(config.statistical),
            "xai": vars(config.xai),
            "seed": config.seed,
        }, f, indent=2, default=_json_serialize)
```

With:
```python
    with open(config.output_dir / "config.json", "w") as f:
        json.dump({
            "datasets": datasets,
            "phases": phases,
            "xai_modes": config.xai_modes,
            "dnn": vars(config.dnn),
            "cnn": vars(config.cnn),
            "rf": vars(config.rf),
            "svm": vars(config.svm),
            "explainer": vars(config.explainer),
            "statistical": vars(config.statistical),
            "xai": vars(config.xai),
            "seed": config.seed,
        }, f, indent=2, default=_json_serialize)
```

- [ ] **Step 8: Verify main.py imports cleanly**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids
uv run python -c "
import sys
sys.path.insert(0, 'experiments/commons')
sys.path.insert(0, 'experiments/1')
sys.path.insert(0, 'experiments/4')
from main import parse_args
print('main.py imports OK')
"
```
Expected: `main.py imports OK`

- [ ] **Step 9: Commit**

```bash
git add experiments/4/main.py
git commit -m "feat(exp4): add --xai-mode flag, YAML config loading, exp1 orchestrator integration"
```

---

### Task 5: Update Visualization to Distinguish Normal vs PA Methods

**Files:**
- Modify: `experiments/4/visualization.py:39-43`

The existing `pipeline` classification logic needs to recognize PA-prefixed XAI methods. Currently it checks if the method is in `{"Chi-Squared", "PCA", "Spearman", "InfoGain", "Full"}` and labels everything else as "XAI-Driven". We need a third category: "PA-XAI-Driven".

- [ ] **Step 1: Update pipeline classification in generate_all_plots**

In `experiments/4/visualization.py`, replace lines 39-46:

```python
    stat_methods = {"Chi-Squared", "PCA", "Spearman", "InfoGain", "Full"}
    df["pipeline"] = df["selection_method"].apply(
        lambda m: "Statistical" if m in stat_methods else "XAI-Driven"
    )
    df.loc[df["selection_method"] == "Full", "pipeline"] = "Baseline"

    # Color palette
    palette = {"Baseline": "#666666", "Statistical": "#2196F3", "XAI-Driven": "#FF5722"}
```

With:

```python
    stat_methods = {"Chi-Squared", "PCA", "Spearman", "InfoGain", "Full"}

    def _classify_pipeline(method: str) -> str:
        if method in stat_methods:
            return "Statistical"
        if method == "Full":
            return "Baseline"
        if method.startswith("PA-"):
            return "PA-XAI"
        return "XAI"

    df["pipeline"] = df["selection_method"].apply(_classify_pipeline)

    # Color palette
    palette = {
        "Baseline": "#666666",
        "Statistical": "#2196F3",
        "XAI": "#FF5722",
        "PA-XAI": "#4CAF50",
    }
```

- [ ] **Step 2: Apply same classification in feature selection plot section**

In `experiments/4/visualization.py`, replace line 146:

```python
                "pipeline": "Statistical" if sel["method_name"] in stat_methods else "XAI-Driven",
```

With:

```python
                "pipeline": _classify_pipeline(sel["method_name"]),
```

Note: `_classify_pipeline` is defined inside `generate_all_plots`, so move it to module level or define it before the feature selection section. The simplest approach: move the function and `stat_methods` to module level at the top of the function body (before both usages).

- [ ] **Step 3: Commit**

```bash
git add experiments/4/visualization.py
git commit -m "feat(exp4): distinguish normal vs PA-XAI methods in visualization"
```

---

### Task 6: End-to-End Smoke Test

**Files:** None (verification only)

- [ ] **Step 1: Verify full import chain works**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids
uv run python -c "
import sys
sys.path.insert(0, 'experiments/commons')
sys.path.insert(0, 'experiments/1')
sys.path.insert(0, 'experiments/4')

from config import load_experiment_config
from xai_selection import run_xai_pipeline, _attributions_to_importance, _iterative_prune
from feature_selection import FeatureSelectionResult, run_statistical_pipeline
from evaluation import evaluate_downstream_model, compute_reduction_summary
from visualization import generate_all_plots, generate_summary_csv
from models import train_all_downstream, NIDSNet, NNWrapper, SKLearnWrapper

config = load_experiment_config()
print(f'Config loaded: xai_modes={config.xai_modes}, output={config.output_dir}')
print(f'Explainer config: bg={config.explainer.shap_background_samples}, lime_samples={config.explainer.lime_num_samples}')
print(f'XAI selection: f1_thresh={config.xai.f1_degradation_threshold}, targets={config.xai.target_features}')
print('All imports OK')
"
```

Expected: All imports succeed, config values match YAML.

- [ ] **Step 2: Verify CLI help works**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids
uv run python experiments/4/main.py --help
```

Expected: Help output includes `--xai-mode {n,p,both}` and `--config` arguments.

- [ ] **Step 3: Commit — final verification commit (if any fixes needed)**

If any fixes were needed during verification:
```bash
git add -A experiments/4/
git commit -m "fix(exp4): address smoke test issues"
```
