# Experiment 3 Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overhaul Experiment 3 to support unified normal+PA explanation pools with reuse from Experiment 1, resume/checkpoint on crash, YAML config, and full parallelism for 64-core/2xL40 hardware.

**Architecture:** A shared `experiments/config.yaml` feeds all experiments via `experiments/config_loader.py`. Experiment 3 loads explanations from exp1 output dirs (normal + protocol-aware) into a unified dict with prefixed keys, computes consensus/interactions/alignment over all pairs in parallel, and checkpoints every phase and sub-phase for crash recovery.

**Tech Stack:** Python 3.13, PyYAML, torch, numpy, scipy, shap, concurrent.futures (ProcessPoolExecutor/ThreadPoolExecutor)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Modify | Add `pyyaml` dependency |
| `experiments/config.yaml` | Create | Shared YAML config (single seed, all experiment sections) |
| `experiments/config_loader.py` | Create | YAML reader that populates existing dataclass trees |
| `experiments/3/config.py` | Modify | Add `ParallelismConfig`, `xai_modes` field, alignment `enabled` flag |
| `experiments/3/main.py` | Modify | Explanation reuse, PA mode, dataset parallelism, checkpoint system |
| `experiments/3/consensus.py` | Modify | Pair-level checkpointing + ProcessPoolExecutor |
| `experiments/3/interactions.py` | Modify | Batch DNN perturbations into single SHAP call |
| `experiments/3/visualizations.py` | Modify | ThreadPool for parallel plot generation |

---

### Task 1: Add PyYAML Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add pyyaml to dependencies**

In `pyproject.toml`, add `"pyyaml>=6.0"` to the `dependencies` list, after the existing entries:

```toml
    "tqdm>=4.66.0",
    "joblib>=1.3.0",
    "imblearn>=0.0",
    "pyyaml>=6.0",
```

- [ ] **Step 2: Install**

Run: `uv sync`
Expected: Resolves and installs pyyaml.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add pyyaml for shared YAML config"
```

---

### Task 2: Create Shared YAML Config

**Files:**
- Create: `experiments/config.yaml`

- [ ] **Step 1: Create the YAML file**

```yaml
# Shared configuration for all experiments.
# One seed to rule them all — every experiment reads from here.

seed: 42
datasets:
  - nsl-kdd
  - cic-ids-2017
  - unsw-nb15
  - cse-cic-ids2018

data:
  data_root: /home/resbears/Downloads/data
  test_size: 0.2
  val_split: 0.1
  apply_smote: true

models:
  dnn:
    hidden_layers: [1024, 768, 512]
    dropout_rate: 0.01
    learning_rate: 0.01
    batch_size: 8192
    epochs: 100
    early_stopping_patience: 10
  rf:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 2
    criterion: gini
    n_jobs: -1

explainer:
  shap_background_samples: 100
  lime_num_features: 10
  lime_num_samples: 5000
  ig_n_steps: 50
  ig_internal_batch_size: 8192
  num_explain_samples: 10000

experiment_1:
  output_dir: experiments/1/results
  models_dir: experiments/1/results

experiment_3:
  output_dir: experiments/3/results
  xai_modes:
    - normal
    - pa

  consensus:
    top_k_values: [5, 10]
    alpha: 0.05
    num_explain_samples: 10000

  interactions:
    top_n_interactions: 20
    shap_interaction_samples: 500
    shap_background_samples: 100

  alignment:
    enabled: true
    top_k_rra: [5, 10, 15]

  parallelism:
    max_dataset_workers: 4
    max_consensus_workers: 16
    max_plot_workers: 8
    num_gpu_devices: 2
```

- [ ] **Step 2: Commit**

```bash
git add experiments/config.yaml
git commit -m "config: add shared YAML config for all experiments"
```

---

### Task 3: Create Config Loader

**Files:**
- Create: `experiments/config_loader.py`

- [ ] **Step 1: Write the config loader**

This module reads `experiments/config.yaml` and returns a plain dict. Each experiment's `config.py` is responsible for populating its own dataclasses from the dict. The loader also computes a config hash for checkpoint invalidation.

```python
"""Shared YAML configuration loader for all experiments.

Usage:
    from config_loader import load_yaml_config
    raw = load_yaml_config()                     # default path
    raw = load_yaml_config("path/to/config.yaml")
"""

import hashlib
import json
from pathlib import Path

import yaml


_DEFAULT_PATH = Path(__file__).parent / "config.yaml"


def load_yaml_config(path: str | Path | None = None) -> dict:
    """Load and return the raw YAML config as a dict.

    Args:
        path: Path to YAML file. Defaults to experiments/config.yaml.

    Returns:
        Parsed YAML dict.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p) as f:
        return yaml.safe_load(f)


def config_section_hash(section: dict) -> str:
    """Compute a deterministic SHA-256 hash of a config section.

    Used by the checkpoint system to detect config changes.
    """
    canonical = json.dumps(section, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

- [ ] **Step 2: Verify it loads**

Run: `uv run python -c "from experiments.config_loader import load_yaml_config; c = load_yaml_config(); print(c['seed'], list(c['experiment_3'].keys()))"`
Expected: `42 ['output_dir', 'xai_modes', 'consensus', 'interactions', 'alignment', 'parallelism']`

- [ ] **Step 3: Commit**

```bash
git add experiments/config_loader.py
git commit -m "feat: add shared YAML config loader with section hashing"
```

---

### Task 4: Update Experiment 3 Config Dataclasses

**Files:**
- Modify: `experiments/3/config.py`

- [ ] **Step 1: Add ParallelismConfig and update Experiment3Config**

Replace the entire `experiments/3/config.py` with:

```python
"""Configuration for Experiment 3: Feature Interaction, Consensus, and Expert Alignment."""

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

# Import shared configs from experiment 1 via importlib to avoid circular import
# (both exp1 and exp3 have a module named "config")
_exp1_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1", "config.py")
_spec = importlib.util.spec_from_file_location("exp1_config", _exp1_config_path)
_exp1_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp1_config)

DataConfig = _exp1_config.DataConfig
DNNConfig = _exp1_config.DNNConfig
RFConfig = _exp1_config.RFConfig
ExplainerConfig = _exp1_config.ExplainerConfig


@dataclass
class ConsensusConfig:
    """Configuration for explainer consensus analysis."""

    top_k_values: list[int] = field(default_factory=lambda: [5, 10])
    num_explain_samples: int = 10000
    alpha: float = 0.05  # Wilcoxon significance threshold


@dataclass
class InteractionConfig:
    """Configuration for feature interaction analysis."""

    top_n_interactions: int = 20
    shap_interaction_samples: int = 500  # SHAP interaction values are O(n * f^2)
    shap_background_samples: int = 100


@dataclass
class AlignmentConfig:
    """Configuration for expert alignment scoring."""

    enabled: bool = True
    top_k_rra: list[int] = field(default_factory=lambda: [5, 10, 15])


@dataclass
class ParallelismConfig:
    """Configuration for parallel execution."""

    max_dataset_workers: int = 4
    max_consensus_workers: int = 16
    max_plot_workers: int = 8
    num_gpu_devices: int = 2


@dataclass
class Experiment3Config:
    data: DataConfig = field(default_factory=DataConfig)
    dnn: DNNConfig = field(default_factory=DNNConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    output_dir: Path = Path("experiments/3/results")
    exp1_output_dir: Path = Path("experiments/1/results")
    xai_modes: list[str] = field(default_factory=lambda: ["normal", "pa"])
    seed: int = 42

    ALL_DATASETS: list[str] = field(
        default_factory=lambda: ["nsl-kdd", "cic-ids-2017", "unsw-nb15", "cse-cic-ids2018"]
    )


def load_experiment3_config(yaml_path: str | Path | None = None) -> Experiment3Config:
    """Load Experiment3Config from YAML, falling back to dataclass defaults.

    Args:
        yaml_path: Path to shared YAML config. None uses default location.

    Returns:
        Populated Experiment3Config.
    """
    # Import here to avoid circular issues at module level
    import sys
    loader_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if loader_dir not in sys.path:
        sys.path.insert(0, loader_dir)
    from config_loader import load_yaml_config

    try:
        raw = load_yaml_config(yaml_path)
    except FileNotFoundError:
        return Experiment3Config()

    config = Experiment3Config()
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

    # Model configs
    models_raw = raw.get("models", {})
    dnn_raw = models_raw.get("dnn", {})
    for k, v in dnn_raw.items():
        if hasattr(config.dnn, k):
            setattr(config.dnn, k, v)
    rf_raw = models_raw.get("rf", {})
    for k, v in rf_raw.items():
        if hasattr(config.rf, k):
            setattr(config.rf, k, v)

    # Explainer config
    exp_raw = raw.get("explainer", {})
    for k, v in exp_raw.items():
        if hasattr(config.explainer, k):
            setattr(config.explainer, k, v)

    # Experiment 3 specific
    exp3_raw = raw.get("experiment_3", {})
    if "output_dir" in exp3_raw:
        config.output_dir = Path(exp3_raw["output_dir"])
    if "xai_modes" in exp3_raw:
        config.xai_modes = exp3_raw["xai_modes"]

    # Exp1 output dir (for reuse)
    exp1_raw = raw.get("experiment_1", {})
    if "output_dir" in exp1_raw:
        config.exp1_output_dir = Path(exp1_raw["output_dir"])

    # Consensus
    cons_raw = exp3_raw.get("consensus", {})
    for k, v in cons_raw.items():
        if hasattr(config.consensus, k):
            setattr(config.consensus, k, v)

    # Interactions
    inter_raw = exp3_raw.get("interactions", {})
    for k, v in inter_raw.items():
        if hasattr(config.interaction, k):
            setattr(config.interaction, k, v)

    # Alignment
    align_raw = exp3_raw.get("alignment", {})
    for k, v in align_raw.items():
        if hasattr(config.alignment, k):
            setattr(config.alignment, k, v)

    # Parallelism
    par_raw = exp3_raw.get("parallelism", {})
    for k, v in par_raw.items():
        if hasattr(config.parallelism, k):
            setattr(config.parallelism, k, v)

    # Datasets
    if "datasets" in raw:
        config.ALL_DATASETS = raw["datasets"]

    return config
```

- [ ] **Step 2: Verify loading works**

Run: `cd /Users/abanisenioluwaorojo/projects/sok-nids && uv run python -c "
import sys; sys.path.insert(0, 'experiments/3')
from config import load_experiment3_config
c = load_experiment3_config()
print('seed:', c.seed)
print('xai_modes:', c.xai_modes)
print('alignment enabled:', c.alignment.enabled)
print('max_consensus_workers:', c.parallelism.max_consensus_workers)
print('datasets:', c.ALL_DATASETS)
"`

Expected:
```
seed: 42
xai_modes: ['normal', 'pa']
alignment enabled: True
max_consensus_workers: 16
datasets: ['nsl-kdd', 'cic-ids-2017', 'unsw-nb15', 'cse-cic-ids2018']
```

- [ ] **Step 3: Commit**

```bash
git add experiments/3/config.py
git commit -m "feat(exp3): add ParallelismConfig, YAML loading, xai_modes and alignment toggle"
```

---

### Task 5: Checkpoint Utilities

**Files:**
- Modify: `experiments/3/main.py` (add checkpoint helpers at the top, before phase functions)

- [ ] **Step 1: Add checkpoint utility functions**

Add these after the existing imports and before `setup_device()` in `experiments/3/main.py`:

```python
import hashlib
import json as json_module  # avoid shadowing
from datetime import datetime, timezone

# Add config_loader to path
_experiments_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _experiments_dir not in sys.path:
    sys.path.insert(0, _experiments_dir)
from config_loader import config_section_hash


def _checkpoint_path(output_dir: Path, dataset_name: str, phase_name: str) -> Path:
    """Return the path for a phase checkpoint marker."""
    return output_dir / dataset_name / f".phase_{phase_name}.done"


def _is_phase_done(output_dir: Path, dataset_name: str, phase_name: str, config_hash: str) -> bool:
    """Check if a phase has already completed with the same config."""
    marker = _checkpoint_path(output_dir, dataset_name, phase_name)
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text())
        return data.get("config_hash") == config_hash
    except (json.JSONDecodeError, KeyError):
        return False


def _mark_phase_done(output_dir: Path, dataset_name: str, phase_name: str, config_hash: str) -> None:
    """Write a phase completion marker."""
    marker = _checkpoint_path(output_dir, dataset_name, phase_name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
    }))
```

- [ ] **Step 2: Commit**

```bash
git add experiments/3/main.py
git commit -m "feat(exp3): add phase checkpoint utilities for crash resume"
```

---

### Task 6: Explanation Reuse and PA Mode in main.py

This is the largest task. It replaces `phase_explain` with a new version that:
1. Supports both normal and PA modes
2. Loads from exp3 cache first, then exp1, then generates fresh
3. Merges into a unified pool with prefixed keys
4. Validates index alignment across modes

**Files:**
- Modify: `experiments/3/main.py`

- [ ] **Step 1: Update imports and sys.path setup**

Replace the existing sys.path/import block (lines 35-73) with:

```python
# Add experiment directories to path for local imports.
_exp3_dir = os.path.dirname(os.path.abspath(__file__))
_exp1_dir = os.path.join(_exp3_dir, "..", "1")
_experiments_dir = os.path.join(_exp3_dir, "..")
for _d in [_exp1_dir, _exp3_dir, _experiments_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from alignment import alignment_to_dict, compute_alignment_scores
from config import Experiment3Config, load_experiment3_config
from config_loader import config_section_hash
from consensus import (
    compute_pairwise_consensus,
    compute_per_attack_consensus,
    consensus_to_dict,
)
from data_loader import DatasetBundle, load_dataset
from explainers import (
    ExplanationResult,
    explain_deeplift,
    explain_ig,
    explain_lime,
    explain_shap_dnn,
    explain_shap_rf,
)
from interactions import (
    aggregate_interaction_matrix,
    compare_interaction_vs_main_effects,
    compute_shap_interaction_values_dnn,
    compute_shap_interaction_values_rf,
    get_top_interactions,
)
from models import (
    DNNWrapper,
    NIDSNet,
    RFWrapper,
    SoftmaxModel,
    load_models,
    save_models,
    train_dnn,
    train_rf,
)
from pa_explainers import (
    pa_explain_deeplift,
    pa_explain_ig,
    pa_explain_lime,
    pa_explain_shap_dnn,
    pa_explain_shap_tree,
    pa_generate_all_explanations,
)
from visualizations import generate_all_plots
```

- [ ] **Step 2: Replace phase_explain with unified load/generate function**

Replace the existing `phase_explain` and `_load_explanations` functions with:

```python
# ============================================================================
# Explanation key naming
# ============================================================================

# Exp1 saves files as "{model}_{method}_attributions.npy" in mode-specific dirs.
# In exp3's unified pool, PA keys get "PA-" prefix on the method name.
# Normal: DNN_SHAP, DNN_LIME, DNN_IG, DNN_DeepLIFT, RF_SHAP, RF_LIME
# PA:     DNN_PA-SHAP, DNN_PA-LIME, DNN_PA-IG, DNN_PA-DeepLIFT, RF_PA-SHAP, RF_PA-LIME

NORMAL_KEYS = ["DNN_SHAP", "DNN_LIME", "DNN_IG", "DNN_DeepLIFT", "RF_SHAP", "RF_LIME"]
PA_KEYS = ["DNN_PA-SHAP", "DNN_PA-LIME", "DNN_PA-IG", "DNN_PA-DeepLIFT", "RF_PA-SHAP", "RF_PA-LIME"]

# Mapping from exp3 unified key -> exp1 on-disk filename (without _attributions.npy)
# Exp1 PA mode saves "DNN_SHAP" (not "DNN_PA-SHAP") in the protocol-aware/ dir.
_EXP1_FILENAME_MAP = {
    "DNN_SHAP": "DNN_SHAP", "DNN_LIME": "DNN_LIME",
    "DNN_IG": "DNN_IG", "DNN_DeepLIFT": "DNN_DeepLIFT",
    "RF_SHAP": "RF_SHAP", "RF_LIME": "RF_LIME",
    "DNN_PA-SHAP": "DNN_SHAP", "DNN_PA-LIME": "DNN_LIME",
    "DNN_PA-IG": "DNN_IG", "DNN_PA-DeepLIFT": "DNN_DeepLIFT",
    "RF_PA-SHAP": "RF_SHAP", "RF_PA-LIME": "RF_LIME",
}


def _exp1_dir_for_mode(exp1_output_dir: Path, mode: str) -> Path:
    """Return the exp1 output directory for a given XAI mode."""
    if mode == "pa":
        return exp1_output_dir / "protocol-aware"
    return exp1_output_dir / "normal"


def _load_explanations_for_mode(
    exp3_explain_dir: Path,
    exp1_explain_dir: Path | None,
    keys: list[str],
    dataset_name: str,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Try to load explanations for a set of keys.

    Load priority: exp3 cache -> exp1 output -> return what we found.

    Returns:
        Tuple of (loaded_explanations_dict, explain_indices_or_None).
    """
    loaded: dict[str, np.ndarray] = {}
    indices = None

    # Try loading indices
    if (exp3_explain_dir / "explain_indices.npy").exists():
        indices = np.load(exp3_explain_dir / "explain_indices.npy")
    elif exp1_explain_dir and (exp1_explain_dir / "explain_indices.npy").exists():
        indices = np.load(exp1_explain_dir / "explain_indices.npy")

    for key in keys:
        # Try exp3 cache first
        exp3_path = exp3_explain_dir / f"{key}_attributions.npy"
        if exp3_path.exists():
            loaded[key] = np.load(exp3_path)
            continue

        # Try exp1 output (different filename for PA keys)
        if exp1_explain_dir:
            exp1_filename = _EXP1_FILENAME_MAP.get(key, key)
            exp1_path = exp1_explain_dir / f"{exp1_filename}_attributions.npy"
            if exp1_path.exists():
                loaded[key] = np.load(exp1_path)
                # Cache in exp3 dir for next run
                exp3_explain_dir.mkdir(parents=True, exist_ok=True)
                np.save(exp3_path, loaded[key])
                logger.info(f"    Loaded {key} from exp1 and cached in exp3")
                continue

    return loaded, indices


def phase_explain(
    dataset: DatasetBundle,
    config: Experiment3Config,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load or generate XAI explanations for all configured modes.

    Returns:
        Tuple of (unified_explanations_dict, explain_indices).
        Keys follow the unified naming: DNN_SHAP, DNN_PA-SHAP, etc.
    """
    logger.info(f"=== EXPLAINING on {dataset.dataset_name} ===")

    unified: dict[str, np.ndarray] = {}
    shared_indices: np.ndarray | None = None

    for mode in config.xai_modes:
        keys = PA_KEYS if mode == "pa" else NORMAL_KEYS
        mode_label = "protocol-aware" if mode == "pa" else "normal"
        logger.info(f"--- Loading/generating {mode_label} explanations ---")

        exp3_explain_dir = config.output_dir / dataset.dataset_name / "explanations" / mode_label
        exp3_explain_dir.mkdir(parents=True, exist_ok=True)

        exp1_explain_dir = _exp1_dir_for_mode(config.exp1_output_dir, mode) / dataset.dataset_name / "explanations"

        # Try loading cached explanations
        loaded, loaded_indices = _load_explanations_for_mode(
            exp3_explain_dir, exp1_explain_dir, keys, dataset.dataset_name,
        )

        if loaded_indices is not None:
            if shared_indices is not None and not np.array_equal(shared_indices, loaded_indices):
                logger.error(
                    f"  Index mismatch between modes for {dataset.dataset_name}! "
                    f"Normal has {len(shared_indices)} indices, {mode_label} has {len(loaded_indices)}. "
                    f"Cross-mode consensus will be skipped."
                )
                # Still use them but flag the mismatch
            if shared_indices is None:
                shared_indices = loaded_indices

        missing_keys = [k for k in keys if k not in loaded]

        if not missing_keys:
            logger.info(f"  All {mode_label} explanations loaded from cache ({len(loaded)} keys)")
            unified.update(loaded)
            continue

        logger.info(f"  Missing {len(missing_keys)} {mode_label} explanations: {missing_keys}")
        logger.info(f"  Generating missing explanations...")

        # Need to generate — load models
        dnn_model, rf_model = load_models(
            config.exp1_output_dir, dataset.dataset_name,
            dataset.X_train.shape[1], dataset.num_classes,
            config.dnn, device,
        )
        dnn_wrapper = DNNWrapper(dnn_model, device)
        rf_wrapper = RFWrapper(rf_model, num_classes=dataset.num_classes)

        # Determine indices (reuse loaded or generate)
        if shared_indices is not None:
            indices = shared_indices
        else:
            n = min(config.consensus.num_explain_samples, len(dataset.X_test))
            rng = np.random.RandomState(config.seed)
            indices = rng.choice(len(dataset.X_test), size=n, replace=False)
            shared_indices = indices

        X_explain = dataset.X_test[indices]

        # Background data
        rng_bg = np.random.RandomState(config.seed)
        # Advance RNG state past the indices draw to stay in sync with exp1
        rng_bg.choice(len(dataset.X_test), size=len(indices), replace=False)
        bg_indices = rng_bg.choice(len(dataset.X_train), size=config.explainer.shap_background_samples, replace=False)
        X_background = dataset.X_train[bg_indices]

        checkpoint_dir = config.output_dir / dataset.dataset_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Generate each missing key
        for key in missing_keys:
            try:
                result = _generate_single_explanation(
                    key, mode, dnn_model, rf_model, dnn_wrapper, rf_wrapper,
                    X_explain, X_background, dataset, device, config, checkpoint_dir,
                )
                if result is not None:
                    loaded[key] = result.attributions
                    np.save(exp3_explain_dir / f"{key}_attributions.npy", result.attributions)
                    logger.info(f"    {key}: {result.attributions.shape}, {result.time_per_sample_ms:.2f} ms/sample")
            except Exception as e:
                logger.error(f"    {key} failed: {e}", exc_info=True)

        # Save indices for this mode
        np.save(exp3_explain_dir / "explain_indices.npy", indices)

        unified.update(loaded)

    if shared_indices is None:
        raise RuntimeError(f"No explanations could be loaded or generated for {dataset.dataset_name}")

    # Save unified indices and labels at dataset level
    ds_dir = config.output_dir / dataset.dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)
    np.save(ds_dir / "explain_indices.npy", shared_indices)
    np.save(ds_dir / "explain_labels.npy", dataset.y_test[shared_indices])

    logger.info(f"  Unified pool: {len(unified)} explanation sets")
    return unified, shared_indices


def _generate_single_explanation(
    key: str,
    mode: str,
    dnn_model,
    rf_model,
    dnn_wrapper,
    rf_wrapper,
    X_explain: np.ndarray,
    X_background: np.ndarray,
    dataset: DatasetBundle,
    device: torch.device,
    config: Experiment3Config,
    checkpoint_dir: Path,
) -> ExplanationResult | None:
    """Generate a single explanation by key name."""
    ds_name = dataset.dataset_name

    if mode == "pa":
        # PA mode generators
        generators = {
            "DNN_PA-SHAP": lambda: pa_explain_shap_dnn(
                dnn_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, device, config.explainer,
            ),
            "DNN_PA-LIME": lambda: pa_explain_lime(
                dnn_wrapper.predict_proba, X_explain, dataset.X_train,
                ds_name, "DNN", config.explainer, checkpoint_dir,
            ),
            "DNN_PA-IG": lambda: pa_explain_ig(
                dnn_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, device, config.explainer,
            ),
            "DNN_PA-DeepLIFT": lambda: pa_explain_deeplift(
                dnn_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, device, config.explainer,
            ),
            "RF_PA-SHAP": lambda: pa_explain_shap_tree(
                rf_model, X_explain, dataset.X_train, dataset.y_train,
                ds_name, config.explainer, "RF",
            ),
            "RF_PA-LIME": lambda: pa_explain_lime(
                rf_wrapper.predict_proba, X_explain, dataset.X_train,
                ds_name, "RF", config.explainer, checkpoint_dir,
            ),
        }
    else:
        # Normal mode generators
        generators = {
            "DNN_SHAP": lambda: explain_shap_dnn(
                dnn_model, X_explain, X_background, device, config.explainer,
            ),
            "DNN_LIME": lambda: explain_lime(
                dnn_wrapper.predict_proba, X_explain, dataset.X_train,
                dataset.feature_names, dataset.num_classes, "DNN", config.explainer,
            ),
            "DNN_IG": lambda: explain_ig(dnn_model, X_explain, device, config.explainer),
            "DNN_DeepLIFT": lambda: explain_deeplift(dnn_model, X_explain, device, config.explainer),
            "RF_SHAP": lambda: explain_shap_rf(rf_model, X_explain, config.explainer),
            "RF_LIME": lambda: explain_lime(
                rf_wrapper.predict_proba, X_explain, dataset.X_train,
                dataset.feature_names, dataset.num_classes, "RF", config.explainer,
            ),
        }

    gen_fn = generators.get(key)
    if gen_fn is None:
        logger.warning(f"  No generator for key: {key}")
        return None

    logger.info(f"    Generating {key}...")
    return gen_fn()
```

- [ ] **Step 3: Update run_experiment to use checkpointing and the new phase_explain**

Replace the `run_experiment` function body with checkpoint-aware logic. The key changes:
- Load config from YAML
- Check phase markers before running each phase
- Mark phases done after completion
- Use `config.xai_modes` to control which explanations are in the pool
- Skip alignment if `config.alignment.enabled` is False

```python
def run_experiment(config: Experiment3Config, datasets: list[str], phases: list[str]):
    """Main experiment runner with checkpoint support."""
    device, num_gpus = setup_device()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    all_phases = {"train", "explain", "consensus", "interactions", "alignment", "visualize"}
    run_all = "all" in phases
    active_phases = all_phases if run_all else set(phases)

    # Remove alignment if disabled
    if not config.alignment.enabled:
        active_phases.discard("alignment")

    # Save config
    with open(config.output_dir / "config.json", "w") as f:
        json.dump({
            "datasets": datasets,
            "phases": list(active_phases),
            "xai_modes": config.xai_modes,
            "consensus": vars(config.consensus),
            "interaction": vars(config.interaction),
            "alignment": vars(config.alignment),
            "parallelism": vars(config.parallelism),
            "seed": config.seed,
        }, f, indent=2, default=_json_serialize)

    all_results = {}
    for ds_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"DATASET: {ds_name}")
        logger.info(f"{'='*60}")
        ds_start = time.time()

        try:
            dataset = load_dataset(ds_name, config.data)
        except Exception as e:
            logger.error(f"Failed to load {ds_name}: {e}", exc_info=True)
            continue

        ds_output = config.output_dir / ds_name
        ds_output.mkdir(parents=True, exist_ok=True)
        ds_results = {}

        # Phase: Train
        if "train" in active_phases:
            train_hash = config_section_hash({"dnn": vars(config.dnn), "rf": vars(config.rf), "seed": config.seed})
            if _is_phase_done(config.output_dir, ds_name, "train", train_hash):
                logger.info("  Train phase: already done (checkpoint found)")
            else:
                try:
                    phase_train(dataset, config, device, num_gpus)
                    _mark_phase_done(config.output_dir, ds_name, "train", train_hash)
                except Exception as e:
                    logger.error(f"Training failed for {ds_name}: {e}", exc_info=True)
                    continue
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Phase: Explain
        explanations = None
        explain_indices = None
        y_explain = None

        if "explain" in active_phases:
            for mode in config.xai_modes:
                explain_hash = config_section_hash({
                    "explainer": vars(config.explainer),
                    "consensus_samples": config.consensus.num_explain_samples,
                    "seed": config.seed, "mode": mode,
                })
                phase_label = f"explain_{mode}"
                if _is_phase_done(config.output_dir, ds_name, phase_label, explain_hash):
                    logger.info(f"  Explain ({mode}): already done (checkpoint found)")
                    continue
                # Not done — will be generated in phase_explain below

            try:
                explanations, explain_indices = phase_explain(dataset, config, device)
                y_explain = dataset.y_test[explain_indices]
                # Mark all explain phases done
                for mode in config.xai_modes:
                    explain_hash = config_section_hash({
                        "explainer": vars(config.explainer),
                        "consensus_samples": config.consensus.num_explain_samples,
                        "seed": config.seed, "mode": mode,
                    })
                    _mark_phase_done(config.output_dir, ds_name, f"explain_{mode}", explain_hash)
            except Exception as e:
                logger.error(f"Explanation failed for {ds_name}: {e}", exc_info=True)
                continue
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Load explanations if not generated in this run
        if explanations is None:
            try:
                explanations, explain_indices, y_explain = _load_all_explanations(
                    config.output_dir / ds_name, config.xai_modes,
                )
                logger.info(f"  Loaded {len(explanations)} explanation sets from disk")
            except Exception as e:
                logger.error(f"Could not load explanations for {ds_name}: {e}")
                if any(p in active_phases for p in ["consensus", "interactions", "alignment", "visualize"]):
                    logger.error("  Cannot proceed without explanations. Skipping remaining phases.")
                    continue

        # Phase: Consensus
        consensus_results = None
        if "consensus" in active_phases and explanations:
            cons_hash = config_section_hash({"consensus": vars(config.consensus), "keys": sorted(explanations.keys())})
            if _is_phase_done(config.output_dir, ds_name, "consensus", cons_hash):
                logger.info("  Consensus phase: already done (checkpoint found)")
                # Load results for later phases
                try:
                    with open(ds_output / "consensus_results.json") as f:
                        consensus_results = json.load(f)
                except Exception:
                    pass
            else:
                try:
                    consensus_results = phase_consensus(dataset, config, explanations, y_explain)
                    ds_results["consensus"] = consensus_results
                    _mark_phase_done(config.output_dir, ds_name, "consensus", cons_hash)
                except Exception as e:
                    logger.error(f"Consensus failed for {ds_name}: {e}", exc_info=True)

        # Phase: Interactions
        interaction_results = None
        if "interactions" in active_phases and explanations:
            inter_hash = config_section_hash({"interaction": vars(config.interaction), "seed": config.seed})
            if _is_phase_done(config.output_dir, ds_name, "interactions", inter_hash):
                logger.info("  Interactions phase: already done (checkpoint found)")
            else:
                try:
                    interaction_results = phase_interactions(
                        dataset, config, explanations, explain_indices, device,
                    )
                    ds_results["interactions"] = interaction_results
                    _mark_phase_done(config.output_dir, ds_name, "interactions", inter_hash)
                except Exception as e:
                    logger.error(f"Interaction analysis failed for {ds_name}: {e}", exc_info=True)
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Phase: Alignment (optional)
        alignment_results = None
        if "alignment" in active_phases and explanations and config.alignment.enabled:
            align_hash = config_section_hash({"alignment": vars(config.alignment), "keys": sorted(explanations.keys())})
            if _is_phase_done(config.output_dir, ds_name, "alignment", align_hash):
                logger.info("  Alignment phase: already done (checkpoint found)")
            else:
                try:
                    alignment_results = phase_alignment(dataset, config, explanations, y_explain)
                    ds_results["alignment"] = alignment_results
                    _mark_phase_done(config.output_dir, ds_name, "alignment", align_hash)
                except Exception as e:
                    logger.error(f"Alignment failed for {ds_name}: {e}", exc_info=True)

        # Phase: Visualize
        if "visualize" in active_phases:
            try:
                phase_visualize(
                    dataset, config,
                    consensus_results, interaction_results, alignment_results,
                    explanations, explain_indices,
                )
                _mark_phase_done(config.output_dir, ds_name, "visualize", "always-rerun")
            except Exception as e:
                logger.error(f"Visualization failed for {ds_name}: {e}", exc_info=True)

        ds_elapsed = time.time() - ds_start
        logger.info(f"Dataset {ds_name} completed in {ds_elapsed:.1f}s")
        all_results[ds_name] = ds_results

    _save_summary(all_results, config.output_dir)
    _print_final_summary(all_results)


def _load_all_explanations(
    ds_output_dir: Path,
    xai_modes: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load all explanations from disk across all modes."""
    explanations = {}
    indices = None

    for mode in xai_modes:
        mode_label = "protocol-aware" if mode == "pa" else "normal"
        explain_dir = ds_output_dir / "explanations" / mode_label

        if not explain_dir.exists():
            continue

        for path in sorted(explain_dir.glob("*_attributions.npy")):
            key = path.stem.replace("_attributions", "")
            explanations[key] = np.load(path)

        idx_path = explain_dir / "explain_indices.npy"
        if idx_path.exists() and indices is None:
            indices = np.load(idx_path)

    # Also check dataset-level indices/labels
    if indices is None and (ds_output_dir / "explain_indices.npy").exists():
        indices = np.load(ds_output_dir / "explain_indices.npy")

    if (ds_output_dir / "explain_labels.npy").exists():
        y_explain = np.load(ds_output_dir / "explain_labels.npy")
    else:
        y_explain = None

    if indices is None:
        raise FileNotFoundError(f"No explain_indices.npy found in {ds_output_dir}")

    return explanations, indices, y_explain
```

- [ ] **Step 4: Update the CLI and main() to use YAML config**

Replace `parse_args` and `main` at the bottom of `main.py`:

```python
def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Feature Interaction, Consensus & Expert Alignment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: experiments/config.yaml)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=["nsl-kdd", "cic-ids-2017", "unsw-nb15", "cse-cic-ids2018"],
        help="Datasets to process (default: from YAML config)",
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        default=["all"],
        choices=["all", "train", "explain", "consensus", "interactions", "alignment", "visualize"],
        help="Experiment phase(s) to run",
    )
    parser.add_argument(
        "--xai-modes",
        nargs="+",
        default=None,
        choices=["normal", "pa"],
        help="XAI modes to include (default: from YAML config)",
    )
    parser.add_argument(
        "--num-explain-samples",
        type=int,
        default=None,
        help="Override number of test samples for explanations",
    )
    parser.add_argument(
        "--interaction-samples",
        type=int,
        default=None,
        help="Override number of samples for SHAP interaction values",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--no-alignment",
        action="store_true",
        help="Disable expert alignment phase",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoints (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoints and rerun everything",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config from YAML
    config = load_experiment3_config(args.config)

    # Apply CLI overrides
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.no_smote:
        config.data.apply_smote = False
    if args.seed is not None:
        config.seed = args.seed
    if args.num_explain_samples is not None:
        config.consensus.num_explain_samples = args.num_explain_samples
        config.explainer.num_explain_samples = args.num_explain_samples
    if args.interaction_samples is not None:
        config.interaction.shap_interaction_samples = args.interaction_samples
    if args.xai_modes:
        config.xai_modes = args.xai_modes
    if args.no_alignment:
        config.alignment.enabled = False

    # Clear checkpoints if --no-resume
    if args.no_resume:
        import glob as glob_mod
        for marker in glob_mod.glob(str(config.output_dir / "*" / ".phase_*.done")):
            os.remove(marker)
        logger.info("Cleared all phase checkpoints")

    datasets = args.datasets or config.ALL_DATASETS
    phases = args.phase

    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger.info("Experiment 3: Feature Interaction, Consensus & Expert Alignment")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Phases: {phases}")
    logger.info(f"XAI modes: {config.xai_modes}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Explain samples: {config.consensus.num_explain_samples}")
    logger.info(f"Interaction samples: {config.interaction.shap_interaction_samples}")
    logger.info(f"Alignment: {'enabled' if config.alignment.enabled else 'disabled'}")

    run_experiment(config, datasets, phases)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Commit**

```bash
git add experiments/3/main.py
git commit -m "feat(exp3): unified explanation pool with exp1 reuse, PA mode, and checkpointing"
```

---

### Task 7: Parallelize Consensus with Per-Pair Checkpointing

**Files:**
- Modify: `experiments/3/consensus.py`

- [ ] **Step 1: Add parallel consensus with checkpointing**

Replace the entire `experiments/3/consensus.py` with:

```python
"""Explainer consensus analysis: pairwise agreement metrics between XAI methods.

Metrics:
- Spearman's rank correlation coefficient (monotonic relationship)
- Kendall's tau-b (concordance of ranked pairs)
- Top-k feature intersection (overlap of most important features)
- Wilcoxon signed-rank test (statistical significance of divergence)
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

from config import ConsensusConfig

logger = logging.getLogger(__name__)


@dataclass
class PairwiseConsensusResult:
    explainer_a: str
    explainer_b: str
    spearman_mean: float
    spearman_std: float
    kendall_mean: float
    kendall_std: float
    top_k_intersection: dict[int, float]  # {k: mean_overlap}
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    wilcoxon_reject_h0: bool


def _pair_checkpoint_path(checkpoint_dir: Path, key_a: str, key_b: str) -> Path:
    """Return the checkpoint file path for a consensus pair."""
    return checkpoint_dir / f"{key_a}__vs__{key_b}.json"


def _load_pair_checkpoint(checkpoint_dir: Path, key_a: str, key_b: str) -> PairwiseConsensusResult | None:
    """Load a cached pair result if it exists."""
    path = _pair_checkpoint_path(checkpoint_dir, key_a, key_b)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return PairwiseConsensusResult(
            explainer_a=data["explainer_a"],
            explainer_b=data["explainer_b"],
            spearman_mean=data["spearman_mean"],
            spearman_std=data["spearman_std"],
            kendall_mean=data["kendall_mean"],
            kendall_std=data["kendall_std"],
            top_k_intersection={int(k): v for k, v in data["top_k_intersection"].items()},
            wilcoxon_statistic=data["wilcoxon_statistic"],
            wilcoxon_p_value=data["wilcoxon_p_value"],
            wilcoxon_reject_h0=data["wilcoxon_reject_h0"],
        )
    except (json.JSONDecodeError, KeyError):
        return None


def _save_pair_checkpoint(checkpoint_dir: Path, result: PairwiseConsensusResult) -> None:
    """Save a single pair result to disk."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _pair_checkpoint_path(checkpoint_dir, result.explainer_a, result.explainer_b)
    path.write_text(json.dumps({
        "explainer_a": result.explainer_a,
        "explainer_b": result.explainer_b,
        "spearman_mean": result.spearman_mean,
        "spearman_std": result.spearman_std,
        "kendall_mean": result.kendall_mean,
        "kendall_std": result.kendall_std,
        "top_k_intersection": {str(k): v for k, v in result.top_k_intersection.items()},
        "wilcoxon_statistic": result.wilcoxon_statistic,
        "wilcoxon_p_value": result.wilcoxon_p_value,
        "wilcoxon_reject_h0": result.wilcoxon_reject_h0,
    }, indent=2))


def _compute_pairwise_rank_correlations(
    attrs_a: np.ndarray,
    attrs_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample Spearman and Kendall correlations between two attribution arrays."""
    n_samples = len(attrs_a)
    spearman_rhos = np.zeros(n_samples)
    kendall_taus = np.zeros(n_samples)

    abs_a = np.abs(attrs_a)
    abs_b = np.abs(attrs_b)

    for i in range(n_samples):
        if np.std(abs_a[i]) < 1e-12 or np.std(abs_b[i]) < 1e-12:
            spearman_rhos[i] = 0.0
            kendall_taus[i] = 0.0
            continue

        rho, _ = stats.spearmanr(abs_a[i], abs_b[i])
        tau, _ = stats.kendalltau(abs_a[i], abs_b[i])
        spearman_rhos[i] = rho if np.isfinite(rho) else 0.0
        kendall_taus[i] = tau if np.isfinite(tau) else 0.0

    return spearman_rhos, kendall_taus


def _compute_top_k_intersection(
    attrs_a: np.ndarray,
    attrs_b: np.ndarray,
    k_values: list[int],
) -> dict[int, float]:
    """Compute mean top-k feature overlap between two explainers."""
    abs_a = np.abs(attrs_a)
    abs_b = np.abs(attrs_b)
    n_features = attrs_a.shape[1]

    results = {}
    for k in k_values:
        k_actual = min(k, n_features)
        overlaps = np.zeros(len(attrs_a))

        for i in range(len(attrs_a)):
            top_a = set(np.argsort(abs_a[i])[::-1][:k_actual])
            top_b = set(np.argsort(abs_b[i])[::-1][:k_actual])
            overlaps[i] = len(top_a & top_b) / k_actual

        results[k] = float(np.mean(overlaps))

    return results


def _compute_wilcoxon_test(
    attrs_a: np.ndarray,
    attrs_b: np.ndarray,
    alpha: float,
) -> tuple[float, float, bool]:
    """Wilcoxon signed-rank test on mean absolute attributions per feature."""
    mean_abs_a = np.mean(np.abs(attrs_a), axis=0)
    mean_abs_b = np.mean(np.abs(attrs_b), axis=0)

    diff = mean_abs_a - mean_abs_b
    if np.all(np.abs(diff) < 1e-12):
        return 0.0, 1.0, False

    try:
        stat, p_value = stats.wilcoxon(mean_abs_a, mean_abs_b, alternative="two-sided")
    except ValueError:
        return 0.0, 1.0, False

    return float(stat), float(p_value), p_value < alpha


def _compute_single_pair(
    key_a: str,
    key_b: str,
    attrs_a: np.ndarray,
    attrs_b: np.ndarray,
    top_k_values: list[int],
    alpha: float,
) -> PairwiseConsensusResult:
    """Compute all consensus metrics for a single explainer pair.

    This is the unit of work for parallel execution.
    """
    spearman_rhos, kendall_taus = _compute_pairwise_rank_correlations(attrs_a, attrs_b)
    top_k = _compute_top_k_intersection(attrs_a, attrs_b, top_k_values)
    w_stat, w_pval, w_reject = _compute_wilcoxon_test(attrs_a, attrs_b, alpha)

    return PairwiseConsensusResult(
        explainer_a=key_a,
        explainer_b=key_b,
        spearman_mean=float(np.mean(spearman_rhos)),
        spearman_std=float(np.std(spearman_rhos)),
        kendall_mean=float(np.mean(kendall_taus)),
        kendall_std=float(np.std(kendall_taus)),
        top_k_intersection=top_k,
        wilcoxon_statistic=w_stat,
        wilcoxon_p_value=w_pval,
        wilcoxon_reject_h0=w_reject,
    )


def tag_pair(key_a: str, key_b: str) -> str:
    """Classify a consensus pair as within-mode or cross-mode."""
    a_is_pa = "PA-" in key_a
    b_is_pa = "PA-" in key_b
    if a_is_pa == b_is_pa:
        return "within-pa" if a_is_pa else "within-normal"
    return "cross-mode"


def compute_pairwise_consensus(
    explanations: dict[str, np.ndarray],
    config: ConsensusConfig,
    max_workers: int = 1,
    checkpoint_dir: Path | None = None,
) -> list[PairwiseConsensusResult]:
    """Compute all pairwise consensus metrics between explainers.

    Args:
        explanations: Dict mapping explainer key to attributions (n_samples, n_features).
        config: Consensus configuration.
        max_workers: Number of parallel workers for pair computation.
        checkpoint_dir: Directory for per-pair checkpoints (resume support).

    Returns:
        List of PairwiseConsensusResult for every explainer pair.
    """
    keys = sorted(explanations.keys())
    all_pairs = list(combinations(keys, 2))
    results = []

    # Load cached pairs
    uncached_pairs = []
    if checkpoint_dir:
        for key_a, key_b in all_pairs:
            cached = _load_pair_checkpoint(checkpoint_dir, key_a, key_b)
            if cached is not None:
                results.append(cached)
                logger.info(f"  Loaded cached: {key_a} vs {key_b}")
            else:
                uncached_pairs.append((key_a, key_b))
    else:
        uncached_pairs = all_pairs

    if not uncached_pairs:
        logger.info(f"  All {len(results)} pairs loaded from cache")
        return results

    logger.info(f"  Computing {len(uncached_pairs)} pairs ({len(results)} cached), workers={max_workers}")

    if max_workers <= 1:
        # Sequential
        for key_a, key_b in uncached_pairs:
            logger.info(f"  Consensus: {key_a} vs {key_b} [{tag_pair(key_a, key_b)}]")
            result = _compute_single_pair(
                key_a, key_b, explanations[key_a], explanations[key_b],
                config.top_k_values, config.alpha,
            )
            results.append(result)
            if checkpoint_dir:
                _save_pair_checkpoint(checkpoint_dir, result)
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for key_a, key_b in uncached_pairs:
                future = pool.submit(
                    _compute_single_pair,
                    key_a, key_b,
                    explanations[key_a], explanations[key_b],
                    config.top_k_values, config.alpha,
                )
                futures[future] = (key_a, key_b)

            for future in as_completed(futures):
                key_a, key_b = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if checkpoint_dir:
                        _save_pair_checkpoint(checkpoint_dir, result)
                    logger.info(
                        f"  {key_a} vs {key_b} [{tag_pair(key_a, key_b)}]: "
                        f"Spearman={result.spearman_mean:.3f}, Kendall={result.kendall_mean:.3f}"
                    )
                except Exception as e:
                    logger.error(f"  {key_a} vs {key_b} failed: {e}", exc_info=True)

    return results


def compute_per_attack_consensus(
    explanations: dict[str, np.ndarray],
    y_labels: np.ndarray,
    label_names: list[str],
    config: ConsensusConfig,
    max_workers: int = 1,
    checkpoint_dir: Path | None = None,
) -> dict[str, list[PairwiseConsensusResult]]:
    """Compute consensus metrics broken down by attack type."""
    benign_labels = {"BENIGN", "benign", "normal", "Normal"}
    unique_labels = np.unique(y_labels)

    per_attack = {}
    for label_idx in unique_labels:
        label_name = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)
        if label_name in benign_labels:
            continue

        mask = y_labels == label_idx
        n_attack = np.sum(mask)
        if n_attack < 10:
            logger.warning(f"  Skipping {label_name}: only {n_attack} samples")
            continue

        logger.info(f"  Attack type: {label_name} ({n_attack} samples)")
        attack_explanations = {k: v[mask] for k, v in explanations.items()}

        attack_ckpt = checkpoint_dir / label_name if checkpoint_dir else None
        per_attack[label_name] = compute_pairwise_consensus(
            attack_explanations, config, max_workers, attack_ckpt,
        )

    return per_attack


def consensus_to_dict(results: list[PairwiseConsensusResult]) -> list[dict]:
    """Convert consensus results to JSON-serializable dicts."""
    return [
        {
            "explainer_a": r.explainer_a,
            "explainer_b": r.explainer_b,
            "pair_type": tag_pair(r.explainer_a, r.explainer_b),
            "spearman_mean": r.spearman_mean,
            "spearman_std": r.spearman_std,
            "kendall_mean": r.kendall_mean,
            "kendall_std": r.kendall_std,
            "top_k_intersection": {str(k): v for k, v in r.top_k_intersection.items()},
            "wilcoxon_statistic": r.wilcoxon_statistic,
            "wilcoxon_p_value": r.wilcoxon_p_value,
            "wilcoxon_reject_h0": r.wilcoxon_reject_h0,
        }
        for r in results
    ]
```

- [ ] **Step 2: Update phase_consensus in main.py to pass parallelism config**

In `experiments/3/main.py`, replace the `phase_consensus` function:

```python
def phase_consensus(
    dataset: DatasetBundle,
    config: Experiment3Config,
    explanations: dict[str, np.ndarray],
    y_explain: np.ndarray,
) -> dict:
    """Run pairwise consensus analysis between all explainers."""
    logger.info(f"=== CONSENSUS ANALYSIS on {dataset.dataset_name} ===")
    output_dir = config.output_dir / dataset.dataset_name
    consensus_ckpt_dir = output_dir / "consensus_checkpoints"

    # Overall consensus
    logger.info("  Computing overall pairwise consensus...")
    overall_results = compute_pairwise_consensus(
        explanations, config.consensus,
        max_workers=config.parallelism.max_consensus_workers,
        checkpoint_dir=consensus_ckpt_dir / "overall",
    )

    for r in overall_results:
        logger.info(
            f"    {r.explainer_a} vs {r.explainer_b} [{tag_pair(r.explainer_a, r.explainer_b)}]: "
            f"Spearman={r.spearman_mean:.3f}+-{r.spearman_std:.3f}, "
            f"Kendall={r.kendall_mean:.3f}+-{r.kendall_std:.3f}, "
            f"Top-5={r.top_k_intersection.get(5, 0):.3f}, "
            f"Top-10={r.top_k_intersection.get(10, 0):.3f}, "
            f"Wilcoxon p={r.wilcoxon_p_value:.2e} ({'REJECT' if r.wilcoxon_reject_h0 else 'ACCEPT'} H0)"
        )

    # Per-attack-type consensus
    logger.info("  Computing per-attack-type consensus...")
    label_names = list(dataset.label_encoder.classes_)
    per_attack = compute_per_attack_consensus(
        explanations, y_explain, label_names, config.consensus,
        max_workers=config.parallelism.max_consensus_workers,
        checkpoint_dir=consensus_ckpt_dir / "per_attack",
    )

    # Save results
    results = {
        "overall": consensus_to_dict(overall_results),
        "per_attack": {
            attack: consensus_to_dict(results_list)
            for attack, results_list in per_attack.items()
        },
    }
    with open(output_dir / "consensus_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_serialize)

    return {"overall": overall_results, "per_attack": per_attack}
```

Also add the `tag_pair` import at the top of main.py:

```python
from consensus import (
    compute_pairwise_consensus,
    compute_per_attack_consensus,
    consensus_to_dict,
    tag_pair,
)
```

- [ ] **Step 3: Commit**

```bash
git add experiments/3/consensus.py experiments/3/main.py
git commit -m "perf(exp3): parallel consensus with per-pair checkpointing and cross-mode tagging"
```

---

### Task 8: Batch DNN Interaction Computation

**Files:**
- Modify: `experiments/3/interactions.py`

- [ ] **Step 1: Replace the sequential DNN interaction loop with batched computation**

Replace `compute_shap_interaction_values_dnn` in `experiments/3/interactions.py`:

```python
def compute_shap_interaction_values_dnn(
    dnn_model: torch.nn.Module,
    X_samples: np.ndarray,
    X_background: np.ndarray,
    device: torch.device,
    config: InteractionConfig,
) -> np.ndarray:
    """Approximate SHAP interaction values for DNN using batched perturbations.

    Instead of looping over top features one-by-one (20 separate SHAP calls),
    batch all perturbations into a single tensor and run one SHAP call.

    Args:
        dnn_model: Trained DNN model.
        X_samples: Input samples, shape (n_samples, n_features).
        X_background: Background reference samples.
        device: Torch device.
        config: Interaction configuration.

    Returns:
        Approximate interaction matrix, shape (n_samples, n_features, n_features).
    """
    import shap

    n = min(config.shap_interaction_samples, len(X_samples))
    n_features = X_samples.shape[1]
    X = X_samples[:n]

    logger.info(f"  Approximating DNN interaction values (batched) on {n} samples...")
    start = time.time()

    base_model = dnn_model.module if isinstance(dnn_model, torch.nn.DataParallel) else dnn_model
    base_model.eval()

    bg = X_background[:config.shap_background_samples]
    bg_tensor = torch.tensor(bg, dtype=torch.float32).to(device)
    explainer = shap.DeepExplainer(base_model, bg_tensor)

    # Get base SHAP values
    x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    base_shap = explainer.shap_values(x_tensor, check_additivity=False)

    with torch.no_grad():
        preds = torch.argmax(base_model(x_tensor), dim=1).cpu().numpy()

    # Extract base attributions for predicted class
    base_attrs = _extract_predicted_class_attrs(base_shap, preds, n, n_features)

    # Identify top features by mean importance
    mean_importance = np.mean(np.abs(base_attrs), axis=0)
    top_features = np.argsort(mean_importance)[::-1][:config.top_n_interactions]
    n_top = len(top_features)

    feature_means = np.mean(X, axis=0)

    # Batch all perturbations: create (n_top * n, n_features) tensor
    # Each block of n rows has feature j replaced with its mean
    X_batch = np.tile(X, (n_top, 1))  # (n_top * n, n_features)
    for i, j in enumerate(top_features):
        X_batch[i * n : (i + 1) * n, j] = feature_means[j]

    logger.info(f"  Batched perturbation tensor: {X_batch.shape} ({n_top} features x {n} samples)")

    # Single batched SHAP call
    x_batch_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
    pert_shap_all = explainer.shap_values(x_batch_tensor, check_additivity=False)

    # Tile predictions to match batch layout
    preds_tiled = np.tile(preds, n_top)
    pert_attrs_flat = _extract_predicted_class_attrs(pert_shap_all, preds_tiled, n_top * n, n_features)

    # Reshape to (n_top, n, n_features)
    pert_attrs = pert_attrs_flat.reshape(n_top, n, n_features)

    # Compute interaction matrix
    interaction_matrix = np.zeros((n, n_features, n_features), dtype=np.float32)
    for idx, j in enumerate(top_features):
        # Interaction effect: change in attribution for all features when j is perturbed
        interaction_matrix[:, :, j] = base_attrs - pert_attrs[idx]

    elapsed = time.time() - start
    logger.info(f"  DNN interaction approximation completed in {elapsed:.1f}s (batched)")

    return interaction_matrix


def _extract_predicted_class_attrs(
    shap_values,
    preds: np.ndarray,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    """Extract attributions for the predicted class from SHAP output."""
    if isinstance(shap_values, list):
        stacked = np.stack(shap_values, axis=0)  # (classes, n, features)
        attrs = np.zeros((n_samples, n_features), dtype=np.float32)
        for i, pred in enumerate(preds):
            attrs[i] = stacked[pred, i]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        attrs = np.zeros((n_samples, n_features), dtype=np.float32)
        for i, pred in enumerate(preds):
            attrs[i] = shap_values[i, :, pred]
    else:
        attrs = np.asarray(shap_values, dtype=np.float32)
    return attrs
```

- [ ] **Step 2: Commit**

```bash
git add experiments/3/interactions.py
git commit -m "perf(exp3): batch DNN SHAP interaction perturbations into single call"
```

---

### Task 9: Parallel Visualization

**Files:**
- Modify: `experiments/3/visualizations.py`

- [ ] **Step 1: Add ThreadPoolExecutor to generate_all_plots**

Replace `generate_all_plots` in `experiments/3/visualizations.py`:

```python
def generate_all_plots(
    consensus_results: list[PairwiseConsensusResult] | None,
    per_attack_consensus: dict | None,
    interaction_matrices: dict | None,
    top_interactions: dict | None,
    alignment_results: list[dict] | None,
    feature_names: list[str],
    dataset_name: str,
    plot_dir: Path,
    config,
    shap_values_for_dependence: np.ndarray | None = None,
    X_data_for_dependence: np.ndarray | None = None,
    max_workers: int = 8,
) -> None:
    """Generate all visualization plots for a dataset using parallel workers."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    plot_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, callable]] = []

    if consensus_results:
        tasks.append(("spearman_heatmap", lambda: plot_consensus_heatmap(
            consensus_results, "spearman_mean",
            f"Spearman Rank Correlation — {dataset_name}",
            plot_dir / f"{dataset_name}_spearman_heatmap.png",
        )))
        tasks.append(("kendall_heatmap", lambda: plot_consensus_heatmap(
            consensus_results, "kendall_mean",
            f"Kendall's Tau — {dataset_name}",
            plot_dir / f"{dataset_name}_kendall_heatmap.png",
        )))
        tasks.append(("wilcoxon_pvalues", lambda: plot_wilcoxon_pvalue_matrix(
            consensus_results, config.consensus.alpha,
            plot_dir / f"{dataset_name}_wilcoxon_pvalues.png",
        )))
        tasks.append(("top_k_intersection", lambda: plot_top_k_intersection(
            consensus_results, config.consensus.top_k_values,
            plot_dir / f"{dataset_name}_top_k_intersection",
        )))

    if interaction_matrices:
        for model_name, matrix in interaction_matrices.items():
            # Capture loop variables
            _mn, _mx = model_name, matrix
            tasks.append((f"interactions_{_mn}", lambda mn=_mn, mx=_mx: plot_interaction_heatmap(
                mx, feature_names,
                top_n=min(15, mx.shape[0]),
                title=f"Feature Interaction Strength ({mn}) — {dataset_name}",
                output_path=plot_dir / f"{dataset_name}_{mn}_interactions.png",
            )))

    if top_interactions and shap_values_for_dependence is not None and X_data_for_dependence is not None:
        for model_name, pairs in top_interactions.items():
            for pair_idx, pair in enumerate(pairs[:5]):
                _mn, _pi, _p = model_name, pair_idx, pair
                tasks.append((f"dependence_{_mn}_{_pi}", lambda mn=_mn, pi=_pi, p=_p: plot_shap_dependence(
                    shap_values_for_dependence,
                    X_data_for_dependence,
                    p["feature_a_idx"],
                    p["feature_b_idx"],
                    feature_names,
                    plot_dir / f"{dataset_name}_{mn}_dependence_{pi}.png",
                )))

    if alignment_results:
        tasks.append(("rra_scores", lambda: plot_alignment_scores(
            alignment_results, "rra_score",
            f"Relevance Rank Accuracy (RRA) — {dataset_name}",
            plot_dir / f"{dataset_name}_rra_scores.png",
        )))
        tasks.append(("rma_scores", lambda: plot_alignment_scores(
            alignment_results, "rma_score",
            f"Relevance Mass Accuracy (RMA) — {dataset_name}",
            plot_dir / f"{dataset_name}_rma_scores.png",
        )))

    if not tasks:
        logger.info("  No plots to generate")
        return

    logger.info(f"  Generating {len(tasks)} plots with {min(max_workers, len(tasks))} workers")

    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"  Plot '{name}' failed: {e}", exc_info=True)
```

- [ ] **Step 2: Update phase_visualize in main.py to pass max_workers**

In `experiments/3/main.py`, update the `generate_all_plots` call inside `phase_visualize`:

```python
    generate_all_plots(
        consensus_results=consensus_results.get("overall") if consensus_results else None,
        per_attack_consensus=consensus_results.get("per_attack") if consensus_results else None,
        interaction_matrices=interaction_results.get("matrices") if interaction_results else None,
        top_interactions=interaction_results.get("top_interactions") if interaction_results else None,
        alignment_results=alignment_results,
        feature_names=dataset.feature_names,
        dataset_name=dataset.dataset_name,
        plot_dir=plot_dir,
        config=config,
        shap_values_for_dependence=shap_values,
        X_data_for_dependence=X_data,
        max_workers=config.parallelism.max_plot_workers,
    )
```

- [ ] **Step 3: Commit**

```bash
git add experiments/3/visualizations.py experiments/3/main.py
git commit -m "perf(exp3): parallel plot generation with ThreadPoolExecutor"
```

---

### Task 10: Update Summary Printer for Cross-Mode Results

**Files:**
- Modify: `experiments/3/main.py`

- [ ] **Step 1: Update _print_final_summary to show cross-mode breakdown**

Replace `_print_final_summary`:

```python
def _print_final_summary(all_results: dict) -> None:
    """Print human-readable summary with within/cross-mode breakdown."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 3 SUMMARY: Feature Interaction, Consensus & Expert Alignment")
    logger.info("=" * 80)

    for ds_name, ds_results in all_results.items():
        logger.info(f"\n--- {ds_name} ---")

        if "consensus" in ds_results and ds_results["consensus"]:
            overall = ds_results["consensus"].get("overall", [])
            if overall:
                # Break down by pair type
                for pair_type in ["within-normal", "within-pa", "cross-mode"]:
                    typed = [r for r in overall if tag_pair(r.explainer_a, r.explainer_b) == pair_type]
                    if not typed:
                        continue
                    mean_sp = np.mean([r.spearman_mean for r in typed])
                    mean_kt = np.mean([r.kendall_mean for r in typed])
                    n_sig = sum(1 for r in typed if r.wilcoxon_reject_h0)
                    logger.info(
                        f"  Consensus [{pair_type}]: "
                        f"Mean Spearman={mean_sp:.3f}, Mean Kendall={mean_kt:.3f}, "
                        f"{n_sig}/{len(typed)} pairs show significant divergence"
                    )

        if "alignment" in ds_results and ds_results["alignment"]:
            alignment = ds_results["alignment"]
            mean_rra = np.mean([r["rra_score"] for r in alignment])
            mean_rma = np.mean([r["rma_score"] for r in alignment])
            logger.info(f"  Alignment: Mean RRA={mean_rra:.3f}, Mean RMA={mean_rma:.3f}")
```

- [ ] **Step 2: Commit**

```bash
git add experiments/3/main.py
git commit -m "feat(exp3): cross-mode consensus breakdown in summary output"
```

---

### Task 11: Smoke Test the Full Pipeline

- [ ] **Step 1: Verify the config loads end-to-end**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids
uv run python -c "
import sys; sys.path.insert(0, 'experiments/3'); sys.path.insert(0, 'experiments')
from config import load_experiment3_config
c = load_experiment3_config()
print('Config loaded OK')
print(f'  seed={c.seed}, modes={c.xai_modes}')
print(f'  alignment={c.alignment.enabled}, workers={c.parallelism.max_consensus_workers}')
print(f'  datasets={c.ALL_DATASETS}')
print(f'  exp1_dir={c.exp1_output_dir}')
"
```

Expected:
```
Config loaded OK
  seed=42, modes=['normal', 'pa']
  alignment=True, workers=16
  datasets=['nsl-kdd', 'cic-ids-2017', 'unsw-nb15', 'cse-cic-ids2018']
  exp1_dir=experiments/1/results
```

- [ ] **Step 2: Verify CLI help works**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids
uv run python experiments/3/main.py --help
```

Expected: Shows all arguments including `--config`, `--xai-modes`, `--no-alignment`, `--no-resume`.

- [ ] **Step 3: Verify checkpoint system**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids
uv run python -c "
import sys, os; sys.path.insert(0, 'experiments/3'); sys.path.insert(0, 'experiments')
from main import _mark_phase_done, _is_phase_done
from pathlib import Path
import tempfile

with tempfile.TemporaryDirectory() as td:
    p = Path(td)
    assert not _is_phase_done(p, 'test', 'train', 'abc123')
    _mark_phase_done(p, 'test', 'train', 'abc123')
    assert _is_phase_done(p, 'test', 'train', 'abc123')
    assert not _is_phase_done(p, 'test', 'train', 'different_hash')
    print('Checkpoint system OK')
"
```

Expected: `Checkpoint system OK`

- [ ] **Step 4: Commit any fixes**

If any fixes were needed, commit them:

```bash
git add -A
git commit -m "fix(exp3): address smoke test issues"
```

---
