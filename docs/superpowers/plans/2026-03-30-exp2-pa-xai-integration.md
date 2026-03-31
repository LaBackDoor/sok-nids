# Experiment 2: pa_xai Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate `pa_xai` into Experiment 2 for side-by-side vanilla vs protocol-aware explainer robustness comparison, replacing the standalone constraint system with `pa_xai`'s `ConstraintEnforcer`.

**Architecture:** Two new wrapper modules (`pa_constraints.py`, `pa_explainers.py`) adapt `pa_xai` classes to Experiment 2's interfaces. Config gains PA method lists and DeepLIFT params. `main.py` wires everything together. `feature_constraints.py` is deleted.

**Tech Stack:** `pa_xai` (ProtocolAwareSHAP/LIME/IG/DeepLIFT, ConstraintEnforcer, DatasetSchema), PyTorch, NumPy, scikit-learn (MinMaxScaler)

**Spec:** `docs/superpowers/specs/2026-03-30-exp2-pa-xai-integration-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `experiments/2/pa_constraints.py` | Adapts `pa_xai.ConstraintEnforcer` to attack projector interface |
| Create | `experiments/2/pa_explainers.py` | Wraps `pa_xai` explainers into `fn(X) -> np.ndarray` closures |
| Modify | `experiments/2/config.py` | Add DeepLIFT + PA method config fields |
| Modify | `experiments/2/main.py` | Wire PA constraints, PA explainers, new CLI flags |
| Delete | `experiments/2/feature_constraints.py` | Replaced by `pa_constraints.py` |

---

### Task 1: Update config.py with DeepLIFT and PA method fields

**Files:**
- Modify: `experiments/2/config.py:46-60`

- [ ] **Step 1: Add DeepLIFT and PA fields to RobustnessConfig**

In `experiments/2/config.py`, update the `RobustnessConfig` dataclass:

```python
@dataclass
class RobustnessConfig:
    """Parameters for mathematical robustness metric evaluation."""

    num_samples: int = 1000
    explanation_methods: list[str] = field(
        default_factory=lambda: ["SHAP", "LIME", "IG", "DeepLIFT"]
    )
    pa_explanation_methods: list[str] = field(
        default_factory=lambda: ["PA-SHAP", "PA-LIME", "PA-IG", "PA-DeepLIFT"]
    )
    distance_norm: str = "l2"  # l2 or linf for Lipschitz computation
    explanation_similarity_epsilon: float = 0.1
    shap_background_samples: int = 100
    lime_num_samples: int = 5000
    lime_num_features: int = 10
    ig_n_steps: int = 50
    ig_internal_batch_size: int = 4096
    deeplift_internal_batch_size: int = 4096
```

Changes from original:
- `explanation_methods` default gains `"DeepLIFT"`
- New field `pa_explanation_methods` with the four PA variants
- New field `deeplift_internal_batch_size`

- [ ] **Step 2: Verify config loads correctly**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids && uv run python -c "
import sys, importlib.util
spec = importlib.util.spec_from_file_location('cfg', 'experiments/2/config.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
c = mod.RobustnessConfig()
print('methods:', c.explanation_methods)
print('pa_methods:', c.pa_explanation_methods)
print('deeplift_batch:', c.deeplift_internal_batch_size)
"
```

Expected output:
```
methods: ['SHAP', 'LIME', 'IG', 'DeepLIFT']
pa_methods: ['PA-SHAP', 'PA-LIME', 'PA-IG', 'PA-DeepLIFT']
deeplift_batch: 4096
```

- [ ] **Step 3: Commit**

```bash
git add experiments/2/config.py
git commit -m "feat(exp2): add DeepLIFT and PA method fields to RobustnessConfig"
```

---

### Task 2: Create pa_constraints.py

**Files:**
- Create: `experiments/2/pa_constraints.py`

- [ ] **Step 1: Write pa_constraints.py**

Create `experiments/2/pa_constraints.py`:

```python
"""Protocol-aware constraint projector for adversarial attacks.

Adapts pa_xai's ConstraintEnforcer + DatasetSchema into the
constraint_projector(X_adv, X_original, epsilon) -> X_adv interface
that attacks.py expects. Replaces feature_constraints.py.
"""

import logging

import numpy as np
import torch

from pa_xai import get_schema
from pa_xai.core.constraints import ConstraintEnforcer

logger = logging.getLogger(__name__)

DATASET_SCHEMA_MAP = {
    "nsl-kdd": "NSL-KDD",
    "cic-ids-2017": "CIC-IDS-2017",
    "unsw-nb15": "UNSW-NB15-CICFlowMeter",
    "cse-cic-ids2018": "CSE-CIC-IDS2018",
}


def make_pa_constraint_projector(dataset_name, scaler, device):
    """Create a constraint projection closure for use in FGSM/PGD attacks.

    Args:
        dataset_name: Dataset key (e.g., "nsl-kdd").
        scaler: Fitted sklearn MinMaxScaler used during preprocessing.
        device: Torch device for the returned tensor.

    Returns:
        callable(X_adv: Tensor, X_original: Tensor, epsilon: float) -> Tensor
        or None if no schema exists for the dataset.
    """
    if dataset_name not in DATASET_SCHEMA_MAP:
        logger.warning(
            f"No pa_xai schema for '{dataset_name}'. "
            "Skipping constrained attacks for this dataset."
        )
        return None

    schema = get_schema(DATASET_SCHEMA_MAP[dataset_name])
    enforcer = ConstraintEnforcer(schema)

    # Precompute scaler parameters as float64 for precision
    data_min = scaler.data_min_.astype(np.float64)
    data_range = (scaler.data_max_ - scaler.data_min_).astype(np.float64)
    # Avoid division by zero for constant features
    data_range_safe = data_range.copy()
    data_range_safe[data_range_safe < 1e-12] = 1.0

    protocol_idx = schema.protocol_index
    protocol_enc = schema.protocol_encoding

    def projector(
        X_adv: torch.Tensor,
        X_original: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        # 1. Convert to numpy
        adv_np = X_adv.detach().cpu().numpy().astype(np.float64)
        orig_np = X_original.detach().cpu().numpy().astype(np.float64)

        # 2. Inverse-scale from [0,1] to original feature space
        adv_orig = adv_np * data_range + data_min

        # 3. Group by protocol and enforce constraints per group
        if protocol_idx is not None:
            orig_orig = orig_np * data_range + data_min
            protocol_vals = np.round(orig_orig[:, protocol_idx]).astype(int)
            unique_protocols = np.unique(protocol_vals)

            for pval in unique_protocols:
                mask = protocol_vals == pval
                group = adv_orig[mask]
                enforcer.enforce(
                    group,
                    protocol_value=float(pval),
                    protocol_encoding=protocol_enc,
                )
                adv_orig[mask] = group
        else:
            enforcer.enforce(
                adv_orig,
                protocol_value=None,
                protocol_encoding=protocol_enc,
            )

        # 4. Re-scale back to [0,1]
        adv_scaled = (adv_orig - data_min) / data_range_safe
        adv_scaled = np.clip(adv_scaled, 0.0, 1.0)

        # 5. Re-apply L-inf epsilon-ball projection
        delta = np.clip(adv_scaled - orig_np, -epsilon, epsilon)
        adv_scaled = np.clip(orig_np + delta, 0.0, 1.0)

        # 6. Convert back to torch
        return torch.tensor(
            adv_scaled, dtype=torch.float32, device=device
        )

    return projector


def pa_constraint_spec_to_dict(dataset_name):
    """Serialize the pa_xai schema constraints to a JSON-compatible dict.

    Returns None if no schema exists for the dataset.
    """
    if dataset_name not in DATASET_SCHEMA_MAP:
        return None

    schema = get_schema(DATASET_SCHEMA_MAP[dataset_name])

    return {
        "dataset_name": dataset_name,
        "schema_name": DATASET_SCHEMA_MAP[dataset_name],
        "num_features": len(schema.feature_names),
        "protocol_feature": schema.protocol_feature,
        "protocol_encoding": schema.protocol_encoding,
        "num_non_negative": len(schema.non_negative_indices),
        "num_tcp_only": len(schema.tcp_only_indices),
        "num_discrete": len(schema.discrete_indices),
        "num_hierarchical": len(schema.hierarchical_index_triples),
        "num_bounded_range": len(schema.bounded_range_index_bounds),
        "num_cross_feature": len(schema.cross_feature_index_tuples),
        "num_std_range": len(schema.std_range_index_triples),
    }
```

- [ ] **Step 2: Verify the module imports and projector can be constructed**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids && uv run python -c "
from experiments.two_pa_constraints_test import test_import
" 2>&1 || uv run python -c "
import sys
sys.path.insert(0, 'experiments/2')
from pa_constraints import make_pa_constraint_projector, pa_constraint_spec_to_dict, DATASET_SCHEMA_MAP
print('Imported successfully')
print('Supported datasets:', list(DATASET_SCHEMA_MAP.keys()))
spec = pa_constraint_spec_to_dict('nsl-kdd')
print('NSL-KDD spec:', spec)
# Test unsupported dataset returns None
result = make_pa_constraint_projector('cic-iov-2024', None, 'cpu')
print('Unsupported dataset projector:', result)
"
```

Expected: Imports succeed, spec dict prints, unsupported dataset returns `None`.

- [ ] **Step 3: Commit**

```bash
git add experiments/2/pa_constraints.py
git commit -m "feat(exp2): add pa_constraints.py wrapping pa_xai ConstraintEnforcer"
```

---

### Task 3: Create pa_explainers.py

**Files:**
- Create: `experiments/2/pa_explainers.py`

- [ ] **Step 1: Write pa_explainers.py**

Create `experiments/2/pa_explainers.py`:

```python
"""Protocol-Aware XAI explainers for Experiment 2 robustness evaluation.

Wraps pa_xai explainer classes into the fn(X) -> np.ndarray interface
that robustness.py expects. Each function returns a closure that generates
attributions for a batch of samples.
"""

import copy
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

DATASET_SCHEMA_MAP = {
    "nsl-kdd": "NSL-KDD",
    "cic-ids-2017": "CIC-IDS-2017",
    "unsw-nb15": "UNSW-NB15-CICFlowMeter",
    "cse-cic-ids2018": "CSE-CIC-IDS2018",
}


def _get_schema(dataset_name: str):
    from pa_xai import get_schema
    return get_schema(DATASET_SCHEMA_MAP[dataset_name])


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DataParallel and return model in eval mode."""
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    base.eval()
    return base


def _clone_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Deep-copy a model to avoid Captum hook conflicts."""
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    cloned = copy.deepcopy(base)
    cloned.to(device)
    cloned.eval()
    return cloned


def make_pa_explain_fn(
    method: str,
    dnn_model: torch.nn.Module,
    dnn_wrapper,
    dataset,
    device: torch.device,
    config,
):
    """Create a PA explain function for robustness evaluation.

    Args:
        method: One of "PA-SHAP", "PA-LIME", "PA-IG", "PA-DeepLIFT".
        dnn_model: Trained DNN (NIDSNet) model.
        dnn_wrapper: DNNWrapper with predict_proba method.
        dataset: DatasetBundle with X_train, y_train, dataset_name.
        device: Torch device.
        config: Experiment2Config (uses config.robustness sub-config).

    Returns:
        callable(X: np.ndarray) -> np.ndarray of shape (n_samples, n_features)

    Raises:
        ValueError: If dataset has no pa_xai schema or method is unknown.
    """
    ds_name = dataset.dataset_name
    if ds_name not in DATASET_SCHEMA_MAP:
        raise ValueError(
            f"No pa_xai schema for '{ds_name}'. Cannot create PA explainer."
        )

    rob_cfg = config.robustness

    if method == "PA-SHAP":
        return _make_pa_shap_fn(
            dnn_model, dataset, device, rob_cfg
        )
    elif method == "PA-LIME":
        return _make_pa_lime_fn(
            dnn_wrapper, dataset, rob_cfg
        )
    elif method == "PA-IG":
        return _make_pa_ig_fn(
            dnn_model, dataset, device, rob_cfg
        )
    elif method == "PA-DeepLIFT":
        return _make_pa_deeplift_fn(
            dnn_model, dataset, device, rob_cfg
        )
    else:
        raise ValueError(f"Unknown PA method: {method}")


def _make_pa_shap_fn(dnn_model, dataset, device, rob_cfg):
    from pa_xai import ProtocolAwareSHAP

    schema = _get_schema(dataset.dataset_name)
    base_model = _unwrap_model(dnn_model)

    explainer = ProtocolAwareSHAP(
        schema, base_model, dataset.X_train, dataset.y_train,
        backend="deep", n_background=rob_cfg.shap_background_samples,
    )

    def fn(X: np.ndarray) -> np.ndarray:
        all_attrs = []
        for i in range(len(X)):
            result = explainer.explain_instance(X[i])
            all_attrs.append(result.attributions)
        return np.stack(all_attrs, axis=0)

    return fn


def _make_pa_lime_fn(dnn_wrapper, dataset, rob_cfg):
    from pa_xai import ProtocolAwareLIME

    schema = _get_schema(dataset.dataset_name)
    explainer = ProtocolAwareLIME(schema)

    def fn(X: np.ndarray) -> np.ndarray:
        all_attrs = []
        for i in range(len(X)):
            result = explainer.explain_instance(
                X[i], dnn_wrapper.predict_proba,
                num_samples=rob_cfg.lime_num_samples,
            )
            all_attrs.append(result.attributions)
        return np.stack(all_attrs, axis=0)

    return fn


def _make_pa_ig_fn(dnn_model, dataset, device, rob_cfg):
    from pa_xai import ProtocolAwareIG

    schema = _get_schema(dataset.dataset_name)
    # Clone model to avoid Captum hook conflicts with other methods
    model_clone = _clone_model(dnn_model, device)

    explainer = ProtocolAwareIG(
        schema, model_clone, dataset.X_train, dataset.y_train,
        constrain_path=True,
    )

    def fn(X: np.ndarray) -> np.ndarray:
        all_attrs = []
        for i in range(len(X)):
            result = explainer.explain_instance(
                X[i].astype(np.float32),
                n_steps=rob_cfg.ig_n_steps,
            )
            all_attrs.append(result.attributions)
        return np.stack(all_attrs, axis=0)

    return fn


def _make_pa_deeplift_fn(dnn_model, dataset, device, rob_cfg):
    from pa_xai import ProtocolAwareDeepLIFT

    schema = _get_schema(dataset.dataset_name)
    # Clone model to avoid Captum hook conflicts with other methods
    model_clone = _clone_model(dnn_model, device)

    explainer = ProtocolAwareDeepLIFT(
        schema, model_clone, dataset.X_train, dataset.y_train,
    )

    def fn(X: np.ndarray) -> np.ndarray:
        all_attrs = []
        for i in range(len(X)):
            result = explainer.explain_instance(
                X[i].astype(np.float32),
            )
            all_attrs.append(result.attributions)
        return np.stack(all_attrs, axis=0)

    return fn
```

- [ ] **Step 2: Verify module imports**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids && uv run python -c "
import sys
sys.path.insert(0, 'experiments/2')
from pa_explainers import make_pa_explain_fn, DATASET_SCHEMA_MAP
print('Imported successfully')
print('Supported datasets:', list(DATASET_SCHEMA_MAP.keys()))
"
```

Expected: Imports succeed without errors.

- [ ] **Step 3: Commit**

```bash
git add experiments/2/pa_explainers.py
git commit -m "feat(exp2): add pa_explainers.py wrapping pa_xai explainer classes"
```

---

### Task 4: Update main.py — imports and make_explain_fn

**Files:**
- Modify: `experiments/2/main.py:76-194`

- [ ] **Step 1: Update imports in main.py**

Replace the `feature_constraints` import at line 77:

Old:
```python
from feature_constraints import build_constraints, make_constraint_projector, spec_to_dict  # noqa: E402
```

New:
```python
from pa_constraints import make_pa_constraint_projector, pa_constraint_spec_to_dict  # noqa: E402
```

Add PA explainer import after line 79 (after the scaffolding import):

```python
from pa_explainers import make_pa_explain_fn  # noqa: E402
```

Also add `explain_deeplift` to the experiment 1 explainer imports. Change line 47-52 from:

```python
from explainers import (
    ExplanationResult,
    explain_ig,
    explain_lime,
    explain_shap_dnn,
    explain_shap_rf,
)
```

to:

```python
from explainers import (
    ExplanationResult,
    explain_deeplift,
    explain_ig,
    explain_lime,
    explain_shap_dnn,
    explain_shap_rf,
)
```

- [ ] **Step 2: Expand make_explain_fn to handle DeepLIFT and PA methods**

Replace the entire `make_explain_fn` function (lines 146-194) with:

```python
def make_explain_fn(
    method: str,
    dnn_model: NIDSNet,
    dnn_wrapper: DNNWrapper,
    rf_wrapper: RFWrapper,
    dataset: DatasetBundle,
    device: torch.device,
    config: Experiment2Config,
):
    """Create a callable explain function for robustness evaluation.

    Supports vanilla methods (SHAP, LIME, IG, DeepLIFT) and
    protocol-aware methods (PA-SHAP, PA-LIME, PA-IG, PA-DeepLIFT).
    """
    # PA methods delegate to pa_explainers module
    if method.startswith("PA-"):
        return make_pa_explain_fn(
            method, dnn_model, dnn_wrapper, dataset, device, config,
        )

    # Vanilla methods
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(
        len(dataset.X_train),
        size=config.robustness.shap_background_samples,
        replace=False,
    )
    X_bg = dataset.X_train[bg_idx]

    explainer_cfg = ExplainerConfig(
        shap_background_samples=config.robustness.shap_background_samples,
        lime_num_features=config.robustness.lime_num_features,
        lime_num_samples=config.robustness.lime_num_samples,
        ig_n_steps=config.robustness.ig_n_steps,
        ig_internal_batch_size=config.robustness.ig_internal_batch_size,
    )

    if method == "SHAP":
        def fn(X):
            r = explain_shap_dnn(dnn_model, X, X_bg, device, explainer_cfg)
            return r.attributions
        return fn

    elif method == "LIME":
        def fn(X):
            r = explain_lime(
                dnn_wrapper.predict_proba, X, dataset.X_train,
                dataset.feature_names, dataset.num_classes, "DNN", explainer_cfg,
            )
            return r.attributions
        return fn

    elif method == "IG":
        def fn(X):
            r = explain_ig(dnn_model, X, device, explainer_cfg)
            return r.attributions
        return fn

    elif method == "DeepLIFT":
        def fn(X):
            r = explain_deeplift(dnn_model, X, device, explainer_cfg)
            return r.attributions
        return fn

    else:
        raise ValueError(f"Unknown method: {method}")
```

- [ ] **Step 3: Verify make_explain_fn handles all 8 methods**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids && uv run python -c "
import sys
sys.path.insert(0, 'experiments/1')
sys.path.insert(0, 'experiments/2')
import importlib.util
spec = importlib.util.spec_from_file_location('exp2_main', 'experiments/2/main.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('main.py loaded successfully')
# Check that make_explain_fn exists and has the right signature
import inspect
sig = inspect.signature(mod.make_explain_fn)
print('make_explain_fn params:', list(sig.parameters.keys()))
"
```

Expected: Module loads, function signature shows the expected parameters.

- [ ] **Step 4: Commit**

```bash
git add experiments/2/main.py
git commit -m "feat(exp2): expand make_explain_fn for DeepLIFT and PA methods"
```

---

### Task 5: Update main.py — phase_adversarial to use pa_constraints

**Files:**
- Modify: `experiments/2/main.py:252-296`

- [ ] **Step 1: Update phase_adversarial function**

Replace the constrained attacks block inside `phase_adversarial` (lines 279-294). Change from:

```python
    # Constrained attacks
    if config.attack.run_constrained:
        logger.info("  Running CONSTRAINED attacks...")
        spec = build_constraints(dataset.dataset_name, dataset.feature_names, dataset.scaler)
        projector = make_constraint_projector(spec, dataset.scaler, device)
        adv_results = generate_adversarial_examples(
            model=dnn_model,
            X=dataset.X_test,
            y=dataset.y_test,
            config=config.attack,
            device=device,
            constraint_projector=projector,
        )
        summary["constrained"] = _save_adversarial_results(
            adv_results, config.output_dir, dataset.dataset_name, "constrained",
            constraint_spec_dict=spec_to_dict(spec),
        )
```

to:

```python
    # Constrained attacks (using pa_xai ConstraintEnforcer)
    if config.attack.run_constrained:
        projector = make_pa_constraint_projector(
            dataset.dataset_name, dataset.scaler, device,
        )
        if projector is None:
            logger.warning(
                f"  Skipping CONSTRAINED attacks for {dataset.dataset_name}: "
                "no pa_xai schema available."
            )
        else:
            logger.info("  Running CONSTRAINED attacks (pa_xai constraints)...")
            adv_results = generate_adversarial_examples(
                model=dnn_model,
                X=dataset.X_test,
                y=dataset.y_test,
                config=config.attack,
                device=device,
                constraint_projector=projector,
            )
            summary["constrained"] = _save_adversarial_results(
                adv_results, config.output_dir, dataset.dataset_name, "constrained",
                constraint_spec_dict=pa_constraint_spec_to_dict(dataset.dataset_name),
            )
```

- [ ] **Step 2: Commit**

```bash
git add experiments/2/main.py
git commit -m "feat(exp2): use pa_xai ConstraintEnforcer for constrained attacks"
```

---

### Task 6: Update main.py — phase_robustness to evaluate PA methods

**Files:**
- Modify: `experiments/2/main.py:312-405`

- [ ] **Step 1: Update phase_robustness to iterate over vanilla + PA methods**

In the `phase_robustness` function, find the loop at line 362:

```python
        for method in config.robustness.explanation_methods:
```

Replace lines 362-396 (the method loop and everything inside the `for mode in modes_to_eval` block after `adv_examples` limiting) with:

```python
        # Combine vanilla and PA methods
        all_methods = list(config.robustness.explanation_methods)
        if not skip_pa:
            all_methods.extend(config.robustness.pa_explanation_methods)

        for method in all_methods:
            logger.info(f"  --- Method: {method} ({mode}) ---")
            try:
                explain_fn = make_explain_fn(
                    method, dnn_model, dnn_wrapper, rf_wrapper,
                    dataset, device, config,
                )
            except Exception as e:
                logger.error(f"  Failed to create explain_fn for {method}: {e}")
                continue

            for adv_name, X_adv in adv_examples.items():
                parts = adv_name.split("_eps")
                attack_name = parts[0]
                epsilon = float(parts[1]) if len(parts) > 1 else 0.0

                try:
                    result = evaluate_robustness_for_method(
                        method_name=method,
                        explain_fn=explain_fn,
                        predict_fn=dnn_wrapper.predict_proba,
                        X_clean=X_clean,
                        X_adv=X_adv,
                        attack_name=attack_name,
                        epsilon=epsilon,
                        config=config.robustness,
                    )
                    result["dataset"] = dataset.dataset_name
                    result["constraint_mode"] = mode
                    all_results.append(result)
                except Exception as e:
                    logger.error(
                        f"  Robustness eval failed for {method}/{adv_name} ({mode}): {e}",
                        exc_info=True,
                    )
```

- [ ] **Step 2: Add skip_pa parameter to phase_robustness signature**

Update the `phase_robustness` function signature to accept the PA skip flag. Change:

```python
def phase_robustness(
    dataset: DatasetBundle,
    dnn_model: NIDSNet,
    dnn_wrapper: DNNWrapper,
    rf_wrapper: RFWrapper,
    config: Experiment2Config,
    device: torch.device,
    adv_results: dict | None = None,
) -> list[dict]:
```

to:

```python
def phase_robustness(
    dataset: DatasetBundle,
    dnn_model: NIDSNet,
    dnn_wrapper: DNNWrapper,
    rf_wrapper: RFWrapper,
    config: Experiment2Config,
    device: torch.device,
    adv_results: dict | None = None,
    skip_pa: bool = False,
) -> list[dict]:
```

- [ ] **Step 3: Commit**

```bash
git add experiments/2/main.py
git commit -m "feat(exp2): evaluate PA methods side-by-side in robustness phase"
```

---

### Task 7: Update main.py — CLI flag and orchestrator wiring

**Files:**
- Modify: `experiments/2/main.py:718-823`

- [ ] **Step 1: Add --no-pa CLI argument**

In the `parse_args()` function, after the `--no-unconstrained` argument (around line 776), add:

```python
    parser.add_argument(
        "--no-pa",
        action="store_true",
        help="Skip protocol-aware (PA) explanation methods in robustness evaluation",
    )
```

- [ ] **Step 2: Wire the flag through main() to run_experiment()**

In `main()`, after line 799 (`config.attack.run_unconstrained = False`), add:

```python
    skip_pa = args.no_pa
```

Then update `run_experiment` signature to accept `skip_pa`:

Change:
```python
def run_experiment(config: Experiment2Config, datasets: list[str], phases: list[str]):
```
to:
```python
def run_experiment(config: Experiment2Config, datasets: list[str], phases: list[str], skip_pa: bool = False):
```

And update the call to `run_experiment` in `main()`:

```python
    run_experiment(config, datasets, phases, skip_pa=skip_pa)
```

- [ ] **Step 3: Pass skip_pa to phase_robustness in run_experiment**

In `run_experiment`, find the call to `phase_robustness` (around line 688) and add the parameter:

Change:
```python
                rob_results = phase_robustness(
                    dataset, dnn_model, dnn_wrapper, rf_wrapper,
                    config, device,
                )
```
to:
```python
                rob_results = phase_robustness(
                    dataset, dnn_model, dnn_wrapper, rf_wrapper,
                    config, device, skip_pa=skip_pa,
                )
```

- [ ] **Step 4: Update logging in main() to show PA status**

After line 817 (`logger.info(f"Robustness samples: {config.robustness.num_samples}")`), add:

```python
    logger.info(f"PA methods: {'disabled' if skip_pa else 'enabled'}")
```

- [ ] **Step 5: Commit**

```bash
git add experiments/2/main.py
git commit -m "feat(exp2): add --no-pa CLI flag and wire through orchestrator"
```

---

### Task 8: Delete feature_constraints.py

**Files:**
- Delete: `experiments/2/feature_constraints.py`

- [ ] **Step 1: Verify feature_constraints.py is no longer imported anywhere**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids && grep -r "feature_constraints" experiments/2/ --include="*.py"
```

Expected: No matches (all imports were replaced in Task 5).

- [ ] **Step 2: Delete the file**

```bash
git rm experiments/2/feature_constraints.py
```

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor(exp2): remove feature_constraints.py (replaced by pa_constraints.py)"
```

---

### Task 9: Smoke test — full module load and dry run

**Files:** None (verification only)

- [ ] **Step 1: Verify main.py loads without import errors**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids && uv run python -c "
import sys
sys.path.insert(0, 'experiments/1')
sys.path.insert(0, 'experiments/2')
import importlib.util
spec = importlib.util.spec_from_file_location('exp2_main', 'experiments/2/main.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('All imports successful')
print('Config:', mod.Experiment2Config())
"
```

Expected: No import errors, config prints successfully.

- [ ] **Step 2: Verify CLI help shows new flags**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids && uv run python experiments/2/main.py --help
```

Expected: Help output includes `--no-pa`, `--no-constrained`, `--no-unconstrained`.

- [ ] **Step 3: Run a minimal adversarial phase to verify pa_constraints projector**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids && uv run python experiments/2/main.py \
    --datasets nsl-kdd \
    --phase adversarial \
    --num-attack-samples 10 \
    --no-unconstrained
```

Expected: Constrained FGSM/PGD attacks complete using pa_xai constraints. Check logs for "Running CONSTRAINED attacks (pa_xai constraints)".

- [ ] **Step 4: Run a minimal robustness phase to verify all 8 methods**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids && uv run python experiments/2/main.py \
    --datasets nsl-kdd \
    --phase robustness \
    --num-robustness-samples 5
```

Expected: Logs show all 8 methods evaluated (SHAP, LIME, IG, DeepLIFT, PA-SHAP, PA-LIME, PA-IG, PA-DeepLIFT). Results saved to `experiments/2/results/nsl-kdd/robustness/robustness_metrics.json`.

- [ ] **Step 5: Verify --no-pa flag works**

Run:
```bash
cd /Users/abanisenioluwaorojo/projects/sok-nids && uv run python experiments/2/main.py \
    --datasets nsl-kdd \
    --phase robustness \
    --num-robustness-samples 5 \
    --no-pa
```

Expected: Logs show only 4 vanilla methods (SHAP, LIME, IG, DeepLIFT). No PA- methods appear.
