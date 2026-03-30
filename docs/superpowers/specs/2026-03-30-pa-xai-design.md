# pa_xai: Domain-Constrained XAI for Network Intrusion Detection

## Problem

Standard XAI methods (LIME, Integrated Gradients, DeepLIFT, SHAP) are domain-blind when applied to NIDS. Each suffers from a different manifestation of the same flaw:

- **LIME**: Random perturbations generate physically impossible network flows (negative byte counts, fractional flags, cross-protocol features).
- **Integrated Gradients**: The default all-zeros baseline is an impossible network state, and every intermediate interpolation point along the integration path is a fractional, impossible packet.
- **DeepLIFT**: Same baseline problem as IG — activation differences computed against an impossible zero-state corrupt the rescale multipliers.
- **SHAP**: Masking features by replacing with background dataset means creates "Frankenstein" packets — e.g., UDP payload with TCP protocol header.

## Solution

A unified `pa_xai` package that applies protocol-aware domain constraints to all four methods, each at the mathematically appropriate intervention point:

| Method | Domain-Blindness Source | Fix |
|--------|------------------------|-----|
| LIME | Random perturbation | Constrained neighborhood generation (existing `pa_lime`) |
| IG | Zero baseline + fractional path | Nearest Benign Prototype baseline + optional path clamping |
| DeepLIFT | Zero baseline | Nearest Benign Prototype baseline (no path — only two discrete evaluations) |
| SHAP | Cross-protocol background masking | Protocol-filtered background + coalition clamping (KernelSHAP only) |

## Architecture

### Approach: Thin Wrappers with Shared Core

Each explainer is a standalone class wrapping the underlying library (Captum/SHAP) and injecting constraints at method-specific points. No abstract base class — shared infrastructure lives in `core/`.

### Package Structure

```
pa_xai/
├── __init__.py                 # Top-level re-exports all public API
├── core/
│   ├── __init__.py
│   ├── schemas.py              # DatasetSchema, HierarchicalConstraint, built-in schemas
│   ├── constraints.py          # ConstraintEnforcer
│   ├── result.py               # ExplanationResult (unified across methods)
│   ├── baseline.py             # NearestBenignPrototype selection
│   └── metrics.py              # Shared metrics: semantic_robustness, sparsity
├── lime/
│   ├── __init__.py
│   ├── explainer.py            # ProtocolAwareLIME
│   ├── fuzzer.py               # DomainConstraintFuzzer
│   └── metrics.py              # LIME-specific: fidelity
├── ig/
│   ├── __init__.py
│   ├── explainer.py            # ProtocolAwareIG
│   └── metrics.py              # IG-specific: path_convergence
├── deeplift/
│   ├── __init__.py
│   ├── explainer.py            # ProtocolAwareDeepLIFT
│   └── metrics.py              # DeepLIFT-specific: convergence_delta
└── shap/
    ├── __init__.py
    ├── explainer.py            # ProtocolAwareSHAP (multi-backend)
    └── metrics.py              # SHAP-specific: additivity_check
```

### pa_lime Backwards Compatibility

The existing `pa_lime/` package becomes a thin re-export shim importing from `pa_xai`. All existing `from pa_lime import ...` statements continue to work unchanged.

---

## Component Designs

### 1. Core: Nearest Benign Prototype (`core/baseline.py`)

Shared baseline selection used by IG, DeepLIFT, and SHAP.

**3-Step Selection Strategy:**

1. **Hard Constraint Filtering (Protocol Lock):** Filter `X_train` to only benign samples (`y_train == benign_label`) that share the exact same protocol as the target packet.
2. **Feature Space Distance:** Compute Euclidean (L2) distance between the target packet and every candidate in the filtered benign subset. Data is assumed MinMaxScaled, making L2 stable.
3. **Prototype Selection:** Return the nearest single sample (`strategy="nearest"`) or the median of the top-k closest (`strategy="median_k"`).

```python
def get_protocol_valid_baseline(
    x_row: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    schema: DatasetSchema,
    benign_label: int = 0,
    top_k: int = 1,
    strategy: str = "nearest",    # "nearest" | "median_k"
) -> np.ndarray:
```

### 2. Core: Unified ExplanationResult (`core/result.py`)

```python
@dataclass(frozen=True)
class ExplanationResult:
    feature_names: list[str]
    attributions: np.ndarray          # shape (D,) — unified name
    method: str                       # "pa_lime" | "pa_ig" | "pa_deeplift" | "pa_shap"
    predicted_class: int | None
    num_samples: int | None           # LIME/KernelSHAP neighborhood size

    # Method-specific optional fields
    r_squared: float | None = None            # LIME only
    intercept: float | None = None            # LIME only
    local_prediction: float | None = None     # LIME only
    convergence_delta: float | None = None    # IG / DeepLIFT
    expected_value: float | None = None       # SHAP
    baseline_used: np.ndarray | None = None   # IG / DeepLIFT

    def top_features(self, k=10, absolute=True) -> list[tuple[str, float]]: ...
    def as_dict(self) -> dict[str, float]: ...
```

LIME's existing `coefficients` field becomes `attributions`. The `pa_lime` shim exposes `coefficients` as a property alias for backwards compatibility.

### 3. Core: Shared Metrics (`core/metrics.py`)

Moved from `pa_lime/metrics.py`:

- `semantic_robustness(x_row, explainer, predict_fn_or_model, epsilon, n_iter)` — works with any explainer that has `explain_instance()`. Generates constrained mutations, re-explains, returns Spearman correlation.
- `sparsity(result, threshold)` — counts features above importance threshold.

---

### 4. Protocol-Aware Integrated Gradients (`ig/explainer.py`)

**Constructor:**
```python
class ProtocolAwareIG:
    def __init__(
        self,
        schema: DatasetSchema,
        model: torch.nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        benign_label: int = 0,
        baseline_top_k: int = 1,
        baseline_strategy: str = "nearest",
        constrain_path: bool = True,
        multiply_by_inputs: bool = True,
        tcp_label_value: int = 6,
    ):
```

**`explain_instance()` flow:**

1. Call `get_protocol_valid_baseline(x_row, ...)` — protocol-locked nearest benign prototype
2. Convert `x_row` and baseline to tensors
3. **If `constrain_path=True`:** Manually implement the integration loop:
   - For each alpha in the quadrature schedule (Gauss-Legendre or Riemann):
     - Compute `x_interp = baseline + alpha * (input - baseline)`
     - Convert to numpy, apply `ConstraintEnforcer.enforce()`, convert back
     - Compute gradient via `torch.autograd.grad()`
     - Accumulate weighted gradients
   - Final attribution = accumulated gradients × (input - baseline)
4. **If `constrain_path=False`:** Pass the protocol-valid baseline directly to Captum's `IntegratedGradients.attribute()` — baseline fix only, no path modification
5. Package into `ExplanationResult`

Non-differentiable clamping is acceptable: gradients are computed on the clamped state, not through the clamp operation.

**IG-specific metric:**
```python
def path_convergence(result: ExplanationResult) -> float:
    """sum(attributions) vs (F(input) - F(baseline)). Closer to 0 = better."""
```

---

### 5. Protocol-Aware DeepLIFT (`deeplift/explainer.py`)

**Constructor:**
```python
class ProtocolAwareDeepLIFT:
    def __init__(
        self,
        schema: DatasetSchema,
        model: torch.nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        benign_label: int = 0,
        baseline_top_k: int = 1,
        baseline_strategy: str = "nearest",
        multiply_by_inputs: bool = True,
        eps: float = 1e-10,
        tcp_label_value: int = 6,
    ):
```

No `constrain_path` parameter — DeepLIFT evaluates only two discrete points (baseline and input). No intermediate samples exist.

**`explain_instance()` flow:**

1. Call `get_protocol_valid_baseline(x_row, ...)` — same selection as IG
2. Convert to tensors
3. Call Captum's `DeepLift.attribute(inputs, baselines, target, return_convergence_delta)`
4. Suppress hook warnings, handle RNN via `_disable_cudnn_for_rnn()`
5. Package into `ExplanationResult`

The entire domain-constraint fix is the baseline replacement. This is the simplest explainer.

**DeepLIFT-specific metric:**
```python
def convergence_delta(result: ExplanationResult) -> float:
    """Completeness: sum(attributions) vs (F(input) - F(baseline)). Closer to 0 = better."""
```

---

### 6. Protocol-Aware SHAP (`shap/explainer.py`)

**Constructor:**
```python
class ProtocolAwareSHAP:
    def __init__(
        self,
        schema: DatasetSchema,
        model: torch.nn.Module | Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        benign_label: int = 0,
        backend: str = "kernel",       # "kernel" | "deep" | "tree"
        n_background: int = 100,
        tcp_label_value: int = 6,
    ):
```

**`explain_instance()` flow varies by backend:**

#### KernelSHAP (`backend="kernel"`)
Both constraint fixes active:
1. **Protocol-filter background:** Select benign samples matching `x_row`'s protocol, subsample to `n_background`.
2. **Create `_ConstrainedKernelExplainer`:** Subclasses `shap.KernelExplainer`, overrides `addsample()` to apply `ConstraintEnforcer.enforce()` on every coalition sample after the mask is applied.
3. Call `shap_values()`, package into `ExplanationResult`.

#### DeepSHAP (`backend="deep"`)
Background filtering only (no synthetic samples to clamp):
1. Protocol-filter background.
2. Pass to `shap.DeepExplainer(model, filtered_background)`.
3. Fall back to `shap.GradientExplainer` for RNN models.
4. Call `shap_values()`, package into `ExplanationResult`.

#### TreeSHAP (`backend="tree"`)
Background filtering only (purely analytical algorithm):
1. Protocol-filter background.
2. Create `shap.TreeExplainer(model, data=filtered_background, feature_perturbation="interventional")`.
3. Call `shap_values()`, package into `ExplanationResult`.

**Internal `_ConstrainedKernelExplainer`:**
```python
class _ConstrainedKernelExplainer(shap.KernelExplainer):
    def __init__(self, model, data, schema, enforcer, protocol_value, tcp_label_value, **kwargs):
        super().__init__(model, data, **kwargs)
        self.schema = schema
        self.enforcer = enforcer
        self.protocol_value = protocol_value
        self.tcp_label_value = tcp_label_value

    def addsample(self, x, m, w):
        offset = self.nsamplesAdded
        super().addsample(x, m, w)
        n_new = self.nsamplesAdded - offset
        if n_new > 0:
            new_rows = self.synth_data[offset * self.N : self.nsamplesAdded * self.N]
            self.enforcer.enforce(
                new_rows, self.protocol_value,
                self.schema.protocol_encoding, self.tcp_label_value
            )
```

**SHAP-specific metric:**
```python
def additivity_check(result: ExplanationResult, expected_value: float) -> float:
    """sum(shap_values) + expected_value ≈ model(x). Returns absolute error."""
```

---

## Public API

### Top-level exports (`pa_xai/__init__.py`)

```python
__all__ = [
    # Core
    "DatasetSchema", "HierarchicalConstraint", "ConstraintEnforcer",
    "ExplanationResult", "get_protocol_valid_baseline",
    "CIC_IDS_2017", "CSE_CIC_IDS2018", "NSL_KDD",
    "UNSW_NB15_NATIVE", "UNSW_NB15_CIC",
    "BUILTIN_SCHEMAS", "get_schema",

    # Explainers
    "ProtocolAwareLIME",
    "ProtocolAwareIG",
    "ProtocolAwareDeepLIFT",
    "ProtocolAwareSHAP",

    # Shared metrics
    "semantic_robustness", "sparsity",

    # Method-specific metrics
    "fidelity", "path_convergence", "convergence_delta", "additivity_check",
]
```

### Usage Example

```python
from pa_xai import (
    ProtocolAwareLIME, ProtocolAwareIG,
    ProtocolAwareDeepLIFT, ProtocolAwareSHAP,
    CIC_IDS_2017, semantic_robustness,
)

schema = CIC_IDS_2017

# LIME — model-agnostic, predict_fn at explain time
lime = ProtocolAwareLIME(schema)
result = lime.explain_instance(x, predict_fn, num_samples=5000)

# IG — model + training data for baseline
ig = ProtocolAwareIG(schema, model, X_train, y_train, constrain_path=True)
result = ig.explain_instance(x, n_steps=50)

# DeepLIFT — model + training data for baseline
dl = ProtocolAwareDeepLIFT(schema, model, X_train, y_train)
result = dl.explain_instance(x)

# SHAP — model + training data for background filtering
shap_k = ProtocolAwareSHAP(schema, model, X_train, y_train, backend="kernel")
shap_d = ProtocolAwareSHAP(schema, model, X_train, y_train, backend="deep")
shap_t = ProtocolAwareSHAP(schema, rf_model, X_train, y_train, backend="tree")
result = shap_k.explain_instance(x)

# Metrics work across all methods
robustness = semantic_robustness(x, ig, model)
```

---

## Testing Strategy

```
tests/test_pa_xai/
├── test_core/
│   ├── test_schemas.py            # Existing schema tests (moved)
│   ├── test_constraints.py        # Existing constraint tests (moved)
│   ├── test_baseline.py           # NearestBenignPrototype tests
│   └── test_metrics.py            # Shared metrics tests
├── test_lime/
│   ├── test_explainer.py          # Existing LIME tests (adjusted imports)
│   ├── test_fuzzer.py             # Existing fuzzer tests (adjusted imports)
│   └── test_metrics.py            # LIME-specific metrics
├── test_ig/
│   ├── test_explainer.py          # IG explanation tests
│   ├── test_baseline_selection.py # Protocol-locked baseline verification
│   ├── test_path_clamping.py      # constrain_path=True enforcement
│   └── test_metrics.py            # path_convergence metric
├── test_deeplift/
│   ├── test_explainer.py          # DeepLIFT explanation tests
│   ├── test_baseline_selection.py # Protocol-locked baseline verification
│   └── test_metrics.py            # convergence_delta metric
├── test_shap/
│   ├── test_kernel_explainer.py   # KernelSHAP + coalition clamping
│   ├── test_deep_explainer.py     # DeepSHAP + filtered background
│   ├── test_tree_explainer.py     # TreeSHAP + filtered background
│   ├── test_background_filter.py  # Protocol filtering logic
│   └── test_metrics.py            # additivity_check metric
└── test_integration.py            # Cross-method end-to-end workflows
```

**Key test patterns:**
- **Baseline tests:** Protocol filtering (TCP input → TCP baseline), nearest-neighbor correctness, median-k smoothing
- **Path clamping tests (IG):** Every interpolation point satisfies non-negativity, hierarchical, discrete, TCP-gating constraints
- **Coalition clamping tests (KernelSHAP):** All synthetic samples satisfy constraints before model evaluation
- **Background filtering tests (all SHAP):** Background only contains protocol-matching benign flows
- **Backwards compatibility:** `from pa_lime import *` continues to work

---

## Dependencies

Existing in `pyproject.toml`:
- `captum >= 0.7.0` (IG, DeepLIFT)
- `shap >= 0.45.0` (all SHAP backends)
- `torch` (model handling)
- `numpy`, `scipy`, `scikit-learn` (distance computation, Ridge regression)

No new dependencies required.
