# Wire `--xai_mode` into Experiment 1

## Problem

Experiment 1 runs unconstrained XAI methods (SHAP, LIME, IG, DeepLIFT) that produce domain-blind explanations. The new `pa_xai` package provides protocol-aware versions. We need a CLI flag to switch between normal and protocol-aware modes, with results going to separate directories.

## Solution

Add `--xai_mode` / `-x` flag to `experiments/1/main.py`:
- `n` (default) — normal unconstrained XAI → `results/normal/`
- `p` — protocol-aware XAI via `pa_xai` → `results/protocol-aware/`

## Changes

### 1. CLI Argument

Add to `parse_args()`:

```python
parser.add_argument(
    "-x", "--xai-mode",
    choices=["n", "p"],
    default="n",
    help="XAI mode: n=normal (unconstrained), p=protocol-aware (pa_xai)",
)
```

### 2. Output Directory

In `main()`, derive the output directory from the mode:

```python
mode_dir = "normal" if args.xai_mode == "n" else "protocol-aware"
config.output_dir = Path("experiments/1/results") / mode_dir
```

If `--output-dir` is explicitly passed, it overrides this. The default `ExperimentConfig.output_dir` (`experiments/1/results`) becomes the base, not the final path.

### 3. Model Loading

Models are trained once and shared. Both modes load models from `results/normal/models/` since training is XAI-mode-independent:

```python
config.models_dir = Path("experiments/1/results/normal")
```

`load_models()` and `load_cnn_models()` in `phase_explain()` and `phase_evaluate()` use `config.models_dir` instead of `config.output_dir` for model loading. The `phase_train()` function always saves to `config.models_dir`.

Add `models_dir` field to `ExperimentConfig`:

```python
@dataclass
class ExperimentConfig:
    ...
    output_dir: Path = Path("experiments/1/results/normal")
    models_dir: Path = Path("experiments/1/results/normal")
```

### 4. Schema Mapping

Map experiment dataset names to `pa_xai` built-in schemas. The `data_loader.py` config shows UNSW-NB15 uses `cic_unsw-nb15_augmented_dataset` (CICFlowMeter features):

```python
DATASET_SCHEMA_MAP = {
    "nsl-kdd": "NSL-KDD",
    "cic-ids-2017": "CIC-IDS-2017",
    "unsw-nb15": "UNSW-NB15-CICFlowMeter",
    "cse-cic-ids2018": "CSE-CIC-IDS2018",
}
```

A helper resolves the schema:

```python
from pa_xai import get_schema

def _get_pa_schema(dataset_name: str):
    return get_schema(DATASET_SCHEMA_MAP[dataset_name])
```

### 5. Protocol-Aware Explanation Functions

Add a new file `experiments/1/pa_explainers.py` containing protocol-aware wrappers that return the same `ExplanationResult` format as the existing `explainers.py`:

```python
def pa_explain_shap_dnn(model, X_explain, X_train, y_train, schema, device, config) -> ExplanationResult
def pa_explain_shap_rf(model, X_explain, X_train, y_train, schema, config) -> ExplanationResult
def pa_explain_lime(predict_fn, X_explain, schema, model_name, config) -> ExplanationResult
def pa_explain_ig(model, X_explain, X_train, y_train, schema, device, config) -> ExplanationResult
def pa_explain_deeplift(model, X_explain, X_train, y_train, schema, device, config) -> ExplanationResult
def pa_generate_all_explanations(..., schema) -> tuple[list[ExplanationResult], np.ndarray]
```

Each function:
1. Instantiates the corresponding `pa_xai` explainer with the dataset schema
2. Loops over `X_explain` samples (or batches) to produce per-sample attributions
3. Returns the same `ExplanationResult` dataclass from `explainers.py` (not `pa_xai.core.result`) for compatibility with the evaluation pipeline

Key differences from normal mode:
- **SHAP DNN** → `ProtocolAwareSHAP(schema, model, X_train, y_train, backend="deep")` — receives full `X_train`/`y_train` for protocol-filtered background
- **SHAP RF/XGB** → `ProtocolAwareSHAP(schema, model, X_train, y_train, backend="tree")`
- **LIME** → `ProtocolAwareLIME(schema)` — called per sample via `explain_instance()`
- **IG** → `ProtocolAwareIG(schema, model, X_train, y_train, constrain_path=True)`
- **DeepLIFT** → `ProtocolAwareDeepLIFT(schema, model, X_train, y_train)`

### 6. Wiring in main.py

In `phase_explain()`, add a branch based on `xai_mode`:

```python
if config.xai_mode == "p":
    from pa_explainers import pa_generate_all_explanations
    results, indices = pa_generate_all_explanations(
        ..., schema=_get_pa_schema(dataset.dataset_name),
    )
else:
    results, indices = generate_all_explanations(...)
```

Similarly for `_generate_cnn_explanations()` and `_make_explain_fn()` (used by `phase_evaluate()`).

Add `xai_mode` field to `ExperimentConfig`:

```python
@dataclass
class ExperimentConfig:
    ...
    xai_mode: str = "n"
```

### 7. What Doesn't Change

- **Training phase** — identical regardless of XAI mode
- **Evaluation metrics** — same metrics applied to both modes
- **Data loading** — same preprocessing pipeline
- **Result format** — same `ExplanationResult` dataclass, same file layout within the mode directory

## Testing

Verify with a small run:

```bash
# Normal mode (existing behavior)
python experiments/1/main.py --datasets nsl-kdd --phase explain --num-explain-samples 10 -x n

# Protocol-aware mode
python experiments/1/main.py --datasets nsl-kdd --phase explain --num-explain-samples 10 -x p
```

Both should produce results in `results/normal/` and `results/protocol-aware/` respectively with the same directory structure within.
