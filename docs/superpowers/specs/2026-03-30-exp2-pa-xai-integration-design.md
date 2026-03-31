# Experiment 2: pa_xai Integration Design

## Summary

Integrate the `pa_xai` library into Experiment 2 (Adversarial Robustness) to:
1. Replace the standalone `feature_constraints.py` with `pa_xai`'s `ConstraintEnforcer` + dataset schemas (single source of truth for domain constraints)
2. Add protocol-aware (PA) explainers alongside vanilla explainers for side-by-side robustness comparison
3. Add DeepLIFT (vanilla + PA) to the robustness evaluation methods

Scope is DNN-only, matching the current experiment 2 setup (FGSM/PGD are gradient-based).

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vanilla vs PA explainers | Side-by-side comparison | Stronger experimental contribution; measures whether PA explanations are more robust |
| Attack constraint system | Replace `feature_constraints.py` with `pa_xai` | Eliminates ~626 lines of duplicated constraint logic; ensures identical domain rules |
| Model scope | DNN-only | FGSM/PGD are gradient-based; matches current experiment 2 scope |
| DeepLIFT | Add to robustness eval | Completes the method coverage (SHAP, LIME, IG, DeepLIFT) |
| Constrained/unconstrained | Keep both modes | Core research question of experiment 2 |

## Architecture

### Approach: Thin wrapper module

Two new files + edits to `main.py` and `config.py`. Existing `attacks.py`, `robustness.py`, `scaffolding.py` remain untouched.

### New Files

#### `experiments/2/pa_constraints.py` (~80 lines)

Replaces `feature_constraints.py` by wrapping `pa_xai`'s `ConstraintEnforcer` + `DatasetSchema` into the `constraint_projector(X_adv, X_original, epsilon)` interface that `attacks.py` expects.

**Factory function:**
```python
def make_pa_constraint_projector(dataset_name, scaler, device) -> callable
```

**Projector closure logic:**
1. Convert torch tensors to numpy
2. Inverse-scale from [0,1] to original feature space using scaler's `data_min_` and `scale_` (where `scale_ = data_max_ - data_min_`)
3. Determine protocol value per sample from `X_original` (read from `schema.protocol_index`)
4. Group samples by protocol value, then call `ConstraintEnforcer.enforce()` per group — `enforce()` takes a single `protocol_value` for the whole batch, so mixed-protocol batches must be split, enforced separately, and reassembled
5. Re-scale back to [0,1]
6. Re-apply L-inf epsilon-ball projection: `delta = clamp(X_adv - X_original, -eps, eps); X_adv = clamp(X_original + delta, 0, 1)`
7. Convert back to torch

**Protocol grouping detail:** For datasets without a protocol column (e.g., `cic-iov-2024` if it ever gets a schema), pass `protocol_value=None` and `protocol_encoding="integer"` — the enforcer skips all protocol-gated steps.

**Key adapter concern:** `pa_xai`'s `ConstraintEnforcer` operates in the schema's native feature space (original values). The attack pipeline works in MinMax-scaled [0,1] space. The adapter must inverse-scale before enforcing, then re-scale after. The scaler's `data_min_` and `data_max_` arrays provide the transform.

**Schema mapping:** Reuses the same `DATASET_SCHEMA_MAP` as experiment 1:
- `"nsl-kdd"` -> `"NSL-KDD"`
- `"cic-ids-2017"` -> `"CIC-IDS-2017"`
- `"unsw-nb15"` -> `"UNSW-NB15-CICFlowMeter"`
- `"cse-cic-ids2018"` -> `"CSE-CIC-IDS2018"`

Note: `"cic-iov-2024"` does not have a `pa_xai` schema yet. If this dataset is run, the constrained mode should fall back gracefully (log a warning, skip constrained attacks for that dataset).

**Constraint spec serialization:** Provide a `pa_constraint_spec_to_dict(schema)` function for saving the constraint specification to JSON (replacing `spec_to_dict`).

#### `experiments/2/pa_explainers.py` (~200 lines)

Wraps `pa_xai` explainers into the `explain_fn(X) -> np.ndarray` signature that `robustness.py` expects.

**Functions:**

```python
def make_pa_explain_fn(
    method: str,           # "PA-SHAP", "PA-LIME", "PA-IG", "PA-DeepLIFT"
    dnn_model, dnn_wrapper, dataset, device, config
) -> callable
```

Returns a closure `fn(X: np.ndarray) -> np.ndarray` that:
1. Instantiates the appropriate `pa_xai` explainer with the dataset's schema
2. Calls `explain_instance()` for each sample in X
3. Stacks and returns attributions as `np.ndarray`

**Per-method details:**

| Method | pa_xai class | Key params |
|--------|-------------|------------|
| PA-SHAP | `ProtocolAwareSHAP` | `backend="deep"`, `n_background=config.shap_background_samples` |
| PA-LIME | `ProtocolAwareLIME` | `num_samples=config.lime_num_samples` |
| PA-IG | `ProtocolAwareIG` | `constrain_path=True`, `n_steps=config.ig_n_steps` |
| PA-DeepLIFT | `ProtocolAwareDeepLIFT` | Default params |

All gradient-based methods (PA-SHAP deep, PA-IG, PA-DeepLIFT) receive the unwrapped base model (not DataParallel). PA-IG and PA-DeepLIFT also receive `X_train` and `y_train` for baseline selection via `get_protocol_valid_baseline()`.

PA-LIME receives `dnn_wrapper.predict_proba` as the predict function.

### Modified Files

#### `experiments/2/config.py`

Changes to `RobustnessConfig`:
- Add `"DeepLIFT"` to `explanation_methods` default: `["SHAP", "LIME", "IG", "DeepLIFT"]`
- Add new field: `pa_explanation_methods: list[str] = ["PA-SHAP", "PA-LIME", "PA-IG", "PA-DeepLIFT"]`
- Add `deeplift_internal_batch_size: int = 4096`

#### `experiments/2/main.py`

**Import changes:**
- Remove: `from feature_constraints import build_constraints, make_constraint_projector, spec_to_dict`
- Add: `from pa_constraints import make_pa_constraint_projector, pa_constraint_spec_to_dict`
- Add: `from pa_explainers import make_pa_explain_fn`
- Add: `from explainers import explain_deeplift` (vanilla DeepLIFT from experiment 1)

**`make_explain_fn()` expansion:**
- Add `"DeepLIFT"` case using experiment 1's `explain_deeplift`
- For PA methods, delegate to `make_pa_explain_fn()` from `pa_explainers.py`

**`phase_adversarial()` changes:**
- Replace `build_constraints()` + `make_constraint_projector()` with `make_pa_constraint_projector()`
- Replace `spec_to_dict()` with `pa_constraint_spec_to_dict()`
- Handle missing schema for `cic-iov-2024` gracefully (skip constrained mode with warning)

**`phase_robustness()` changes:**
- Iterate over both `config.robustness.explanation_methods` and `config.robustness.pa_explanation_methods`
- Results tagged with method name (e.g., `"PA-SHAP"` vs `"SHAP"`) for side-by-side comparison

**CLI args:**
- Add `--no-pa` flag to skip PA explanation methods (useful for faster testing)

### Deleted Files

- `experiments/2/feature_constraints.py` — fully replaced by `pa_constraints.py`

## Data Flow

```
Adversarial phase:
  DNN model + test data
    -> FGSM/PGD (unconstrained: no projector)
    -> FGSM/PGD (constrained: pa_constraints projector via pa_xai ConstraintEnforcer)
    -> saved to results/{dataset}/adversarial/{unconstrained,constrained}/

Robustness phase:
  For each constraint_mode in [unconstrained, constrained]:
    For each method in [SHAP, LIME, IG, DeepLIFT, PA-SHAP, PA-LIME, PA-IG, PA-DeepLIFT]:
      explain_fn(X_clean) vs explain_fn(X_adv)
        -> Lipschitz, ExplSim, ClassEq metrics
        -> tagged with {method, attack, epsilon, constraint_mode}
```

## Output Format

Results JSON entries gain a method field that distinguishes vanilla from PA:
```json
{
  "method": "PA-SHAP",
  "attack": "fgsm",
  "epsilon": 0.1,
  "constraint_mode": "constrained",
  "dataset": "nsl-kdd",
  "lipschitz": { ... },
  "similarity": { ... },
  "classification_equivalence": { ... }
}
```

Plots updated to show vanilla vs PA as separate series (similar to how constrained vs unconstrained are already shown).

## Edge Cases

- **`cic-iov-2024` has no pa_xai schema:** Constrained attacks and PA explainers skip this dataset with a warning log. Vanilla explainers still run.
- **Captum hook conflicts:** PA-IG and PA-DeepLIFT clone the model (via `copy.deepcopy`) to avoid Captum hook state leaking between methods, following experiment 1's pattern.
- **Memory pressure:** PA explainers instantiate per-method, not all at once. Each explainer is created inside `make_pa_explain_fn()` closure, so GC can reclaim after the method's evaluation completes.

## Testing

- Run with `--phase adversarial --datasets nsl-kdd --num-attack-samples 50` to verify constrained attack projection works with pa_xai constraints
- Run with `--phase robustness --datasets nsl-kdd --num-robustness-samples 20` to verify all 8 methods produce attributions
- Compare constrained attack success rates before/after migration to ensure constraint behavior is equivalent
- Verify `--no-pa` flag skips PA methods correctly
