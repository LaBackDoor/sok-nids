# Experiment 3 Overhaul: Unified XAI Consensus, Interactions & Expert Alignment

**Date:** 2026-04-04
**Status:** Draft

## Problem Statement

Experiment 3 currently:
- Only runs normal-mode explainers (no protocol-aware / pa_xai support)
- Regenerates explanations from scratch even when Experiment 1 already produced them
- Has no resume/checkpoint capability — crashes require full reruns
- Runs everything sequentially despite being CPU/GPU parallelizable
- Has no shared config system across experiments (each experiment has its own Python dataclasses, duplicated seed values)

The target machine has 64 cores, 256 GB RAM, and 2x L40 GPUs (48 GB each).

## Goals

1. **Unified explanation pool**: Merge normal and PA-mode explanations into a single pool with prefixed keys, enabling within-mode, cross-mode, and combined consensus analysis in a single pass.
2. **Explanation reuse**: Load cached explanations from Experiment 1 before generating fresh ones.
3. **Resume on crash**: Phase-level and sub-phase-level checkpointing so no work is repeated.
4. **YAML config**: Single `experiments/config.yaml` with one `seed` value shared by all experiments.
5. **Parallelism**: Exploit all 64 cores and both GPUs for dataset-level, pair-level, and batch-level parallelism.
6. **Alignment as optional**: Keep the expert alignment phase (hardcoded domain knowledge, not live experts) but make it togglable.

## Non-Goals

- Changing Experiment 1 or 2's code (we only read their outputs).
- Adding new explainer methods.
- Changing the expert ground truth dictionaries.

---

## Architecture

### 1. YAML Config System

**File:** `experiments/config.yaml`

Single source of truth. One `seed` value at the top level. Per-experiment sections override defaults.

```yaml
seed: 42
datasets: [nsl-kdd, cic-ids-2017, unsw-nb15, cse-cic-ids2018]

data:
  data_root: data/
  test_size: 0.2
  val_split: 0.1
  apply_smote: true

models:
  dnn:
    hidden_layers: [1024, 768, 512]
    learning_rate: 0.01
    epochs: 100
    batch_size: 256
    dropout: 0.3
  rf:
    n_estimators: 100
    max_depth: 10
    n_jobs: -1

explainer:
  shap_background_samples: 100
  lime_num_samples: 5000
  num_explain_samples: 10000

experiment_1:
  xai_mode: both
  output_dir: experiments/1/results

experiment_2:
  output_dir: experiments/2/results

experiment_3:
  output_dir: experiments/3/results
  xai_modes: [normal, pa]

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

**Loader:** `experiments/config_loader.py` — reads YAML, returns typed dataclass trees. Each experiment imports `load_config()` and extracts its section. The existing Python dataclass definitions in each experiment's `config.py` remain as the typed interface — the YAML loader populates them.

### 2. Explanation Key Naming Convention

All explanations use a unified naming scheme:

| Mode | Model | Method | Key |
|------|-------|--------|-----|
| Normal | DNN | SHAP | `DNN_SHAP` |
| Normal | DNN | LIME | `DNN_LIME` |
| Normal | DNN | IG | `DNN_IG` |
| Normal | DNN | DeepLIFT | `DNN_DeepLIFT` |
| Normal | RF | SHAP | `RF_SHAP` |
| Normal | RF | LIME | `RF_LIME` |
| PA | DNN | SHAP | `DNN_PA-SHAP` |
| PA | DNN | LIME | `DNN_PA-LIME` |
| PA | DNN | IG | `DNN_PA-IG` |
| PA | DNN | DeepLIFT | `DNN_PA-DeepLIFT` |
| PA | RF | SHAP | `RF_PA-SHAP` |
| PA | RF | LIME | `RF_PA-LIME` |

Total: 12 explainer keys when both modes enabled. C(12,2) = 66 consensus pairs.

File naming on disk: `{KEY}_attributions.npy` (e.g., `DNN_PA-SHAP_attributions.npy`).

### 3. Explanation Load/Generate Pipeline

```
For each dataset, for each xai_mode in config.xai_modes:

  1. Check exp3 output dir for cached:
     exp3/results/<dataset>/explanations/<KEY>_attributions.npy
     → If ALL keys for this mode exist + explain_indices.npy matches, load and skip.

  2. Check exp1 output dir:
     exp1/results/normal/<dataset>/explanations/  (for normal mode)
     exp1/results/protocol-aware/<dataset>/explanations/  (for pa mode)
     → If attributions exist, copy to exp3 output dir.
     → Derive explain_labels.npy from: dataset.y_test[loaded_indices]

  3. Generate fresh only for missing explainer keys.
     → Normal mode: use explainers.py functions (from exp1)
     → PA mode: use pa_explainers.py functions (from exp1)

  4. Validate index alignment across modes:
     → Load explain_indices.npy from both normal and PA.
     → Assert np.array_equal(normal_indices, pa_indices).
     → If mismatch: log error with details, skip cross-mode consensus for this dataset.
```

### 4. Resume / Checkpoint System

#### Phase-level markers

Each phase writes a marker file on successful completion:

```
experiments/3/results/<dataset>/
  .phase_train.done
  .phase_explain_normal.done
  .phase_explain_pa.done
  .phase_consensus.done
  .phase_interactions.done
  .phase_alignment.done
  .phase_visualize.done
```

Marker files contain a JSON payload: `{"completed_at": "ISO timestamp", "config_hash": "sha256 of relevant config section"}`. On resume, if the config hash matches, the phase is skipped. If config changed, the marker is invalidated and the phase reruns.

#### Sub-phase checkpointing

**Consensus pairs:** Each pair result is saved as it completes:
```
experiments/3/results/<dataset>/consensus/DNN_SHAP__vs__DNN_PA-SHAP.json
```
On resume, `compute_pairwise_consensus()` loads existing pair files and only computes missing pairs.

**Per-attack consensus:** Same pattern — each `<attack>/<pair>.json` is saved individually.

**Explain phase:** Delegates to exp1's existing checkpointing (PA-LIME saves every 500 samples via `.npz` files in a checkpoint dir).

### 5. Parallelism Strategy

#### A. Dataset-level (ProcessPoolExecutor)

```python
# In run_experiment():
with ProcessPoolExecutor(max_workers=config.parallelism.max_dataset_workers) as pool:
    futures = {}
    for i, ds_name in enumerate(datasets):
        gpu_id = i % config.parallelism.num_gpu_devices
        futures[pool.submit(run_single_dataset, ds_name, config, gpu_id)] = ds_name
    for future in as_completed(futures):
        all_results[futures[future]] = future.result()
```

Each dataset gets pinned to a GPU via `torch.cuda.set_device(gpu_id)`. With 2 L40s and 4 datasets, 2 run concurrently.

#### B. Consensus pairs (ProcessPoolExecutor)

```python
# In compute_pairwise_consensus():
pairs = list(combinations(keys, 2))
uncached = [p for p in pairs if not _pair_cached(p, checkpoint_dir)]

with ProcessPoolExecutor(max_workers=config.parallelism.max_consensus_workers) as pool:
    futures = {pool.submit(_compute_single_pair, a, b, explanations[a], explanations[b], config): (a,b)
               for a, b in uncached}
    for future in as_completed(futures):
        result = future.result()
        _save_pair_checkpoint(result, checkpoint_dir)
        results.append(result)
```

66 pairs across 16 workers on 64 cores. Each pair is CPU-bound (Spearman/Kendall correlations on ~10k samples).

#### C. DNN interaction batching

Replace the sequential loop over 20 top features with a single batched SHAP call:

```python
# Current: 20 separate SHAP calls
for j in top_features:  # 20 iterations
    X_perturbed[:, j] = feature_means[j]
    pert_shap = explainer.shap_values(X_perturbed_tensor)

# New: 1 batched SHAP call
X_batch = np.tile(X, (len(top_features), 1, 1))  # (20, n, f)
for i, j in enumerate(top_features):
    X_batch[i, :, j] = feature_means[j]
X_flat = X_batch.reshape(-1, n_features)  # (20*n, f)
pert_shap_all = explainer.shap_values(torch.tensor(X_flat).to(device))
# Reshape back to (20, n, f) and compute interactions
```

Memory check: 20 * 500 * 122 features * 4 bytes = ~5 MB — fits easily in 48 GB L40.

#### D. Per-attack consensus (ProcessPoolExecutor)

Each attack type's consensus is independent. Parallelize across attacks with a process pool, reusing the consensus workers.

#### E. Visualization (ThreadPoolExecutor)

Individual plot generation has no data dependencies. Use `ThreadPoolExecutor(max_workers=8)` with matplotlib's Agg backend (already set).

### 6. Alignment Phase (Optional)

- Controlled by `experiment_3.alignment.enabled` in YAML.
- When enabled, both normal and PA explainers are scored against the hardcoded expert ground truth.
- The cross-mode comparison (does PA improve expert alignment?) is a novel contribution.
- When disabled, the phase is skipped entirely — no markers written.
- The existing `alignment.py` module is unchanged; it already accepts any `explanations` dict.

### 7. Experiment 3 Phase Flow (Updated)

```
1. Load YAML config
2. For each dataset (parallel across GPUs):
   a. TRAIN     — reuse exp1 models or train fresh [checkpoint: .phase_train.done]
   b. EXPLAIN   — for each xai_mode:
                   load from exp3 cache → load from exp1 → generate fresh
                   validate index alignment across modes
                   [checkpoint: .phase_explain_normal.done, .phase_explain_pa.done]
   c. CONSENSUS — unified pool, all C(k,2) pairs in parallel
                   within-mode + cross-mode pairs computed together
                   per-pair checkpointing for resume
                   [checkpoint: .phase_consensus.done]
   d. INTERACTIONS — RF TreeExplainer + DNN batched approximation
                     [checkpoint: .phase_interactions.done]
   e. ALIGNMENT — optional, scores all explainers against expert GT
                   [checkpoint: .phase_alignment.done]
   f. VISUALIZE — all plots in parallel via ThreadPool
                   [checkpoint: .phase_visualize.done]
3. Save combined summary
```

### 8. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `experiments/config.yaml` | **Create** | Shared YAML config |
| `experiments/config_loader.py` | **Create** | YAML → dataclass loader |
| `experiments/3/config.py` | **Modify** | Add `ParallelismConfig`, update `Experiment3Config` to accept YAML overrides |
| `experiments/3/main.py` | **Modify** | Add explanation reuse, pa_xai mode, dataset parallelism, checkpoint system, YAML loading |
| `experiments/3/consensus.py` | **Modify** | Add pair-level checkpointing, process pool parallelism |
| `experiments/3/interactions.py` | **Modify** | Batch DNN perturbations into single SHAP call |
| `experiments/3/visualizations.py` | **Modify** | ThreadPool for parallel plot generation |
| `experiments/3/pa_explainers.py` | **Create** | Thin wrapper importing exp1's pa_explainers for fresh generation |

### 9. Consensus Result Tagging

To distinguish within-mode vs cross-mode results at analysis/viz time without separate computation:

```python
def tag_pair(key_a: str, key_b: str) -> str:
    """Classify a consensus pair."""
    a_is_pa = "PA-" in key_a
    b_is_pa = "PA-" in key_b
    if a_is_pa == b_is_pa:
        return "within-pa" if a_is_pa else "within-normal"
    return "cross-mode"
```

Visualizations can filter/group by tag. The consensus heatmap gets colored annotations showing within vs cross pairs.

---

## Key Constraints

- **No changes to exp1/exp2 code** — we only read their output directories.
- **Backward compatible** — exp3 still works standalone if exp1 outputs don't exist (generates fresh).
- **Deterministic** — same seed + same config = same results, regardless of parallelism order.
- **Memory safe** — 12 explainer arrays of shape (10000, ~120) = ~60 MB total. Interaction matrices at 500 samples = ~30 MB. Well within 256 GB.
