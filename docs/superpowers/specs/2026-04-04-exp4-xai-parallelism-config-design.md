# Experiment 4: XAI Parallelism, Config Centralization, and Normal/PA Mode Support

**Date:** 2026-04-04
**Status:** Approved

## Problem

Experiment 4 (XAI-Driven Dimensionality Reduction vs Statistical Baselines) has three issues:

1. **No parallelism** — LIME runs sequentially, all XAI methods run sequentially, no checkpointing. On a multi-core/multi-GPU machine this is 10-20x slower than it needs to be.
2. **Hardcoded config** — Exp 4 has its own `config.py` with duplicated defaults instead of reading from the shared `experiments/config.yaml`.
3. **No PA-XAI support** — Only runs vanilla SHAP/LIME. Cannot compare normal vs protocol-aware feature selection, which is needed for the SoK comparison.
4. **Missing XAI methods** — Only SHAP and LIME. No IG or DeepLIFT, unlike Experiments 1/2/3.

## Approach

**Import Exp 1's explainer orchestrators** rather than duplicating code. Exp 4 already depends on Exp 1 for `data_loader` and model architectures. The explainers are a natural extension.

## Design

### 1. config.yaml Changes

Add `experiment_4` section to `experiments/config.yaml`:

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

Shared sections (`seed`, `datasets`, `data`, `models`, `explainer`) are already present and reused as-is.

### 2. config.py Rewrite

- Load from `config.yaml` via `config_loader.load_yaml_config()`
- Reuse Exp 1's dataclasses for shared configs (`DataConfig`, `DNNConfig`, `RFConfig`, `ExplainerConfig`) via `importlib` — same pattern Exp 3 uses
- Keep Exp 4-only dataclasses (`CNNConfig`, `SVMConfig`, `StatisticalSelectionConfig`, `XAISelectionConfig`) but populate them from YAML
- Add `xai_modes: list[str]` field to `ExperimentConfig`
- Provide `load_experiment_config()` factory that reads YAML, constructs dataclasses, returns `ExperimentConfig`
- CLI args override YAML values after loading

### 3. xai_selection.py Rewrite

Core change. Replaces inline SHAP/LIME with Exp 1's orchestrators.

**Imports:**
- `explainers.generate_all_explanations` (normal mode)
- `pa_explainers.pa_generate_all_explanations` (pa_xai mode)

**Attribution → Feature Importance:**
Both orchestrators return `list[ExplanationResult]`, each with `attributions: np.ndarray` of shape `(n_samples, n_features)`. Convert to per-feature importance via `mean(abs(attributions), axis=0)`.

**Iterative Pruning:**
For each `ExplanationResult`, feed the feature importance into the existing `_iterative_prune()` function (unchanged — it's method-agnostic).

**Dual Mode Execution:**
When `xai_modes: [normal, pa]`, run the pipeline twice. Results tagged with prefix: `"SHAP-DNN"` vs `"PA-SHAP-DNN"` for downstream comparison.

**Parallelism (inherited from Exp 1):**
- LIME: threading/loky with 75% CPUs, checkpointing every 500 instances
- Phase 2: concurrent LIME + tree SHAP via ThreadPoolExecutor
- Phase 3: multi-GPU IG || DeepLIFT distribution
- cuDNN disabling for RNN models

**XAI Methods (now 4 instead of 2):**
- SHAP (DNN via DeepExplainer, RF via TreeExplainer)
- LIME (DNN, RF)
- Integrated Gradients (DNN)
- DeepLIFT (DNN)

Total feature selection methods per mode: up to 6 (SHAP-DNN, SHAP-RF, LIME-DNN, LIME-RF, IG-DNN, DeepLIFT-DNN).

### 4. main.py Changes

- Add `--xai-mode` CLI flag: `n` (normal), `p` (PA), `both` (default from YAML)
- `phase_select` loops over `config.xai_modes`, calling appropriate orchestrator per mode
- Pass `DatasetBundle` directly to orchestrators (it already has all required fields)
- Pass checkpoint dir: `config.output_dir / dataset_name / "checkpoints"`
- `load_or_train_baseline_models` unchanged
- `phase_benchmark` unchanged (already benchmarks any list of `FeatureSelectionResult`)

### 5. Files Changed

| File | Action |
|------|--------|
| `experiments/config.yaml` | Add `experiment_4` section |
| `experiments/4/config.py` | Rewrite: load from YAML, reuse Exp 1 dataclasses |
| `experiments/4/xai_selection.py` | Rewrite: import Exp 1 orchestrators, add IG/DeepLIFT, normal+pa |
| `experiments/4/main.py` | Add `--xai-mode`, checkpoint dir, dataset bundle passthrough |
| `experiments/4/feature_selection.py` | No changes |
| `experiments/4/models.py` | No changes |
| `experiments/4/evaluation.py` | No changes |
| `experiments/4/visualization.py` | No changes |

### 6. Result

- `uv run python experiments/4/main.py` — runs both normal + PA XAI by default
- `--xai-mode n` for normal only, `--xai-mode p` for PA only
- Config adjusted in one place: `experiments/config.yaml`
- LIME: sequential → parallel (10-20x speedup)
- IG + DeepLIFT added as feature importance sources
- Multi-GPU distribution for gradient-based methods
- Checkpointing for LIME crash resilience
- Normal vs PA feature selection comparison in downstream benchmarks
