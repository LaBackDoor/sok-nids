"""XAI explanation methods: SHAP, LIME, Integrated Gradients, DeepLIFT."""

import io
import json
import logging
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm

from config import ExplainerConfig
from models import SoftmaxModel

logger = logging.getLogger(__name__)


def _has_rnn_modules(model: torch.nn.Module) -> bool:
    """Check if model contains LSTM, GRU, or RNN layers."""
    return any(
        isinstance(m, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN))
        for m in model.modules()
    )


@contextmanager
def _disable_cudnn_for_rnn(model):
    """Disable cuDNN if model contains RNN layers.

    cuDNN's fused RNN kernels do not support backward in eval mode,
    which breaks gradient-based explainers (SHAP, IG, DeepLIFT).
    """
    has_rnn = _has_rnn_modules(model)
    if has_rnn:
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        if has_rnn:
            torch.backends.cudnn.enabled = prev


def _fix_xgb_base_score(model):
    """Fix XGBoost 2.x/3.x base_score vector for SHAP compatibility.

    XGBoost 2.x+ stores base_score as a per-class vector string
    (e.g. '[0E0,0E0,0E0]') for multi:softprob.  SHAP's XGBTreeModelLoader
    reads the model via save_raw("ubj") and calls float(base_score), which
    fails on vector strings.

    We monkey-patch XGBTreeModelLoader.__init__ to fix the base_score in
    the parsed UBJ dict before any float() conversion.
    """
    _patch_shap_xgb_loader()
    return model


def _patch_shap_xgb_loader():
    """Monkey-patch SHAP's XGBTreeModelLoader to handle vector base_score.

    Replaces __init__ with a version that normalises learner_model_param
    ["base_score"] from a vector string to its first scalar element right
    after the UBJ blob is decoded — before any float() call.
    """
    import io as _io
    import scipy.special
    from shap.explainers._tree import (
        SingleTree,
        XGBTreeModelLoader,
        _check_xgboost_version,
        decode_ubjson_buffer,
    )

    if getattr(XGBTreeModelLoader, '_base_score_patched', False):
        return

    def _patched_init(self, xgb_model):
        import xgboost as xgb

        _check_xgboost_version(xgb.__version__)
        model = xgb_model

        raw = xgb_model.save_raw(raw_format="ubj")
        with _io.BytesIO(raw) as fd:
            jmodel = decode_ubjson_buffer(fd)

        learner = jmodel["learner"]
        learner_model_param = learner["learner_model_param"]
        objective = learner["objective"]

        # --- FIX: normalise vector base_score to scalar ---
        bs = learner_model_param.get("base_score", "0.5")
        if isinstance(bs, str) and bs.startswith("["):
            scalar = bs.strip("[]").split(",")[0].strip()
            learner_model_param["base_score"] = scalar
            logger.debug("Patched SHAP XGB base_score '%s' → '%s'", bs, scalar)
        # --- END FIX ---

        booster = learner["gradient_booster"]
        n_classes = max(int(learner_model_param["num_class"]), 1)
        n_targets = max(int(learner_model_param["num_target"]), 1)
        n_targets = max(n_targets, n_classes)

        if "gbtree" in booster and "model" not in booster:
            booster = booster["gbtree"]
        if booster["model"].get("iteration_indptr", None) is not None:
            iteration_indptr = np.asarray(booster["model"]["iteration_indptr"], dtype=np.int32)
            diff = np.diff(iteration_indptr)
        else:
            n_parallel_trees = int(booster["model"]["gbtree_model_param"]["num_parallel_tree"])
            diff = np.repeat(n_targets * n_parallel_trees, model.num_boosted_rounds())
        if np.any(diff != diff[0]):
            raise ValueError("vector-leaf is not yet supported.:", diff)

        self.n_trees_per_iter = int(diff[0])
        self.n_targets = n_targets
        self.base_score = float(learner_model_param["base_score"])
        assert self.n_trees_per_iter > 0

        self.name_obj = objective["name"]
        self.name_gbm = booster["name"]
        base_score = float(learner_model_param["base_score"])
        if self.name_obj in ("binary:logistic", "reg:logistic"):
            self.base_score = scipy.special.logit(base_score)
        elif self.name_obj in (
            "reg:gamma", "reg:tweedie", "count:poisson",
            "survival:cox", "survival:aft",
        ):
            self.base_score = np.log(self.base_score)
        else:
            self.base_score = base_score

        self.num_feature = int(learner_model_param["num_feature"])
        self.num_class = int(learner_model_param["num_class"])

        trees = booster["model"]["trees"]
        self.num_trees = len(trees)

        self.node_parents = []
        self.node_cleft = []
        self.node_cright = []
        self.node_sindex = []
        self.children_default = []
        self.sum_hess = []
        self.values = []
        self.thresholds = []
        self.threshold_types = []
        self.features = []
        self.split_types = []
        self.categories = []

        feature_types = model.feature_types
        if feature_types is not None:
            cat_feature_indices = np.where(np.asarray(feature_types) == "c")[0]
            self.cat_feature_indices = cat_feature_indices if len(cat_feature_indices) > 0 else None
        else:
            self.cat_feature_indices = None

        def to_integers(data):
            assert isinstance(data, list)
            return np.asanyarray(data, dtype=np.uint8)

        for i in range(self.num_trees):
            tree = trees[i]
            self.node_parents.append(np.asarray(tree["parents"]))
            self.node_cleft.append(np.asarray(tree["left_children"], dtype=np.int32))
            self.node_cright.append(np.asarray(tree["right_children"], dtype=np.int32))
            self.node_sindex.append(np.asarray(tree["split_indices"], dtype=np.uint32))

            base_weight = np.asarray(tree["base_weights"], dtype=np.float32)
            if base_weight.size != self.node_cleft[-1].size:
                raise ValueError("vector-leaf is not yet supported.")

            default_left = to_integers(tree["default_left"])
            default_child = np.where(default_left == 1, self.node_cleft[-1], self.node_cright[-1]).astype(np.int64)
            self.children_default.append(default_child)
            self.sum_hess.append(np.asarray(tree["sum_hessian"], dtype=np.float64))

            is_leaf = self.node_cleft[-1] == -1
            split_cond = np.asarray(tree["split_conditions"], dtype=np.float32)
            leaf_weight = np.where(is_leaf, split_cond, 0.0)
            thresholds = np.where(is_leaf, 0.0, split_cond)
            thresholds = np.where(is_leaf, 0.0, np.nextafter(thresholds, -np.float32(np.inf)))
            threshold_types = np.zeros_like(thresholds, dtype=np.int32)

            self.values.append(leaf_weight.reshape(leaf_weight.size, 1))
            self.thresholds.append(thresholds)
            self.threshold_types.append(threshold_types)

            split_idx = np.asarray(tree["split_indices"], dtype=np.int64)
            self.features.append(split_idx)

            split_types = to_integers(tree["split_type"])
            self.split_types.append(split_types)
            cat_segments = tree["categories_segments"]
            cat_sizes = tree["categories_sizes"]
            cat_nodes = tree["categories_nodes"]
            assert len(cat_segments) == len(cat_sizes) == len(cat_nodes)
            cats = tree["categories"]

            tree_categories = self.parse_categories(cat_nodes, cat_segments, cat_sizes, cats, self.node_cleft[-1])
            self.categories.append(tree_categories)

    XGBTreeModelLoader._base_score_patched = True
    XGBTreeModelLoader.__init__ = _patched_init


@dataclass
class ExplanationResult:
    attributions: np.ndarray  # (n_samples, n_features)
    method_name: str
    model_name: str
    time_per_sample_ms: float
    total_time_s: float
    summary_plot_time_s: float = 0.0


def explain_shap_dnn(
    model: torch.nn.Module,
    X_explain: np.ndarray,
    X_background: np.ndarray,
    device: torch.device,
    config: ExplainerConfig,
) -> ExplanationResult:
    """Generate SHAP explanations for DNN.

    Uses DeepExplainer for standard DNNs (Linear, Conv, etc.) and
    GradientExplainer for models containing RNN layers (LSTM, GRU)
    which DeepExplainer cannot decompose.
    """
    import shap

    # Unwrap DataParallel
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    base_model.eval().float()

    # NOTE: Do NOT wrap with SoftmaxModel for SHAP DeepExplainer.
    # SHAP requires the additivity axiom (sum of SHAP values = model output - baseline),
    # which softmax's non-linear normalization breaks.

    bg_tensor = torch.tensor(X_background[: config.shap_background_samples], dtype=torch.float32).to(device)
    use_gradient = _has_rnn_modules(base_model)

    if use_gradient:
        logger.info(f"  SHAP (GradientExplainer) on {len(X_explain)} samples")
    else:
        logger.info(f"  SHAP (DeepExplainer) on {len(X_explain)} samples")

    with _disable_cudnn_for_rnn(base_model):
        if use_gradient:
            explainer = shap.GradientExplainer(base_model, bg_tensor)
        else:
            explainer = shap.DeepExplainer(base_model, bg_tensor)

        start = time.time()
        explain_tensor = torch.tensor(X_explain, dtype=torch.float32).to(device)
        if use_gradient:
            shap_values = explainer.shap_values(explain_tensor)
        else:
            shap_values = explainer.shap_values(explain_tensor, check_additivity=False)
        elapsed = time.time() - start

    # Get predicted classes
    base_model.eval()
    with torch.no_grad():
        preds = torch.argmax(base_model(explain_tensor), dim=1).cpu().numpy()

    # Extract attributions for the predicted class per sample
    if isinstance(shap_values, list):
        # List of (samples, features) arrays, one per class
        stacked = np.stack(shap_values, axis=0)  # (classes, samples, features)
        attributions = np.zeros((len(X_explain), X_explain.shape[1]), dtype=np.float32)
        for i, pred in enumerate(preds):
            attributions[i] = stacked[pred, i]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # (samples, features, classes)
        attributions = np.zeros((len(X_explain), X_explain.shape[1]), dtype=np.float32)
        for i, pred in enumerate(preds):
            attributions[i] = shap_values[i, :, pred]
    else:
        attributions = np.asarray(shap_values)

    return ExplanationResult(
        attributions=attributions,
        method_name="SHAP",
        model_name="DNN",
        time_per_sample_ms=(elapsed / len(X_explain)) * 1000,
        total_time_s=elapsed,
    )


def explain_shap_rf(
    model,
    X_explain: np.ndarray,
    config: ExplainerConfig,
) -> ExplanationResult:
    """Generate SHAP explanations for RF using TreeExplainer."""
    import shap

    logger.info(f"  SHAP (TreeExplainer) on {len(X_explain)} samples")
    explainer = shap.TreeExplainer(model)

    start = time.time()
    shap_values = explainer.shap_values(X_explain)
    elapsed = time.time() - start

    preds = model.predict(X_explain)

    # Extract attributions for the predicted class per sample
    if isinstance(shap_values, list):
        # List of (samples, features) arrays, one per class
        stacked = np.stack(shap_values, axis=0)  # (classes, samples, features)
        attributions = np.zeros((len(X_explain), X_explain.shape[1]), dtype=np.float32)
        for i, pred in enumerate(preds):
            attributions[i] = stacked[pred, i]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # (samples, features, classes)
        attributions = np.zeros((len(X_explain), X_explain.shape[1]), dtype=np.float32)
        for i, pred in enumerate(preds):
            attributions[i] = shap_values[i, :, pred]
    else:
        attributions = np.asarray(shap_values)

    return ExplanationResult(
        attributions=attributions,
        method_name="SHAP",
        model_name="RF",
        time_per_sample_ms=(elapsed / len(X_explain)) * 1000,
        total_time_s=elapsed,
    )


def _lime_explain_single(args):
    """Explain a single sample with LIME (top-level function for pickling)."""
    from lime.lime_tabular import LimeTabularExplainer

    sample, pred, X_train, feature_names, num_classes, n_features, lime_num_features, lime_num_samples, predict_fn = args

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=[str(i) for i in range(num_classes)],
        mode="classification",
        discretize_continuous=True,
    )
    exp = explainer.explain_instance(
        sample,
        predict_fn,
        num_features=lime_num_features,
        num_samples=lime_num_samples,
        labels=(int(pred),),
    )
    row = np.zeros(n_features)
    pred_label = int(pred)
    if pred_label in exp.as_map():
        feature_weights = dict(exp.as_map()[pred_label])
    else:
        feature_weights = dict(exp.as_map()[exp.available_labels()[0]])
    for feat_idx, weight in feature_weights.items():
        row[feat_idx] = weight
    return row


def explain_lime(
    predict_fn,
    X_explain: np.ndarray,
    X_train: np.ndarray,
    feature_names: list[str],
    num_classes: int,
    model_name: str,
    config: ExplainerConfig,
) -> ExplanationResult:
    """Generate LIME explanations (model-agnostic), parallelized across CPUs."""
    import os

    from sklearn.utils.parallel import Parallel, delayed
    from lime.lime_tabular import LimeTabularExplainer

    logger.info(f"  LIME on {len(X_explain)} samples for {model_name}")

    n_features = X_explain.shape[1]

    # Get predicted classes so LIME explains the predicted label
    preds = np.argmax(predict_fn(X_explain), axis=1)

    # Use 75% of available CPUs, but cap for large datasets to limit memory
    total_cpus = os.cpu_count() or 1
    n_jobs = max(1, int(total_cpus * 0.75))
    n_train = X_train.shape[0]
    if n_train > 5_000_000:
        n_jobs = min(n_jobs, 4)
    logger.info(f"  LIME parallelizing with {n_jobs}/{total_cpus} CPUs")

    # Instantiate once — avoids recomputing X_train statistics per sample
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=[str(c) for c in range(num_classes)],
        mode="classification",
        discretize_continuous=True,
    )

    def _explain_one(i):
        exp = explainer.explain_instance(
            X_explain[i],
            predict_fn,
            num_features=config.lime_num_features,
            num_samples=config.lime_num_samples,
            labels=(int(preds[i]),),
        )
        row = np.zeros(n_features)
        pred_label = int(preds[i])
        if pred_label in exp.as_map():
            feature_weights = dict(exp.as_map()[pred_label])
        else:
            feature_weights = dict(exp.as_map()[exp.available_labels()[0]])
        for feat_idx, weight in feature_weights.items():
            row[feat_idx] = weight
        return row

    start = time.time()
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=1)(
        delayed(_explain_one)(i) for i in range(len(X_explain))
    )
    attributions = np.array(results)
    elapsed = time.time() - start

    return ExplanationResult(
        attributions=attributions,
        method_name="LIME",
        model_name=model_name,
        time_per_sample_ms=(elapsed / len(X_explain)) * 1000,
        total_time_s=elapsed,
    )


def explain_ig(
    model: torch.nn.Module,
    X_explain: np.ndarray,
    device: torch.device,
    config: ExplainerConfig,
) -> ExplanationResult:
    """Generate Integrated Gradients explanations for DNN."""
    from captum.attr import IntegratedGradients

    logger.info(f"  Integrated Gradients on {len(X_explain)} samples")
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    base_model.eval()

    # Wrap with softmax so attributions are on probability space
    softmax_model = SoftmaxModel(base_model)
    softmax_model.eval()
    ig = IntegratedGradients(softmax_model)
    baseline = torch.zeros(1, X_explain.shape[1], dtype=torch.float32).to(device)

    # Get predicted classes for target
    with torch.no_grad():
        input_tensor = torch.tensor(X_explain, dtype=torch.float32).to(device)
        preds = torch.argmax(base_model(input_tensor), dim=1)

    start = time.time()
    all_attrs = []
    batch_size = config.ig_internal_batch_size
    with _disable_cudnn_for_rnn(base_model):
        for i in range(0, len(X_explain), batch_size):
            batch = input_tensor[i : i + batch_size].requires_grad_(True)
            batch_preds = preds[i : i + batch_size]
            attrs = ig.attribute(
                batch,
                baselines=baseline.expand(len(batch), -1),
                target=batch_preds,
                n_steps=config.ig_n_steps,
                internal_batch_size=batch_size,
            )
            all_attrs.append(attrs.detach().cpu().numpy())
    elapsed = time.time() - start

    attributions = np.concatenate(all_attrs, axis=0)

    return ExplanationResult(
        attributions=attributions,
        method_name="IG",
        model_name="DNN",
        time_per_sample_ms=(elapsed / len(X_explain)) * 1000,
        total_time_s=elapsed,
    )


def explain_deeplift(
    model: torch.nn.Module,
    X_explain: np.ndarray,
    device: torch.device,
    config: ExplainerConfig,
) -> ExplanationResult:
    """Generate DeepLIFT explanations for DNN."""
    from captum.attr import DeepLift

    logger.info(f"  DeepLIFT on {len(X_explain)} samples")
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    base_model.eval()

    # Wrap with softmax so attributions are on probability space
    softmax_model = SoftmaxModel(base_model)
    softmax_model.eval()
    dl = DeepLift(softmax_model)
    baseline = torch.zeros(1, X_explain.shape[1], dtype=torch.float32).to(device)

    # Get predicted classes
    with torch.no_grad():
        input_tensor = torch.tensor(X_explain, dtype=torch.float32).to(device)
        preds = torch.argmax(base_model(input_tensor), dim=1)

    start = time.time()
    all_attrs = []
    batch_size = config.ig_internal_batch_size
    with warnings.catch_warnings(), _disable_cudnn_for_rnn(base_model):
        warnings.filterwarnings("ignore", message="Setting forward, backward hooks")
        for i in range(0, len(X_explain), batch_size):
            batch = input_tensor[i : i + batch_size].requires_grad_(True)
            batch_preds = preds[i : i + batch_size]
            attrs = dl.attribute(
                batch,
                baselines=baseline.expand(len(batch), -1),
                target=batch_preds,
            )
            all_attrs.append(attrs.detach().cpu().numpy())
    elapsed = time.time() - start

    attributions = np.concatenate(all_attrs, axis=0)

    return ExplanationResult(
        attributions=attributions,
        method_name="DeepLIFT",
        model_name="DNN",
        time_per_sample_ms=(elapsed / len(X_explain)) * 1000,
        total_time_s=elapsed,
    )


def generate_all_explanations(
    dnn_model: torch.nn.Module,
    rf_model,
    dnn_wrapper,
    rf_wrapper,
    dataset,
    device: torch.device,
    config: ExplainerConfig,
    xgb_model=None,
    xgb_wrapper=None,
) -> list[ExplanationResult]:
    """Generate explanations from all applicable XAI methods for both models."""
    n = min(config.num_explain_samples, len(dataset.X_test))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(dataset.X_test), size=n, replace=False)
    X_explain = dataset.X_test[indices]

    # Background data for SHAP
    bg_indices = rng.choice(len(dataset.X_train), size=config.shap_background_samples, replace=False)
    X_background = dataset.X_train[bg_indices]

    results = []

    # === DNN explanations ===
    if dnn_model is not None and dnn_wrapper is not None:
        logger.info("--- DNN Explanations ---")
        try:
            results.append(explain_shap_dnn(dnn_model, X_explain, X_background, device, config))
        except Exception as e:
            logger.error(f"SHAP DNN failed: {e}")

        try:
            results.append(
                explain_lime(
                    dnn_wrapper.predict_proba, X_explain, dataset.X_train,
                    dataset.feature_names, dataset.num_classes, "DNN", config,
                )
            )
        except Exception as e:
            logger.error(f"LIME DNN failed: {e}")

        try:
            results.append(explain_ig(dnn_model, X_explain, device, config))
        except Exception as e:
            logger.error(f"IG failed: {e}")

        try:
            results.append(explain_deeplift(dnn_model, X_explain, device, config))
        except Exception as e:
            logger.error(f"DeepLIFT failed: {e}")

    # === RF explanations ===
    if rf_model is not None and rf_wrapper is not None:
        logger.info("--- RF Explanations ---")
        try:
            results.append(explain_shap_rf(rf_model, X_explain, config))
        except Exception as e:
            logger.error(f"SHAP RF failed: {e}")

        try:
            results.append(
                explain_lime(
                    rf_wrapper.predict_proba, X_explain, dataset.X_train,
                    dataset.feature_names, dataset.num_classes, "RF", config,
                )
            )
        except Exception as e:
            logger.error(f"LIME RF failed: {e}")

    # === XGBoost explanations ===
    if xgb_model is not None and xgb_wrapper is not None:
        logger.info("--- XGBoost Explanations ---")
        try:
            _fix_xgb_base_score(xgb_model)
            results.append(explain_shap_rf(xgb_model, X_explain, config))
            # Rename model_name to XGB
            results[-1].model_name = "XGB"
        except Exception as e:
            logger.error(f"SHAP XGB failed: {e}")

        try:
            results.append(
                explain_lime(
                    xgb_wrapper.predict_proba, X_explain, dataset.X_train,
                    dataset.feature_names, dataset.num_classes, "XGB", config,
                )
            )
        except Exception as e:
            logger.error(f"LIME XGB failed: {e}")

    logger.info(f"Generated {len(results)} explanation sets")
    return results, indices


def generate_and_time_summary_plots(
    results: list[ExplanationResult],
    X_explain: np.ndarray,
    feature_names: list[str],
    output_dir,
) -> None:
    """Generate SHAP-style global summary plots and record the generation time.

    For each explanation result, creates a beeswarm/summary plot over the full
    batch of explanation samples and stores the elapsed wall-clock time in
    ``result.summary_plot_time_s``.
    """
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    plot_dir = Path(output_dir) / "summary_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        logger.info(
            f"  Generating global summary plot for "
            f"{result.model_name}/{result.method_name}"
        )
        start = time.time()

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            result.attributions,
            X_explain,
            feature_names=feature_names,
            show=False,
            plot_type="dot",
        )
        fname = f"{result.model_name}_{result.method_name}_summary.png"
        plt.savefig(plot_dir / fname, dpi=150, bbox_inches="tight")
        plt.close("all")

        elapsed = time.time() - start
        result.summary_plot_time_s = elapsed
        logger.info(
            f"    Summary plot saved ({elapsed:.2f}s): {fname}"
        )
