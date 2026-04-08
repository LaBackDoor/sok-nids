"""Protocol-Aware SHAP explainer for NIDS (multi-backend)."""

from __future__ import annotations

import copy
import json
import warnings

import numpy as np
import torch
import torch.nn as nn

from pa_xai.core.constraints import ConstraintEnforcer
from pa_xai.core.result import ExplanationResult
from pa_xai.core.schemas import (
    DatasetSchema,
    TCP_PROTOCOL_INT,
    detect_protocol_encoding,
)


def _subsample_background(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_background: int,
    benign_label: int = 0,
) -> np.ndarray:
    """Subsample benign training data for SHAP background.

    Filters to benign samples so SHAP values answer "what makes this
    flow different from normal traffic?"  No protocol filtering — the
    background includes all protocols so SHAP can attribute to
    protocol differences.
    """
    candidates = X_train[y_train == benign_label]
    if len(candidates) == 0:
        raise ValueError("No benign samples found in training data.")
    if len(candidates) <= n_background:
        return candidates.copy()
    rng = np.random.RandomState(42)
    idx = rng.choice(len(candidates), size=n_background, replace=False)
    return candidates[idx]


def _has_rnn_modules(model: nn.Module) -> bool:
    return any(isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)) for m in model.modules())


def _extract_class_attributions(shap_values, target: int, n_features: int) -> np.ndarray:
    # Handle shap.Explanation objects (newer SHAP API)
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values

    if isinstance(shap_values, list):
        return np.asarray(shap_values[target]).flatten()[:n_features]
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            return shap_values[0, :, target]
        elif shap_values.ndim == 2:
            return shap_values.flatten()[:n_features]
        else:
            return shap_values.flatten()[:n_features]
    else:
        return np.asarray(shap_values).flatten()[:n_features]


class _ConstrainedKernelExplainer:
    def __init__(self, predict_fn, background, schema, enforcer,
                 protocol_value, protocol_encoding, tcp_label_value):
        import shap
        self._schema = schema
        self._enforcer = enforcer
        self._protocol_value = protocol_value
        self._protocol_encoding = protocol_encoding
        self._tcp_label_value = tcp_label_value

        original_predict = predict_fn
        def constrained_predict(X):
            X_clamped = X.copy()
            if len(X_clamped.shape) == 1:
                X_clamped = X_clamped.reshape(1, -1)
            self._enforcer.enforce(
                X_clamped, self._protocol_value,
                self._protocol_encoding, self._tcp_label_value,
            )
            return original_predict(X_clamped)

        self._explainer = shap.KernelExplainer(constrained_predict, background)

    def shap_values(self, X, **kwargs):
        return self._explainer.shap_values(X, **kwargs)

    @property
    def expected_value(self):
        return self._explainer.expected_value


class ProtocolAwareSHAP:
    """SHAP explainer for NIDS (multi-backend: kernel, deep, tree).

    Background is a subsample of benign training data across all
    protocols.  No protocol filtering — so SHAP can attribute to
    protocol differences.  Benign-only so SHAP values answer
    "what makes this flow different from normal traffic?"
    """

    def __init__(
        self,
        schema: DatasetSchema,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        benign_label: int = 0,
        backend: str = "kernel",
        n_background: int = 100,
        tcp_label_value: int = TCP_PROTOCOL_INT,
    ) -> None:
        if backend not in ("kernel", "deep", "tree"):
            raise ValueError(f"backend must be 'kernel', 'deep', or 'tree', got {backend!r}")
        self.schema = schema
        self.model = model
        self.backend = backend
        self.tcp_label_value = tcp_label_value
        self.enforcer = ConstraintEnforcer(schema)
        self._background = _subsample_background(X_train, y_train, n_background, benign_label)
        self._tree_explainer: object | None = None
        # Patch SHAP's XGBTreeModelLoader to handle vector base_score
        if backend == "tree" and hasattr(model, 'get_booster'):
            self._patch_shap_xgb_loader()

    @staticmethod
    def _patch_shap_xgb_loader():
        """Monkey-patch SHAP's XGBTreeModelLoader to handle vector base_score.

        XGBoost 2.x+ stores base_score as a per-class vector string.
        SHAP's TreeExplainer reads UBJ and calls float(base_score), which
        fails.  We patch the loader to normalise the vector to a scalar.
        """
        import io as _io
        import logging
        import scipy.special
        from shap.explainers._tree import (
            XGBTreeModelLoader,
            _check_xgboost_version,
            decode_ubjson_buffer,
        )

        if getattr(XGBTreeModelLoader, '_base_score_patched', False):
            return

        log = logging.getLogger(__name__)

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
                log.debug("Patched SHAP XGB base_score '%s' → '%s'", bs, scalar)
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

    def _resolve_protocol_params(self, x_row):
        encoding = self.schema.protocol_encoding
        protocol_value = None
        tcp_val = self.tcp_label_value
        if self.schema.protocol_index is not None:
            protocol_value = x_row[self.schema.protocol_index]
            if encoding == "auto":
                encoding = detect_protocol_encoding(
                    x_row, self.schema.protocol_feature, self.schema.feature_names
                )
        return protocol_value, encoding, tcp_val

    def _predict_target(self, x_row):
        if self.backend == "tree":
            X = x_row.reshape(1, -1)
            if hasattr(self.model, 'get_booster'):
                import cupy as cp
                X = cp.asarray(X)
            return int(self.model.predict(X)[0])
        elif self.backend == "kernel":
            preds = self.model(x_row.reshape(1, -1))
            return int(np.argmax(preds[0]))
        else:
            with torch.no_grad():
                t = torch.tensor(x_row, dtype=torch.float32).unsqueeze(0)
                device = next(self.model.parameters()).device
                logits = self.model(t.to(device))
                return int(torch.argmax(logits, dim=1).item())

    def _explain_kernel(self, x_row, target, nsamples):
        protocol_value, encoding, tcp_val = self._resolve_protocol_params(x_row)
        explainer = _ConstrainedKernelExplainer(
            self.model, self._background, self.schema, self.enforcer,
            protocol_value, encoding, tcp_val,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            shap_values = explainer.shap_values(x_row.reshape(1, -1), nsamples=nsamples)
        n_features = len(x_row)
        attributions = _extract_class_attributions(shap_values, target, n_features)
        ev = explainer.expected_value
        expected_value = float(ev[target]) if isinstance(ev, (list, np.ndarray)) else float(ev)
        return attributions, expected_value

    def _explain_deep(self, x_row, target):
        import shap
        device = next(self.model.parameters()).device
        model_clone = copy.deepcopy(self.model).to(device)
        bg_tensor = torch.tensor(self._background, dtype=torch.float32).to(device)
        base_model = model_clone.module if isinstance(model_clone, torch.nn.DataParallel) else model_clone
        base_model.eval()
        use_gradient = _has_rnn_modules(base_model)
        # Disable cuDNN for RNN models — fused kernels don't support backward in eval mode
        prev_cudnn = torch.backends.cudnn.enabled
        if use_gradient:
            torch.backends.cudnn.enabled = False
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if use_gradient:
                    explainer = shap.GradientExplainer(base_model, bg_tensor)
                else:
                    explainer = shap.DeepExplainer(base_model, bg_tensor)
                x_tensor = torch.tensor(x_row, dtype=torch.float32).unsqueeze(0).to(device)
                if use_gradient:
                    shap_values = explainer.shap_values(x_tensor)
                else:
                    shap_values = explainer.shap_values(x_tensor, check_additivity=False)
        finally:
            if use_gradient:
                torch.backends.cudnn.enabled = prev_cudnn
        n_features = len(x_row)
        attributions = _extract_class_attributions(shap_values, target, n_features)
        if use_gradient:
            # GradientExplainer does not set expected_value — compute manually
            with torch.no_grad():
                ev_tensor = base_model(bg_tensor).mean(0).cpu().numpy()
            ev = ev_tensor
        else:
            ev = explainer.expected_value
        expected_value = float(ev[target]) if isinstance(ev, (list, np.ndarray)) else float(ev)
        return attributions, expected_value

    def _get_tree_explainer(self):
        """Get or create a cached TreeExplainer."""
        import shap
        if self._tree_explainer is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self._tree_explainer = shap.TreeExplainer(
                    self.model, data=self._background,
                    feature_perturbation="interventional",
                )
        return self._tree_explainer

    def _explain_tree(self, x_row, target):
        explainer = self._get_tree_explainer()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            shap_values = explainer.shap_values(x_row.reshape(1, -1), check_additivity=False)
        n_features = len(x_row)
        attributions = _extract_class_attributions(shap_values, target, n_features)
        ev = explainer.expected_value
        expected_value = float(ev[target]) if isinstance(ev, (list, np.ndarray)) else float(ev)
        return attributions, expected_value

    def explain_instance(
        self,
        x_row: np.ndarray,
        target: int | None = None,
        nsamples: int | str = "auto",
    ) -> ExplanationResult:
        if target is None:
            target = self._predict_target(x_row)

        if self.backend == "kernel":
            attributions, expected_value = self._explain_kernel(x_row, target, nsamples)
        elif self.backend == "deep":
            attributions, expected_value = self._explain_deep(x_row, target)
        elif self.backend == "tree":
            attributions, expected_value = self._explain_tree(x_row, target)

        return ExplanationResult(
            feature_names=list(self.schema.feature_names),
            attributions=attributions,
            method="pa_shap",
            predicted_class=target,
            num_samples=nsamples if isinstance(nsamples, int) else None,
            expected_value=expected_value,
        )

    # ------------------------------------------------------------------
    # Batch methods — process all samples in a single vectorised call
    # ------------------------------------------------------------------

    def explain_batch_deep(
        self,
        X_explain: np.ndarray,
    ) -> np.ndarray:
        """Batch Deep/GradientExplainer: one explainer, one shap_values call.

        Returns attributions array of shape (N, D) with each row containing
        the SHAP values for the predicted class of that sample.
        """
        import shap

        device = next(self.model.parameters()).device
        model_clone = copy.deepcopy(self.model).to(device)
        bg_tensor = torch.tensor(self._background, dtype=torch.float32).to(device)
        base_model = (
            model_clone.module
            if isinstance(model_clone, torch.nn.DataParallel)
            else model_clone
        )
        base_model.eval()
        use_gradient = _has_rnn_modules(base_model)

        prev_cudnn = torch.backends.cudnn.enabled
        if use_gradient:
            torch.backends.cudnn.enabled = False
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if use_gradient:
                    explainer = shap.GradientExplainer(base_model, bg_tensor)
                else:
                    explainer = shap.DeepExplainer(base_model, bg_tensor)

                explain_tensor = torch.tensor(
                    X_explain, dtype=torch.float32,
                ).to(device)

                if use_gradient:
                    shap_values = explainer.shap_values(explain_tensor)
                else:
                    shap_values = explainer.shap_values(
                        explain_tensor, check_additivity=False,
                    )
        finally:
            if use_gradient:
                torch.backends.cudnn.enabled = prev_cudnn

        # Predict classes for all samples
        with torch.no_grad():
            explain_t = torch.tensor(
                X_explain, dtype=torch.float32,
            ).to(device)
            preds = torch.argmax(base_model(explain_t), dim=1).cpu().numpy()

        # Extract per-sample attributions for predicted class (vectorised)
        n = len(X_explain)
        # Handle shap.Explanation objects
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values
        if isinstance(shap_values, list):
            stacked = np.stack(shap_values, axis=0)  # (classes, N, D)
            attributions = stacked[preds, np.arange(n)]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            attributions = shap_values[np.arange(n), :, preds]
        else:
            attributions = np.asarray(shap_values)

        return attributions

    def explain_batch_tree(
        self,
        X_explain: np.ndarray,
    ) -> np.ndarray:
        """Batch TreeExplainer: one shap_values call for all samples.

        Returns attributions array of shape (N, D) with each row containing
        the SHAP values for the predicted class of that sample.
        """
        explainer = self._get_tree_explainer()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            shap_values = explainer.shap_values(X_explain, check_additivity=False)

        # Predict classes
        if hasattr(self.model, "get_booster"):
            import xgboost as xgb
            try:
                import cupy as cp
                dm = xgb.DMatrix(cp.asarray(X_explain))
            except ImportError:
                dm = xgb.DMatrix(X_explain)
            raw = self.model.get_booster().predict(dm)
            if raw.ndim == 1:
                preds = (raw > 0.5).astype(np.intp)
            else:
                preds = np.argmax(raw, axis=1).astype(np.intp)
        else:
            preds = self.model.predict(X_explain)

        # Extract per-sample attributions for predicted class (vectorised)
        n = len(X_explain)
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values
        if isinstance(shap_values, list):
            stacked = np.stack(shap_values, axis=0)  # (classes, N, D)
            attributions = stacked[preds, np.arange(n)]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            attributions = shap_values[np.arange(n), :, preds]
        else:
            attributions = np.asarray(shap_values)

        return attributions

    def explain_pcap(
        self,
        pcap_path: str,
        predict_fn,
        feature_fn,
        feature_names: list[str],
        mode: str = "packet",
        target: int | None = None,
        nsamples: int | str = "auto",
    ) -> ExplanationResult:
        """Generate a SHAP explanation from a PCAP file."""
        from pa_xai.pcap.pipeline import PcapPipeline

        pipeline = PcapPipeline()
        if mode == "packet":
            packets = pipeline.parser.parse_packets(pcap_path)
            if not packets:
                raise ValueError("No packets found in PCAP")
            x_row = feature_fn(packets[0])
        else:
            flows = pipeline.parser.parse_flows(pcap_path)
            if not flows:
                raise ValueError("No flows found in PCAP")
            x_row = feature_fn(flows[0])

        return self.explain_instance(x_row, target=target, nsamples=nsamples)
