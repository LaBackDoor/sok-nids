"""Protocol-Aware LIME explainer for NIDS model auditing."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from pa_xai.lime.fuzzer import DomainConstraintFuzzer
from pa_xai.core.result import ExplanationResult
from pa_xai.core.schemas import DatasetSchema


class ProtocolAwareLIME:
    """Protocol-Aware Local Interpretable Model-Agnostic Explanations.

    Args:
        schema: DatasetSchema with feature metadata and protocol constraints.
        tcp_label_value: For string-encoded protocol columns, the label-encoded
            integer representing TCP.
        ridge_alpha: Regularization strength for the Ridge surrogate.
    """

    def __init__(
        self,
        schema: DatasetSchema,
        tcp_label_value: int | float | None = None,
        ridge_alpha: float = 1.0,
    ) -> None:
        self.schema = schema
        self.fuzzer = DomainConstraintFuzzer(schema, tcp_label_value=tcp_label_value)
        self.ridge_alpha = ridge_alpha

    def explain_instance(
        self,
        x_row: np.ndarray,
        predict_fn,
        num_samples: int = 5000,
        sigma: float | np.ndarray = 0.1,
        kernel_width: float | None = None,
        class_to_explain: int | None = None,
    ) -> ExplanationResult:
        """Generate a local explanation for a single instance.

        Args:
            x_row: 1D array of shape (D,).
            predict_fn: Callable (N, D) -> (N,) or (N, C).
            num_samples: Number of neighborhood samples.
            sigma: Perturbation scale. Scalar or per-feature array.
            kernel_width: Exponential kernel width. Default: 0.75 * sqrt(D).
            class_to_explain: For multi-class, which column to explain.

        Returns:
            ExplanationResult with attributions, fidelity, feature names.
        """
        d = len(x_row)

        # 1. Generate constrained neighborhood
        neighborhood = self.fuzzer.generate(x_row, num_samples, sigma)

        # 2. Get black-box predictions
        raw_preds = predict_fn(neighborhood)

        # Handle multi-class
        predicted_class = None
        if raw_preds.ndim == 2:
            if class_to_explain is None:
                class_to_explain = int(np.argmax(raw_preds[0]))
            predicted_class = class_to_explain
            y_neighborhood = raw_preds[:, class_to_explain]
        else:
            y_neighborhood = raw_preds

        # 3. Compute proximity weights (euclidean distance in original space)
        query = neighborhood[0:1]
        distances = pairwise_distances(
            neighborhood, query, metric="euclidean"
        ).flatten()

        if kernel_width is None:
            kernel_width = 0.75 * np.sqrt(d)

        weights = np.exp(-(distances ** 2) / (2 * kernel_width ** 2))

        # 4. Fit weighted Ridge surrogate
        surrogate = Ridge(alpha=self.ridge_alpha)
        surrogate.fit(neighborhood, y_neighborhood, sample_weight=weights)

        r_squared = surrogate.score(
            neighborhood, y_neighborhood, sample_weight=weights
        )
        local_prediction = float(surrogate.predict(query)[0])

        return ExplanationResult(
            feature_names=list(self.schema.feature_names),
            attributions=surrogate.coef_,
            method="pa_lime",
            predicted_class=predicted_class,
            num_samples=num_samples,
            r_squared=float(r_squared),
            intercept=float(surrogate.intercept_),
            local_prediction=local_prediction,
        )

    def explain_pcap(
        self,
        pcap_path: str,
        predict_fn,
        feature_fn,
        feature_names: list[str],
        mode: str = "packet",
        num_samples: int = 5000,
        sigma: float = 0.1,
        kernel_width: float | None = None,
        class_to_explain: int | None = None,
        max_retries: int = 10,
    ) -> ExplanationResult:
        """Generate a local explanation from a PCAP file.

        Args:
            pcap_path: Path to the PCAP file.
            predict_fn: Callable over list[ParsedPacket|ParsedFlow] -> np.ndarray.
            feature_fn: Callable: ParsedPacket|ParsedFlow -> np.ndarray (1D).
            feature_names: Names for features returned by feature_fn.
            mode: "packet" or "flow".
        """
        from pa_xai.pcap.pipeline import PcapPipeline

        pipeline = PcapPipeline(max_retries=max_retries)
        neighborhood_samples = pipeline.generate_neighborhood(pcap_path, num_samples, sigma, mode=mode)

        features = np.array([feature_fn(s) for s in neighborhood_samples])
        d = features.shape[1]

        raw_preds = predict_fn(neighborhood_samples)

        predicted_class = None
        if raw_preds.ndim == 2:
            if class_to_explain is None:
                class_to_explain = int(np.argmax(raw_preds[0]))
            predicted_class = class_to_explain
            y_neighborhood = raw_preds[:, class_to_explain]
        else:
            y_neighborhood = raw_preds

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        query_scaled = features_scaled[0:1]

        distances = pairwise_distances(features_scaled, query_scaled, metric="euclidean").flatten()

        if kernel_width is None:
            kernel_width = 0.75 * np.sqrt(d)

        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

        surrogate = Ridge(alpha=self.ridge_alpha)
        surrogate.fit(features_scaled, y_neighborhood, sample_weight=weights)

        r_squared = surrogate.score(features_scaled, y_neighborhood, sample_weight=weights)
        local_prediction = float(surrogate.predict(query_scaled)[0])

        return ExplanationResult(
            feature_names=list(feature_names),
            attributions=surrogate.coef_,
            method="pa_lime_pcap",
            predicted_class=predicted_class,
            num_samples=num_samples,
            r_squared=float(r_squared),
            intercept=float(surrogate.intercept_),
            local_prediction=local_prediction,
        )
