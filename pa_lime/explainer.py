"""Protocol-Aware LIME explainer for NIDS model auditing."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from pa_lime.fuzzer import DomainConstraintFuzzer
from pa_lime.result import ExplanationResult
from pa_lime.schemas import DatasetSchema


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
            ExplanationResult with coefficients, fidelity, feature names.
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

        # 3. Compute proximity weights
        scaler = StandardScaler()
        neighborhood_scaled = scaler.fit_transform(neighborhood)
        query_scaled = neighborhood_scaled[0:1]

        distances = pairwise_distances(
            neighborhood_scaled, query_scaled, metric="euclidean"
        ).flatten()

        if kernel_width is None:
            kernel_width = 0.75 * np.sqrt(d)

        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

        # 4. Fit weighted Ridge surrogate
        surrogate = Ridge(alpha=self.ridge_alpha)
        surrogate.fit(neighborhood_scaled, y_neighborhood, sample_weight=weights)

        r_squared = surrogate.score(
            neighborhood_scaled, y_neighborhood, sample_weight=weights
        )
        local_prediction = float(surrogate.predict(query_scaled)[0])

        return ExplanationResult(
            feature_names=list(self.schema.feature_names),
            coefficients=surrogate.coef_,
            intercept=float(surrogate.intercept_),
            r_squared=float(r_squared),
            predicted_class=predicted_class,
            local_prediction=local_prediction,
            num_samples=num_samples,
        )
