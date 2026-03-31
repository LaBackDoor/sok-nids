"""Nearest Benign Prototype baseline selection for gradient-based explainers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from pa_xai.core.schemas import DatasetSchema


def get_protocol_valid_baseline(
    x_row: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    schema: DatasetSchema,
    benign_label: int = 0,
    top_k: int = 1,
    strategy: str = "nearest",
) -> np.ndarray:
    """Select a protocol-valid benign baseline for reference-based explainers.

    3-step selection:
      1. Filter X_train to benign samples matching x_row's protocol.
      2. Compute L2 distance from x_row to each candidate.
      3. Return nearest or median of top-k.

    Raises:
        ValueError: If no benign samples match the target protocol.

    Note:
        L2 distance is computed in the raw feature space. For meaningful
        nearest-neighbor selection, input data should be pre-scaled (e.g.,
        MinMaxScaler or StandardScaler) so all features contribute equally
        to the distance computation.
    """
    benign_mask = y_train == benign_label

    if schema.protocol_index is not None:
        protocol_value = x_row[schema.protocol_index]
        proto_mask = X_train[:, schema.protocol_index] == protocol_value
        mask = benign_mask & proto_mask
    else:
        mask = benign_mask

    candidates = X_train[mask]
    if len(candidates) == 0:
        # Fall back to all benign samples regardless of protocol
        import warnings
        warnings.warn(
            "No benign samples with matching protocol found. "
            "Falling back to nearest benign sample across all protocols.",
            stacklevel=2,
        )
        candidates = X_train[benign_mask]
        if len(candidates) == 0:
            raise ValueError(
                "No benign samples found in training data at all."
            )

    if len(candidates) < 10:
        import warnings
        warnings.warn(
            f"Only {len(candidates)} benign samples match the target protocol. "
            f"Baseline selection may be unreliable.",
            stacklevel=2,
        )

    distances = cdist(x_row.reshape(1, -1), candidates, metric="euclidean").flatten()
    sorted_idx = np.argsort(distances)

    if strategy == "nearest" or top_k <= 1:
        return candidates[sorted_idx[0]].copy()
    elif strategy == "median_k":
        k = min(top_k, len(candidates))
        top_k_samples = candidates[sorted_idx[:k]]
        baseline = np.median(top_k_samples, axis=0)
        # Enforce constraints on the synthetic median vector
        from pa_xai.core.constraints import ConstraintEnforcer
        enforcer = ConstraintEnforcer(schema)
        protocol_value = None
        if schema.protocol_index is not None:
            protocol_value = baseline[schema.protocol_index]
        enforcer.enforce(
            baseline.reshape(1, -1),
            protocol_value=protocol_value,
            protocol_encoding=schema.protocol_encoding,
        )
        return baseline
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'nearest' or 'median_k'.")
