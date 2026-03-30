"""Vectorized constraint enforcement for protocol-aware neighborhood generation."""

from __future__ import annotations

import numpy as np

from pa_xai.core.schemas import DatasetSchema, TCP_PROTOCOL_INT


class ConstraintEnforcer:
    """Applies physical network protocol invariants to a neighborhood matrix.

    All operations are vectorized for O(1) complexity relative to constraint count.

    Args:
        schema: The DatasetSchema defining which constraints apply.
    """

    def __init__(self, schema: DatasetSchema) -> None:
        self.schema = schema

    def enforce(
        self,
        neighborhood: np.ndarray,
        protocol_value: float | None,
        protocol_encoding: str,
        tcp_label_value: int | float = TCP_PROTOCOL_INT,
    ) -> np.ndarray:
        """Apply all constraints in-place and return the neighborhood.

        Constraint order:
          1. Fix protocol column to original value
          2. Non-negativity clamping
          3. Hierarchical ordering (min <= mean <= max)
          4. Discrete rounding
          5. Protocol-gated TCP feature zeroing

        Args:
            neighborhood: Shape (N, D) matrix of perturbed samples.
            protocol_value: The original instance's protocol value, or None.
            protocol_encoding: 'integer' or 'string'.
            tcp_label_value: The numeric value representing TCP.

        Returns:
            The constrained neighborhood (modified in-place).
        """
        s = self.schema

        # 1. Fix protocol column
        if s.protocol_index is not None and protocol_value is not None:
            neighborhood[:, s.protocol_index] = protocol_value

        # 2. Non-negativity
        if s.non_negative_indices:
            idx = s.non_negative_indices
            neighborhood[:, idx] = np.maximum(0, neighborhood[:, idx])

        # 3. Hierarchical: mean = max(mean, min); max = max(max, mean)
        for max_i, mean_i, min_i in s.hierarchical_index_triples:
            neighborhood[:, mean_i] = np.maximum(
                neighborhood[:, mean_i], neighborhood[:, min_i]
            )
            neighborhood[:, max_i] = np.maximum(
                neighborhood[:, max_i], neighborhood[:, mean_i]
            )

        # 4. Discrete rounding
        if s.discrete_indices:
            idx = s.discrete_indices
            neighborhood[:, idx] = np.round(neighborhood[:, idx])

        # 5. Protocol-gated TCP feature zeroing
        if s.tcp_only_indices and protocol_value is not None:
            is_tcp = int(round(protocol_value)) == int(tcp_label_value)
            if not is_tcp:
                neighborhood[:, s.tcp_only_indices] = 0.0

        return neighborhood
