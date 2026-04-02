"""Vectorized constraint enforcement for protocol-aware neighborhood generation."""

from __future__ import annotations

import numpy as np

from pa_xai.core.schemas import (
    DatasetSchema,
    TCP_PROTOCOL_INT,
    UDP_PROTOCOL_INT,
    ICMP_PROTOCOL_INT,
)


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
        udp_label_value: int | float = UDP_PROTOCOL_INT,
        icmp_label_value: int | float = ICMP_PROTOCOL_INT,
    ) -> np.ndarray:
        """Apply all constraints in-place and return the neighborhood.

        Constraint order:
          1. Fix protocol column to original value
          2. Non-negativity clamping
          3. Hierarchical ordering (min <= mean <= max)
          4. Std <= range (std clamped to max - min)
          5. Cross-feature arithmetic (ratio, sum_ratio, square, equal)
          6. Bounded range clamping
          7. Discrete rounding
          8. Protocol-gated TCP feature zeroing
          9. UDP-specific zeroing
          10. ICMP-specific zeroing
          10b. Connection-only zeroing (zeroed when protocol IS ICMP)
          11. Duplicate feature equality

        Args:
            neighborhood: Shape (N, D) matrix of perturbed samples.
            protocol_value: The original instance's protocol value, or None.
            protocol_encoding: 'integer' or 'string'.
            tcp_label_value: The numeric value representing TCP.
            udp_label_value: The numeric value representing UDP.
            icmp_label_value: The numeric value representing ICMP.

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

        # 3. Hierarchical: sort so min <= mean <= max (unbiased)
        for max_i, mean_i, min_i in s.hierarchical_index_triples:
            triple = np.stack([
                neighborhood[:, min_i],
                neighborhood[:, mean_i],
                neighborhood[:, max_i],
            ], axis=1)
            triple.sort(axis=1)
            neighborhood[:, min_i] = triple[:, 0]
            neighborhood[:, mean_i] = triple[:, 1]
            neighborhood[:, max_i] = triple[:, 2]

        # 4. Std <= range
        for std_i, max_i, min_i in s.std_range_index_triples:
            range_val = neighborhood[:, max_i] - neighborhood[:, min_i]
            neighborhood[:, std_i] = np.maximum(0, np.minimum(neighborhood[:, std_i], range_val))

        # 5. Cross-feature arithmetic
        for derived_i, operand_indices, relation in s.cross_feature_index_tuples:
            if relation == "equal":
                neighborhood[:, derived_i] = neighborhood[:, operand_indices[0]]
            elif relation == "square":
                neighborhood[:, derived_i] = neighborhood[:, operand_indices[0]] ** 2
            elif relation == "ratio":
                num = neighborhood[:, operand_indices[0]]
                den = neighborhood[:, operand_indices[1]]
                with np.errstate(divide="ignore", invalid="ignore"):
                    neighborhood[:, derived_i] = np.where(np.abs(den) > 1e-10, num / den, 0.0)
            elif relation == "sum_ratio":
                a = neighborhood[:, operand_indices[0]]
                b = neighborhood[:, operand_indices[1]]
                den = neighborhood[:, operand_indices[2]]
                with np.errstate(divide="ignore", invalid="ignore"):
                    neighborhood[:, derived_i] = np.where(np.abs(den) > 1e-10, (a + b) / den, 0.0)

        # 6. Bounded range clamping
        for feat_i, lower, upper in s.bounded_range_index_bounds:
            neighborhood[:, feat_i] = np.clip(neighborhood[:, feat_i], lower, upper)

        # 7. Discrete rounding
        if s.discrete_indices:
            idx = s.discrete_indices
            neighborhood[:, idx] = np.round(neighborhood[:, idx])

        # 8. Protocol-gated TCP feature zeroing
        if s.tcp_only_indices and protocol_value is not None:
            is_tcp = int(round(protocol_value)) == int(tcp_label_value)
            if not is_tcp:
                neighborhood[:, s.tcp_only_indices] = 0.0

        # 9. UDP-specific zeroing
        if s.udp_only_indices and protocol_value is not None:
            is_udp = int(round(protocol_value)) == int(udp_label_value)
            if not is_udp:
                neighborhood[:, s.udp_only_indices] = 0.0

        # 10. ICMP-specific zeroing
        if s.icmp_only_indices and protocol_value is not None:
            is_icmp = int(round(protocol_value)) == int(icmp_label_value)
            if not is_icmp:
                neighborhood[:, s.icmp_only_indices] = 0.0

        # 10b. Connection-only zeroing (zeroed when protocol IS ICMP)
        if s.connection_only_indices and protocol_value is not None:
            is_icmp = int(round(protocol_value)) == int(icmp_label_value)
            if is_icmp:
                neighborhood[:, s.connection_only_indices] = 0.0

        # 11. Duplicate feature equality
        for idx_a, idx_b in s.duplicate_index_pairs:
            neighborhood[:, idx_b] = neighborhood[:, idx_a]

        return neighborhood
