"""Conditioned Marginal Sampling fuzzer for LIME neighborhood generation.

Generates domain-valid NIDS neighborhoods:

1. **Protocol**: sampled from training distribution (or held fixed if
   no X_train).  This allows LIME to attribute to protocol changes.
2. **Discrete features**: sampled from training data filtered by the
   sampled protocol value for that row.
3. **Continuous features**: Gaussian perturbation around the input.
4. **Enforcer**: applied per-protocol-group to repair cross-feature
   constraints (hierarchical, arithmetic, bounded ranges, etc.).

Every row the model sees is a physically valid packet.
"""

from __future__ import annotations

import numpy as np

from pa_xai.core.constraints import ConstraintEnforcer
from pa_xai.core.schemas import (
    DatasetSchema,
    TCP_PROTOCOL_INT,
    UDP_PROTOCOL_INT,
    ICMP_PROTOCOL_INT,
    detect_protocol_encoding,
)


class DomainConstraintFuzzer:
    """Generates constrained neighborhoods for NIDS flow data.

    Uses conditioned marginal sampling: protocol is sampled from
    the training distribution, discrete features are sampled from
    their protocol-filtered training distribution, continuous
    features get Gaussian noise, and the ConstraintEnforcer repairs
    cross-feature violations per protocol group.

    Args:
        schema: DatasetSchema defining feature metadata and constraints.
        X_train: Training data for protocol and discrete feature sampling.
            If None, all discrete features (including protocol) are held
            fixed at the input value.
        tcp_label_value: For string-encoded protocol columns, the integer
            label representing TCP.
    """

    def __init__(
        self,
        schema: DatasetSchema,
        X_train: np.ndarray | None = None,
        tcp_label_value: int | float | None = None,
        udp_label_value: int | float | None = None,
        icmp_label_value: int | float | None = None,
    ) -> None:
        self.schema = schema
        self.enforcer = ConstraintEnforcer(schema)
        self.tcp_label_value = tcp_label_value
        self.udp_label_value = udp_label_value
        self.icmp_label_value = icmp_label_value

        # Precompute distributions from training data
        self._protocol_values: np.ndarray | None = None
        self._discrete_distributions: dict[int, dict[int, np.ndarray]] = {}
        if X_train is not None:
            self._build_distributions(X_train)

    def _build_distributions(self, X_train: np.ndarray) -> None:
        """Build empirical distributions from training data.

        - Protocol distribution: all protocol values in X_train (for sampling).
        - Per-protocol discrete distributions: for each protocol, the
          observed values of each discrete feature.
        """
        s = self.schema

        # Protocol distribution
        if s.protocol_index is not None:
            self._protocol_values = X_train[:, s.protocol_index].copy()
            protocols = np.unique(self._protocol_values)
        else:
            protocols = np.array([None])

        # Per-protocol discrete feature distributions
        if not s.discrete_indices:
            return

        for proto in protocols:
            proto_key = int(proto) if proto is not None else -1
            if proto is not None and s.protocol_index is not None:
                mask = X_train[:, s.protocol_index] == proto
                subset = X_train[mask]
            else:
                subset = X_train

            if len(subset) == 0:
                continue

            feat_dists: dict[int, np.ndarray] = {}
            for idx in s.discrete_indices:
                if idx == s.protocol_index:
                    continue
                feat_dists[idx] = subset[:, idx].copy()
            self._discrete_distributions[proto_key] = feat_dists

    def _resolve_encoding(self, x_row: np.ndarray):
        """Resolve protocol encoding and label values."""
        encoding = self.schema.protocol_encoding
        tcp_val = TCP_PROTOCOL_INT
        udp_val = UDP_PROTOCOL_INT
        icmp_val = ICMP_PROTOCOL_INT

        if self.schema.protocol_index is not None:
            if encoding == "auto":
                encoding = detect_protocol_encoding(
                    x_row, self.schema.protocol_feature, self.schema.feature_names
                )
            if encoding == "string":
                tcp_val = self.tcp_label_value if self.tcp_label_value is not None else TCP_PROTOCOL_INT
                udp_val = self.udp_label_value if self.udp_label_value is not None else UDP_PROTOCOL_INT
                icmp_val = self.icmp_label_value if self.icmp_label_value is not None else ICMP_PROTOCOL_INT

        return encoding, tcp_val, udp_val, icmp_val

    def generate(
        self,
        x_row: np.ndarray,
        num_samples: int,
        sigma: float | np.ndarray,
    ) -> np.ndarray:
        """Generate a constrained neighborhood around a single instance.

        Generation strategy:
          1. Sample protocol from training distribution (categorical).
          2. For each row, sample discrete features from that row's
             protocol-filtered training distribution.
          3. Gaussian noise on continuous features.
          4. Group by protocol, run enforcer per group.
          5. Row 0 is the original input (enforcer-cleaned).

        Args:
            x_row: 1D array of shape (D,).
            num_samples: Number of synthetic samples.
            sigma: Scalar or per-feature array of shape (D,).

        Returns:
            2D array of shape (num_samples, D) with row 0 being x_row
            after constraint enforcement.
        """
        s = self.schema
        d = len(x_row)

        # Start with copies of x_row
        neighborhood = np.tile(x_row, (num_samples, 1)).astype(np.float64)

        # Identify continuous columns (everything not discrete, not protocol)
        discrete_set = set(s.discrete_indices)
        if s.protocol_index is not None:
            discrete_set.add(s.protocol_index)
        cont_indices = [i for i in range(d) if i not in discrete_set]

        # --- Continuous features: Gaussian perturbation ---
        if cont_indices:
            if np.isscalar(sigma):
                noise = np.random.normal(0, sigma, (num_samples, len(cont_indices)))
            else:
                sigma_arr = np.asarray(sigma)
                noise = np.random.normal(0, sigma_arr[cont_indices], (num_samples, len(cont_indices)))
            neighborhood[1:, cont_indices] += noise[1:]

        # --- Protocol: sample from training distribution ---
        if s.protocol_index is not None and self._protocol_values is not None:
            sampled_protocols = np.random.choice(
                self._protocol_values, size=num_samples - 1,
            )
            neighborhood[1:, s.protocol_index] = sampled_protocols

        # --- Discrete features: sample per-row's protocol ---
        if self._discrete_distributions:
            for i in range(1, num_samples):
                proto_key = int(neighborhood[i, s.protocol_index]) if s.protocol_index is not None else -1
                if proto_key in self._discrete_distributions:
                    for idx, values in self._discrete_distributions[proto_key].items():
                        neighborhood[i, idx] = np.random.choice(values)
            # else: discrete features stay at x_row values

        # --- Pin row 0 to original ---
        neighborhood[0, :] = x_row

        # --- Enforce constraints per protocol group ---
        encoding, tcp_val, udp_val, icmp_val = self._resolve_encoding(x_row)

        if s.protocol_index is not None:
            unique_protos = np.unique(neighborhood[:, s.protocol_index])
            for proto in unique_protos:
                mask = neighborhood[:, s.protocol_index] == proto
                group = neighborhood[mask]
                self.enforcer.enforce(
                    group,
                    protocol_value=proto,
                    protocol_encoding=encoding,
                    tcp_label_value=tcp_val,
                    udp_label_value=udp_val,
                    icmp_label_value=icmp_val,
                )
                neighborhood[mask] = group
        else:
            self.enforcer.enforce(
                neighborhood,
                protocol_value=None,
                protocol_encoding=encoding,
                tcp_label_value=tcp_val,
                udp_label_value=udp_val,
                icmp_label_value=icmp_val,
            )

        return neighborhood
