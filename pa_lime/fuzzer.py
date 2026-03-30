"""Domain-Constraint Tabular Fuzzer for protocol-aware neighborhood generation."""

from __future__ import annotations

import numpy as np

from pa_lime.constraints import ConstraintEnforcer
from pa_lime.schemas import (
    DatasetSchema,
    TCP_PROTOCOL_INT,
    detect_protocol_encoding,
)


class DomainConstraintFuzzer:
    """Generates constrained neighborhoods for NIDS flow data.

    Args:
        schema: The DatasetSchema defining feature metadata and constraints.
        tcp_label_value: For string-encoded protocol columns, the integer
            label representing TCP. Ignored for integer-encoded schemas.
    """

    def __init__(
        self,
        schema: DatasetSchema,
        tcp_label_value: int | float | None = None,
    ) -> None:
        self.schema = schema
        self.enforcer = ConstraintEnforcer(schema)
        self.tcp_label_value = tcp_label_value

    def generate(
        self,
        x_row: np.ndarray,
        num_samples: int,
        sigma: float | np.ndarray,
    ) -> np.ndarray:
        """Generate a constrained neighborhood around a single instance.

        Args:
            x_row: 1D array of shape (D,).
            num_samples: Number of synthetic samples.
            sigma: Scalar or per-feature array of shape (D,).

        Returns:
            2D array of shape (num_samples, D) with first row equal to x_row.
        """
        d = len(x_row)
        noise = np.random.normal(0, sigma, (num_samples, d))
        neighborhood = x_row + noise

        # Resolve protocol encoding
        encoding = self.schema.protocol_encoding
        protocol_value = None
        tcp_val = TCP_PROTOCOL_INT

        if self.schema.protocol_index is not None:
            protocol_value = x_row[self.schema.protocol_index]
            if encoding == "auto":
                encoding = detect_protocol_encoding(
                    x_row, self.schema.protocol_feature, self.schema.feature_names
                )
            if encoding == "string":
                tcp_val = self.tcp_label_value if self.tcp_label_value is not None else TCP_PROTOCOL_INT

        self.enforcer.enforce(
            neighborhood,
            protocol_value=protocol_value,
            protocol_encoding=encoding,
            tcp_label_value=tcp_val,
        )

        # Pin first row to exact original, then re-apply TCP-only zeroing so
        # the original row also satisfies protocol constraints.
        neighborhood[0, :] = x_row
        if self.schema.tcp_only_indices and protocol_value is not None:
            is_tcp = int(round(float(protocol_value))) == int(tcp_val)
            if not is_tcp:
                neighborhood[0, self.schema.tcp_only_indices] = 0.0
        return neighborhood
