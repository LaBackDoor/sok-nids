"""Protocol-aware constraint projector for adversarial attacks.

Adapts pa_xai's ConstraintEnforcer + DatasetSchema into the
constraint_projector(X_adv, X_original, epsilon) -> X_adv interface
that attacks.py expects. Replaces feature_constraints.py.
"""

import logging

import numpy as np
import torch

from pa_xai import get_schema
from pa_xai.core.constraints import ConstraintEnforcer

logger = logging.getLogger(__name__)

DATASET_SCHEMA_MAP = {
    "nsl-kdd": "NSL-KDD",
    "cic-ids-2017": "CIC-IDS-2017",
    "unsw-nb15": "UNSW-NB15-CICFlowMeter",
    "cse-cic-ids2018": "CSE-CIC-IDS2018",
}


def make_pa_constraint_projector(dataset_name, scaler, device):
    """Create a constraint projection closure for use in FGSM/PGD attacks.

    Args:
        dataset_name: Dataset key (e.g., "nsl-kdd").
        scaler: Fitted sklearn MinMaxScaler used during preprocessing.
        device: Torch device for the returned tensor.

    Returns:
        callable(X_adv: Tensor, X_original: Tensor, epsilon: float) -> Tensor
        or None if no schema exists for the dataset.
    """
    if dataset_name not in DATASET_SCHEMA_MAP:
        logger.warning(
            f"No pa_xai schema for '{dataset_name}'. "
            "Skipping constrained attacks for this dataset."
        )
        return None

    schema = get_schema(DATASET_SCHEMA_MAP[dataset_name])
    enforcer = ConstraintEnforcer(schema)

    # Precompute scaler parameters as float64 for precision
    data_min = scaler.data_min_.astype(np.float64)
    data_range = (scaler.data_max_ - scaler.data_min_).astype(np.float64)
    # Avoid division by zero for constant features
    data_range_safe = data_range.copy()
    data_range_safe[data_range_safe < 1e-12] = 1.0

    protocol_idx = schema.protocol_index
    protocol_enc = schema.protocol_encoding

    def projector(
        X_adv: torch.Tensor,
        X_original: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        # 1. Convert to numpy
        adv_np = X_adv.detach().cpu().numpy().astype(np.float64)
        orig_np = X_original.detach().cpu().numpy().astype(np.float64)

        # 2. Inverse-scale from [0,1] to original feature space
        adv_orig = adv_np * data_range + data_min

        # 3. Group by protocol and enforce constraints per group
        if protocol_idx is not None:
            orig_orig = orig_np * data_range + data_min
            protocol_vals = np.round(orig_orig[:, protocol_idx]).astype(int)
            unique_protocols = np.unique(protocol_vals)

            for pval in unique_protocols:
                mask = protocol_vals == pval
                group = adv_orig[mask]
                enforcer.enforce(
                    group,
                    protocol_value=float(pval),
                    protocol_encoding=protocol_enc,
                )
                adv_orig[mask] = group
        else:
            enforcer.enforce(
                adv_orig,
                protocol_value=None,
                protocol_encoding=protocol_enc,
            )

        # 4. Re-scale back to [0,1]
        adv_scaled = (adv_orig - data_min) / data_range_safe
        adv_scaled = np.clip(adv_scaled, 0.0, 1.0)

        # 5. Re-apply L-inf epsilon-ball projection
        delta = np.clip(adv_scaled - orig_np, -epsilon, epsilon)
        adv_scaled = np.clip(orig_np + delta, 0.0, 1.0)

        # 6. Convert back to torch
        return torch.tensor(
            adv_scaled, dtype=torch.float32, device=device
        )

    return projector


def pa_constraint_spec_to_dict(dataset_name):
    """Serialize the pa_xai schema constraints to a JSON-compatible dict.

    Returns None if no schema exists for the dataset.
    """
    if dataset_name not in DATASET_SCHEMA_MAP:
        return None

    schema = get_schema(DATASET_SCHEMA_MAP[dataset_name])

    return {
        "dataset_name": dataset_name,
        "schema_name": DATASET_SCHEMA_MAP[dataset_name],
        "num_features": len(schema.feature_names),
        "protocol_feature": schema.protocol_feature,
        "protocol_encoding": schema.protocol_encoding,
        "num_non_negative": len(schema.non_negative_indices),
        "num_tcp_only": len(schema.tcp_only_indices),
        "num_discrete": len(schema.discrete_indices),
        "num_hierarchical": len(schema.hierarchical_index_triples),
        "num_bounded_range": len(schema.bounded_range_index_bounds),
        "num_cross_feature": len(schema.cross_feature_index_tuples),
        "num_std_range": len(schema.std_range_index_triples),
    }
