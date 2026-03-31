"""PA-LIME: Protocol-Aware LIME for NIDS model auditing.

This package is a backwards-compatibility shim. All functionality
has moved to pa_xai. Existing imports continue to work.
"""

from pa_xai.core.constraints import ConstraintEnforcer
from pa_xai.core.metrics import sparsity
from pa_xai.core.result import ExplanationResult
from pa_xai.core.schemas import (
    BUILTIN_SCHEMAS,
    CIC_IDS_2017,
    CSE_CIC_IDS2018,
    DatasetSchema,
    HierarchicalConstraint,
    NSL_KDD,
    UNSW_NB15_CIC,
    UNSW_NB15_NATIVE,
    get_schema,
)
from pa_xai.lime import ProtocolAwareLIME, fidelity, semantic_robustness

__all__ = [
    "ProtocolAwareLIME",
    "ExplanationResult",
    "DatasetSchema",
    "HierarchicalConstraint",
    "CIC_IDS_2017",
    "CSE_CIC_IDS2018",
    "NSL_KDD",
    "UNSW_NB15_NATIVE",
    "UNSW_NB15_CIC",
    "BUILTIN_SCHEMAS",
    "get_schema",
    "semantic_robustness",
    "fidelity",
    "sparsity",
]
