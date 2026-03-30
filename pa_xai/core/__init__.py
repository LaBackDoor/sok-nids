"""Core infrastructure for Protocol-Aware XAI."""

from pa_xai.core.baseline import get_protocol_valid_baseline
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
    TCP_PROTOCOL_INT,
    UNSW_NB15_CIC,
    UNSW_NB15_NATIVE,
    detect_protocol_encoding,
    get_schema,
)

__all__ = [
    "ConstraintEnforcer",
    "DatasetSchema",
    "ExplanationResult",
    "HierarchicalConstraint",
    "CIC_IDS_2017",
    "CSE_CIC_IDS2018",
    "NSL_KDD",
    "UNSW_NB15_NATIVE",
    "UNSW_NB15_CIC",
    "BUILTIN_SCHEMAS",
    "get_schema",
    "detect_protocol_encoding",
    "get_protocol_valid_baseline",
    "sparsity",
    "TCP_PROTOCOL_INT",
]
