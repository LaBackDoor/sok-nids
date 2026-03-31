"""Backwards-compatibility shim. All functionality has moved to pa_xai.core.schemas."""
from pa_xai.core.schemas import (
    BUILTIN_SCHEMAS,
    CIC_IDS_2017,
    CSE_CIC_IDS2018,
    DatasetSchema,
    HierarchicalConstraint,
    NSL_KDD,
    UNSW_NB15_CIC,
    UNSW_NB15_NATIVE,
    detect_protocol_encoding,
    get_schema,
)

__all__ = [
    "DatasetSchema",
    "HierarchicalConstraint",
    "CIC_IDS_2017",
    "CSE_CIC_IDS2018",
    "NSL_KDD",
    "UNSW_NB15_NATIVE",
    "UNSW_NB15_CIC",
    "BUILTIN_SCHEMAS",
    "get_schema",
    "detect_protocol_encoding",
]
