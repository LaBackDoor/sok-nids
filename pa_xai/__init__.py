"""PA-XAI: Protocol-Aware Explainable AI for Network Intrusion Detection."""

from pa_xai.core import (
    BUILTIN_SCHEMAS,
    CIC_IDS_2017,
    CSE_CIC_IDS2018,
    ConstraintEnforcer,
    DatasetSchema,
    HierarchicalConstraint,
    NSL_KDD,
    UNSW_NB15_CIC,
    UNSW_NB15_NATIVE,
    get_schema,
)
from pa_xai.core.baseline import get_protocol_valid_baseline
from pa_xai.core.metrics import sparsity
from pa_xai.core.result import ExplanationResult
from pa_xai.deeplift import ProtocolAwareDeepLIFT, convergence_delta
from pa_xai.ig import ProtocolAwareIG, path_convergence
from pa_xai.lime import ProtocolAwareLIME, fidelity, semantic_robustness
from pa_xai.shap import ProtocolAwareSHAP, additivity_check

__all__ = [
    # Core
    "DatasetSchema",
    "HierarchicalConstraint",
    "ConstraintEnforcer",
    "ExplanationResult",
    "get_protocol_valid_baseline",
    "CIC_IDS_2017",
    "CSE_CIC_IDS2018",
    "NSL_KDD",
    "UNSW_NB15_NATIVE",
    "UNSW_NB15_CIC",
    "BUILTIN_SCHEMAS",
    "get_schema",
    # Explainers
    "ProtocolAwareLIME",
    "ProtocolAwareIG",
    "ProtocolAwareDeepLIFT",
    "ProtocolAwareSHAP",
    # Shared metrics
    "semantic_robustness",
    "sparsity",
    # Method-specific metrics
    "fidelity",
    "path_convergence",
    "convergence_delta",
    "additivity_check",
]
