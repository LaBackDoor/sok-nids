"""PA-LIME: Protocol-Aware LIME for NIDS model auditing.

A domain-aware extension of LIME that enforces physical network protocol
invariants during perturbation sampling, providing robust and semantically
valid explanations for Network Intrusion Detection System classifiers.

Supported datasets (built-in schemas):
  - CIC-IDS-2017
  - CSE-CIC-IDS2018
  - NSL-KDD
  - UNSW-NB15 (native Argus/Bro features)
  - UNSW-NB15 (CICFlowMeter-augmented)

Custom schemas can be defined via DatasetSchema and HierarchicalConstraint.
"""

from pa_lime.explainer import ProtocolAwareLIME
from pa_lime.metrics import fidelity, semantic_robustness, sparsity
from pa_lime.result import ExplanationResult
from pa_lime.schemas import (
    CIC_IDS_2017,
    CSE_CIC_IDS2018,
    UNSW_NB15_CIC,
    UNSW_NB15_NATIVE,
    NSL_KDD,
    BUILTIN_SCHEMAS,
    DatasetSchema,
    HierarchicalConstraint,
    get_schema,
)

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
