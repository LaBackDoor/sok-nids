"""Protocol-Aware LIME sub-package."""

from pa_xai.lime.explainer import ProtocolAwareLIME
from pa_xai.lime.fuzzer import DomainConstraintFuzzer
from pa_xai.lime.metrics import fidelity, semantic_robustness

__all__ = [
    "ProtocolAwareLIME",
    "DomainConstraintFuzzer",
    "fidelity",
    "semantic_robustness",
]
