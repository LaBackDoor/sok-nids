"""Protocol-Aware Integrated Gradients sub-package."""

from pa_xai.ig.explainer import ProtocolAwareIG
from pa_xai.ig.metrics import path_convergence

__all__ = ["ProtocolAwareIG", "path_convergence"]
