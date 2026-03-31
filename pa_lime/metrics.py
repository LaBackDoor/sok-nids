"""Backwards-compatibility shim. All functionality has moved to pa_xai."""
from pa_xai.core.metrics import sparsity
from pa_xai.lime.metrics import fidelity, semantic_robustness

__all__ = ["sparsity", "fidelity", "semantic_robustness"]
