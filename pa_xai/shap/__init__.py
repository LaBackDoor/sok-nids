"""Protocol-Aware SHAP sub-package."""

from pa_xai.shap.explainer import ProtocolAwareSHAP
from pa_xai.shap.metrics import additivity_check

__all__ = ["ProtocolAwareSHAP", "additivity_check"]
