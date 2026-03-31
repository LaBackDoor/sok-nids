"""Backwards-compatibility shim. All functionality has moved to pa_xai.lime."""
from pa_xai.lime.explainer import ProtocolAwareLIME

__all__ = ["ProtocolAwareLIME"]
