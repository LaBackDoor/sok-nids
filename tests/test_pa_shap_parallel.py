"""Tests for parallelized PA-SHAP DNN."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments', '1'))

from pa_explainers import pa_explain_shap_dnn
from explainers import ExplanationResult


def test_pa_shap_dnn_uses_parallel():
    """Verify PA-SHAP DNN uses Parallel instead of sequential loop."""
    import inspect
    source = inspect.getsource(pa_explain_shap_dnn)
    assert "Parallel" in source, "pa_explain_shap_dnn should use sklearn Parallel"
    assert "for i in range(len(X_explain))" not in source, (
        "Sequential loop should be replaced with Parallel"
    )
