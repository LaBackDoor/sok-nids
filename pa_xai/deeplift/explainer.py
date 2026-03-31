"""Protocol-Aware DeepLIFT explainer for NIDS."""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from pa_xai.core.baseline import get_protocol_valid_baseline
from pa_xai.core.result import ExplanationResult
from pa_xai.core.schemas import DatasetSchema, TCP_PROTOCOL_INT


class _SoftmaxModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)


def _has_rnn_modules(model: nn.Module) -> bool:
    return any(isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)) for m in model.modules())


@contextmanager
def _disable_cudnn_for_rnn(model):
    has_rnn = _has_rnn_modules(model)
    if has_rnn:
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        if has_rnn:
            torch.backends.cudnn.enabled = prev


class ProtocolAwareDeepLIFT:
    """Protocol-Aware DeepLIFT.

    Uses Nearest Benign Prototype baselines. No path clamping —
    DeepLIFT evaluates only baseline and input.
    """

    def __init__(
        self,
        schema: DatasetSchema,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        benign_label: int = 0,
        baseline_top_k: int = 1,
        baseline_strategy: str = "nearest",
        multiply_by_inputs: bool = True,
        eps: float = 1e-10,
        tcp_label_value: int = TCP_PROTOCOL_INT,
    ) -> None:
        self.schema = schema
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.benign_label = benign_label
        self.baseline_top_k = baseline_top_k
        self.baseline_strategy = baseline_strategy
        self.multiply_by_inputs = multiply_by_inputs
        self.eps = eps
        self.tcp_label_value = tcp_label_value

    def _get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def explain_instance(
        self,
        x_row: np.ndarray,
        target: int | None = None,
        return_convergence_delta: bool = False,
    ) -> ExplanationResult:
        from captum.attr import DeepLift

        device = self._get_device()

        baseline = get_protocol_valid_baseline(
            x_row, self.X_train, self.y_train, self.schema,
            benign_label=self.benign_label,
            top_k=self.baseline_top_k,
            strategy=self.baseline_strategy,
        )

        x_tensor = torch.tensor(x_row, dtype=torch.float32, device=device).unsqueeze(0)
        baseline_tensor = torch.tensor(baseline, dtype=torch.float32, device=device).unsqueeze(0)

        if target is None:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x_tensor)
                target = int(torch.argmax(logits, dim=1).item())

        softmax_model = _SoftmaxModel(self.model)
        softmax_model.eval()
        dl = DeepLift(softmax_model, eps=self.eps)

        convergence_delta = None
        with warnings.catch_warnings(), _disable_cudnn_for_rnn(self.model):
            warnings.filterwarnings("ignore", message="Setting forward, backward hooks")
            x_input = x_tensor.requires_grad_(True)
            result = dl.attribute(
                x_input,
                baselines=baseline_tensor,
                target=target,
                return_convergence_delta=return_convergence_delta,
            )
            if return_convergence_delta:
                attrs, delta = result
                convergence_delta = float(delta.item())
            else:
                attrs = result

        attributions = attrs.detach().cpu().numpy().flatten()

        return ExplanationResult(
            feature_names=list(self.schema.feature_names),
            attributions=attributions,
            method="pa_deeplift",
            predicted_class=target,
            num_samples=None,
            convergence_delta=convergence_delta,
            baseline_used=baseline,
        )
