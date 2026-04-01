"""DeepLIFT explainer for NIDS with min-logit baseline selection."""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from pa_xai.core.result import ExplanationResult
from pa_xai.core.schemas import DatasetSchema


_DEEPLIFT_SUPPORTED = (
    nn.ReLU, nn.ELU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.Softplus,
    nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.Softmax,
)

_DEEPLIFT_UNSUPPORTED_COMMON = (nn.GELU, nn.SiLU, nn.Mish)


class _SoftmaxModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.model(x))


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
    """DeepLIFT for NIDS with min-logit baseline selection.

    Baseline: for each output class, the training sample whose logit
    is closest to zero is precomputed at init (per the original
    DeepLIFT paper's guidance on near-zero-output baselines).  This
    is deterministic and requires no per-explanation forward passes.

    DeepLIFT evaluates only two points (baseline and input), so no
    path interpolation or constraint enforcement is needed.
    """

    def __init__(
        self,
        schema: DatasetSchema,
        model: nn.Module,
        X_train: np.ndarray,
        multiply_by_inputs: bool = True,
        eps: float = 1e-10,
        batch_size: int = 512,
    ) -> None:
        self.schema = schema
        self.model = model
        unsupported = [
            type(m).__name__ for m in model.modules()
            if isinstance(m, _DEEPLIFT_UNSUPPORTED_COMMON)
        ]
        if unsupported:
            warnings.warn(
                f"Model contains activations unsupported by DeepLIFT's rescale rule: "
                f"{unsupported}. These layers will use standard gradients instead.",
                stacklevel=2,
            )
        self.multiply_by_inputs = multiply_by_inputs
        self.eps = eps

        # Precompute one baseline per output class: the training sample
        # whose logit for that class is closest to zero.
        self._baselines = self._precompute_baselines(X_train, batch_size)

    def _get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def _precompute_baselines(
        self, X_train: np.ndarray, batch_size: int,
    ) -> dict[int, np.ndarray]:
        """Forward-pass all training samples and keep one per class.

        For each output class, stores the training sample whose logit
        for that class is closest to zero.
        """
        device = self._get_device()
        self.model.eval()

        n = len(X_train)
        # Accumulate per-class best: {class_idx: (min_abs_logit, sample)}
        best: dict[int, tuple[float, np.ndarray]] = {}

        for start in range(0, n, batch_size):
            batch_np = X_train[start : start + batch_size]
            batch_t = torch.tensor(batch_np, dtype=torch.float32, device=device)
            logits = self.model(batch_t)  # (B, num_classes)
            abs_logits = logits.abs().cpu().numpy()

            num_classes = abs_logits.shape[1]
            for c in range(num_classes):
                col = abs_logits[:, c]
                local_best_idx = int(col.argmin())
                local_best_val = float(col[local_best_idx])
                if c not in best or local_best_val < best[c][0]:
                    best[c] = (local_best_val, batch_np[local_best_idx].copy())

        baselines = {c: sample for c, (_, sample) in best.items()}
        return baselines

    def explain_instance(
        self,
        x_row: np.ndarray,
        target: int | None = None,
        return_convergence_delta: bool = False,
    ) -> ExplanationResult:
        from captum.attr import DeepLift

        device = self._get_device()
        x_tensor = torch.tensor(x_row, dtype=torch.float32, device=device).unsqueeze(0)

        if target is None:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x_tensor)
                target = int(torch.argmax(logits, dim=1).item())

        baseline = self._baselines[target]
        baseline_tensor = torch.tensor(baseline, dtype=torch.float32, device=device).unsqueeze(0)

        softmax_model = _SoftmaxModel(self.model)
        softmax_model.eval()
        dl = DeepLift(softmax_model, eps=self.eps, multiply_by_inputs=self.multiply_by_inputs)

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

    def explain_pcap(
        self,
        pcap_path: str,
        feature_fn,
        feature_names: list[str],
        mode: str = "packet",
        target: int | None = None,
        return_convergence_delta: bool = False,
    ) -> ExplanationResult:
        """Generate a DeepLIFT explanation from a PCAP file."""
        from pa_xai.pcap.pipeline import PcapPipeline

        pipeline = PcapPipeline()
        if mode == "packet":
            packets = pipeline.parser.parse_packets(pcap_path)
            if not packets:
                raise ValueError("No packets found in PCAP")
            x_row = feature_fn(packets[0])
        else:
            flows = pipeline.parser.parse_flows(pcap_path)
            if not flows:
                raise ValueError("No flows found in PCAP")
            x_row = feature_fn(flows[0])

        return self.explain_instance(x_row, target=target, return_convergence_delta=return_convergence_delta)
