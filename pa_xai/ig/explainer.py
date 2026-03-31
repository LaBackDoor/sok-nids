"""Protocol-Aware Integrated Gradients explainer for NIDS."""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from pa_xai.core.baseline import get_protocol_valid_baseline
from pa_xai.core.constraints import ConstraintEnforcer
from pa_xai.core.result import ExplanationResult
from pa_xai.core.schemas import DatasetSchema, TCP_PROTOCOL_INT, detect_protocol_encoding


class _SoftmaxModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.model(x))


def _has_rnn_modules(model: nn.Module) -> bool:
    return any(
        isinstance(m, (nn.LSTM, nn.GRU, nn.RNN))
        for m in model.modules()
    )


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


def _gauss_legendre_alphas(n_steps: int):
    nodes, weights = np.polynomial.legendre.leggauss(n_steps)
    alphas = 0.5 * (1.0 + nodes)
    step_sizes = 0.5 * weights
    return step_sizes, alphas


class ProtocolAwareIG:
    """Protocol-Aware Integrated Gradients."""

    def __init__(
        self,
        schema: DatasetSchema,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        benign_label: int = 0,
        baseline_top_k: int = 1,
        baseline_strategy: str = "nearest",
        constrain_path: bool = True,
        multiply_by_inputs: bool = True,
        use_softmax: bool = True,
        tcp_label_value: int = TCP_PROTOCOL_INT,
    ) -> None:
        self.schema = schema
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.benign_label = benign_label
        self.baseline_top_k = baseline_top_k
        self.baseline_strategy = baseline_strategy
        self.constrain_path = constrain_path
        self.multiply_by_inputs = multiply_by_inputs
        self.use_softmax = use_softmax
        self.tcp_label_value = tcp_label_value
        self.enforcer = ConstraintEnforcer(schema)

    def _get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _resolve_protocol_params(self, x_row: np.ndarray):
        encoding = self.schema.protocol_encoding
        protocol_value = None
        tcp_val = self.tcp_label_value
        if self.schema.protocol_index is not None:
            protocol_value = x_row[self.schema.protocol_index]
            if encoding == "auto":
                encoding = detect_protocol_encoding(
                    x_row, self.schema.protocol_feature, self.schema.feature_names
                )
            if encoding == "string":
                tcp_val = self.tcp_label_value
        return protocol_value, encoding, tcp_val

    def _constrained_ig(self, x_tensor, baseline_tensor, x_row, target, n_steps, method):
        """Compute IG with constraint enforcement at each interpolation step.

        Note: Clamping intermediate points to satisfy domain constraints alters
        the integration path, which breaks the IG completeness axiom
        (sum(attributions) != F(input) - F(baseline)). The convergence_delta
        field reports the magnitude of this deviation. This is a deliberate
        trade-off: domain-valid intermediate states vs. theoretical completeness.
        """
        device = self._get_device()
        protocol_value, encoding, tcp_val = self._resolve_protocol_params(x_row)
        step_sizes, alphas = _gauss_legendre_alphas(n_steps)

        if self.use_softmax:
            forward_model = _SoftmaxModel(self.model)
        else:
            forward_model = self.model
        forward_model.eval()

        # Generate all interpolation points at once (numpy)
        x_np = x_row.astype(np.float32)
        baseline_np = baseline_tensor.detach().cpu().numpy()
        interp_batch = np.array([
            baseline_np + alpha * (x_np - baseline_np) for alpha in alphas
        ])  # shape: (n_steps, D)

        # Apply constraints once to the entire batch
        self.enforcer.enforce(interp_batch, protocol_value, encoding, tcp_val)

        # Convert to torch once
        interp_tensor = torch.tensor(interp_batch, dtype=torch.float32, device=device)

        # Compute gradients per step
        total_grads = torch.zeros_like(x_tensor)
        for i, step_size in enumerate(step_sizes):
            interp_point = interp_tensor[i].requires_grad_(True)
            output = forward_model(interp_point.unsqueeze(0))
            output[0, target].backward()
            total_grads += interp_point.grad * step_size

        if self.multiply_by_inputs:
            attributions = total_grads * (x_tensor - baseline_tensor)
        else:
            attributions = total_grads

        # Compute convergence delta (measures deviation from completeness axiom)
        with torch.no_grad():
            F_input = forward_model(x_tensor.unsqueeze(0))[0, target].item()
            F_baseline = forward_model(baseline_tensor.unsqueeze(0))[0, target].item()
        attr_np = attributions.detach().cpu().numpy()
        delta = float(attr_np.sum()) - (F_input - F_baseline)

        return attr_np, delta

    def _captum_ig(self, x_tensor, baseline_tensor, target, n_steps, method,
                   internal_batch_size, return_convergence_delta):
        from captum.attr import IntegratedGradients
        if self.use_softmax:
            forward_model = _SoftmaxModel(self.model)
        else:
            forward_model = self.model
        forward_model.eval()
        ig = IntegratedGradients(forward_model, multiply_by_inputs=self.multiply_by_inputs)

        result = ig.attribute(
            x_tensor.unsqueeze(0),
            baselines=baseline_tensor.unsqueeze(0),
            target=target,
            n_steps=n_steps,
            method=method,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=return_convergence_delta,
        )

        if return_convergence_delta:
            attrs, delta = result
            return attrs.detach().cpu().numpy().flatten(), float(delta.item())
        else:
            return result.detach().cpu().numpy().flatten(), None

    def explain_instance(
        self,
        x_row: np.ndarray,
        target: int | None = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: int | None = None,
        return_convergence_delta: bool = False,
    ) -> ExplanationResult:
        device = self._get_device()

        baseline = get_protocol_valid_baseline(
            x_row, self.X_train, self.y_train, self.schema,
            benign_label=self.benign_label,
            top_k=self.baseline_top_k,
            strategy=self.baseline_strategy,
        )

        x_tensor = torch.tensor(x_row, dtype=torch.float32, device=device)
        baseline_tensor = torch.tensor(baseline, dtype=torch.float32, device=device)

        if target is None:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x_tensor.unsqueeze(0))
                target = int(torch.argmax(logits, dim=1).item())

        convergence_delta = None
        with _disable_cudnn_for_rnn(self.model):
            if self.constrain_path:
                attributions, convergence_delta = self._constrained_ig(
                    x_tensor, baseline_tensor, x_row, target, n_steps, method,
                )
            else:
                attributions, convergence_delta = self._captum_ig(
                    x_tensor, baseline_tensor, target, n_steps, method,
                    internal_batch_size, return_convergence_delta,
                )

        return ExplanationResult(
            feature_names=list(self.schema.feature_names),
            attributions=attributions,
            method="pa_ig",
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
        n_steps: int = 50,
        method: str = "gausslegendre",
    ) -> ExplanationResult:
        """Generate an IG explanation from a PCAP file."""
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

        return self.explain_instance(x_row, target=target, n_steps=n_steps, method=method)
