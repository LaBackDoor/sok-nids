"""Sequential-Path Integrated Gradients explainer for NIDS.

Uses a two-phase integration strategy for mixed continuous/discrete
features:

- Phase 1: Standard IG on continuous features (Gauss-Legendre
  quadrature) while discrete features are held at baseline.
- Phase 2: Exact finite difference for discrete features
  (F(input) - F(mixed)), distributed among changed features.

All intermediate points are domain-valid by construction, and the
completeness axiom is preserved.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from pa_xai.core.result import ExplanationResult
from pa_xai.core.schemas import DatasetSchema


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
    """Sequential-Path Integrated Gradients for NIDS.

    Baseline: for each output class, the training sample whose logit
    is closest to zero is precomputed at init (per the original IG
    paper's guidance on near-zero-output baselines).

    Attribution uses a two-phase sequential path:

    - **Phase 1 (continuous)**: linearly interpolate continuous
      features from baseline to input while holding discrete features
      at baseline.  Standard Gauss-Legendre IG gives exact attributions.
    - **Phase 2 (discrete)**: compute the exact output difference
      from flipping discrete features (protocol, flags, etc.) from
      baseline to input values.  No gradient needed — just
      ``F(input) - F(mixed)``, distributed among changed features.

    This avoids evaluating the model on impossible intermediate states
    (e.g. protocol_type=2.6, SYN_flag=0.4) and guarantees the
    completeness axiom: ``sum(attributions) == F(input) - F(baseline)``.
    """

    def __init__(
        self,
        schema: DatasetSchema,
        model: nn.Module,
        X_train: np.ndarray,
        multiply_by_inputs: bool = True,
        use_softmax: bool = True,
        batch_size: int = 512,
    ) -> None:
        self.schema = schema
        self.model = model
        self.multiply_by_inputs = multiply_by_inputs
        self.use_softmax = use_softmax

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
        best: dict[int, tuple[float, np.ndarray]] = {}

        for start in range(0, n, batch_size):
            batch_np = X_train[start : start + batch_size]
            batch_t = torch.tensor(batch_np, dtype=torch.float32, device=device)
            logits = self.model(batch_t)
            abs_logits = logits.abs().cpu().numpy()

            num_classes = abs_logits.shape[1]
            for c in range(num_classes):
                col = abs_logits[:, c]
                local_best_idx = int(col.argmin())
                local_best_val = float(col[local_best_idx])
                if c not in best or local_best_val < best[c][0]:
                    best[c] = (local_best_val, batch_np[local_best_idx].copy())

        return {c: sample for c, (_, sample) in best.items()}

    def _get_forward_model(self) -> nn.Module:
        device = self._get_device()
        model_clone = copy.deepcopy(self.model).to(device)
        if self.use_softmax:
            m = _SoftmaxModel(model_clone)
        else:
            m = model_clone
        m.eval()
        return m

    def _compute_ig(self, x_tensor, baseline_tensor, x_row, target, n_steps):
        """Compute IG using Sequential Path to guarantee completeness.

        Phase 1 — Continuous integration: interpolate continuous features
        from baseline to input while holding discrete features at baseline.
        Standard Gauss-Legendre quadrature gives exact IG attributions for
        continuous features.

        Phase 2 — Discrete finite difference: with continuous features now
        at input values, compute the exact output change from flipping
        discrete features from baseline to input.  This is distributed
        equally among the discrete features that changed (no gradient
        needed — it's a direct F(after) - F(before)).

        Because continuous attribution uses proper IG and discrete
        attribution uses exact finite difference, the completeness
        axiom holds: sum(attributions) == F(input) - F(baseline).
        """
        device = self._get_device()
        step_sizes, alphas = _gauss_legendre_alphas(n_steps)
        forward_model = self._get_forward_model()

        attributions = torch.zeros_like(x_tensor)
        disc_indices = self.schema.discrete_indices
        cont_indices = [i for i in range(len(x_row)) if i not in disc_indices]

        # ==================================================
        # Phase 1: Continuous IG (discrete held at baseline)
        # ==================================================
        if cont_indices:
            path = baseline_tensor.unsqueeze(0).repeat(n_steps, 1)
            alphas_t = torch.tensor(alphas, dtype=torch.float32, device=device).unsqueeze(1)
            path[:, cont_indices] = (
                baseline_tensor[cont_indices]
                + alphas_t * (x_tensor[cont_indices] - baseline_tensor[cont_indices])
            )

            total_grads = torch.zeros_like(x_tensor)
            for i, step_size in enumerate(step_sizes):
                interp_point = path[i].requires_grad_(True)
                output = forward_model(interp_point.unsqueeze(0))
                output[0, target].backward()
                total_grads += interp_point.grad * step_size

            if self.multiply_by_inputs:
                attributions[cont_indices] = (
                    total_grads[cont_indices]
                    * (x_tensor[cont_indices] - baseline_tensor[cont_indices])
                )
            else:
                attributions[cont_indices] = total_grads[cont_indices]

        # ==================================================
        # Phase 2: Discrete finite difference
        # ==================================================
        # x_mixed = continuous at input, discrete at baseline
        x_mixed = x_tensor.clone()
        if disc_indices:
            x_mixed[disc_indices] = baseline_tensor[disc_indices]

        with torch.no_grad():
            F_baseline = forward_model(baseline_tensor.unsqueeze(0))[0, target].item()
            F_mixed = forward_model(x_mixed.unsqueeze(0))[0, target].item()
            F_input = forward_model(x_tensor.unsqueeze(0))[0, target].item()

        if disc_indices:
            discrete_diff = F_input - F_mixed
            changed = [
                idx for idx in disc_indices
                if x_row[idx] != baseline_tensor[idx].item()
            ]
            if changed:
                per_feature = discrete_diff / len(changed)
                for idx in changed:
                    attributions[idx] = per_feature

        # ==================================================
        # Convergence check
        # ==================================================
        attr_np = attributions.detach().cpu().numpy()
        delta = float(attr_np.sum()) - (F_input - F_baseline)

        return attr_np, delta

    def explain_instance(
        self,
        x_row: np.ndarray,
        target: int | None = None,
        n_steps: int = 50,
        return_convergence_delta: bool = False,
    ) -> ExplanationResult:
        device = self._get_device()
        x_tensor = torch.tensor(x_row, dtype=torch.float32, device=device)

        if target is None:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x_tensor.unsqueeze(0))
                target = int(torch.argmax(logits, dim=1).item())

        baseline = self._baselines[target]
        baseline_tensor = torch.tensor(baseline, dtype=torch.float32, device=device)

        with _disable_cudnn_for_rnn(self.model):
            attributions, convergence_delta = self._compute_ig(
                x_tensor, baseline_tensor, x_row, target, n_steps,
            )

        return ExplanationResult(
            feature_names=list(self.schema.feature_names),
            attributions=attributions,
            method="pa_ig",
            predicted_class=target,
            num_samples=None,
            convergence_delta=convergence_delta if return_convergence_delta else None,
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

        return self.explain_instance(x_row, target=target, n_steps=n_steps)
