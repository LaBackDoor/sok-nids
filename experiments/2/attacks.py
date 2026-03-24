"""Adversarial attack implementations: FGSM and PGD for NIDS DNN models.

Implements M1 (Input Manipulation) threat model attacks with L-infinity norm constraints.
"""

import logging
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib.util as _ilu
import os as _os
_spec = _ilu.spec_from_file_location(
    "exp2_config", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "config.py")
)
_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
AttackConfig = _cfg.AttackConfig

logger = logging.getLogger(__name__)


def fgsm_attack(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    device: torch.device,
    constraint_projector: Optional[Callable] = None,
) -> np.ndarray:
    """Fast Gradient Sign Method (FGSM) single-step attack.

    x_adv = x + epsilon * sign(grad_x L(model(x), y))
    Clamps to [0, 1] (assumes Min-Max scaled input).
    If constraint_projector is provided, projects onto the feasible feature set.
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    X_tensor.requires_grad_(True)

    output = model(X_tensor)
    loss = F.cross_entropy(output, y_tensor)
    loss.backward()

    perturbation = epsilon * X_tensor.grad.sign()
    X_adv = (X_tensor + perturbation).clamp(0, 1)

    if constraint_projector is not None:
        X_adv = constraint_projector(X_adv.detach(), X_tensor.detach(), epsilon)

    return X_adv.detach().cpu().numpy().astype(np.float32)


def pgd_attack(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    step_size: float,
    num_steps: int,
    device: torch.device,
    constraint_projector: Optional[Callable] = None,
) -> np.ndarray:
    """Projected Gradient Descent (PGD) iterative attack.

    x^{t+1} = Proj_{B(x, epsilon)}(x^t + alpha * sign(grad_x L(model(x^t), y)))
    Projects perturbation back to L-inf epsilon-ball around original input.
    If constraint_projector is provided, projects onto the feasible feature set
    after each gradient step (alternating projection).
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    # Random initialization within epsilon-ball
    X_adv = X_tensor + torch.empty_like(X_tensor).uniform_(-epsilon, epsilon)
    X_adv = X_adv.clamp(0, 1).detach()

    if constraint_projector is not None:
        X_adv = constraint_projector(X_adv, X_tensor, epsilon)

    for _ in range(num_steps):
        X_adv.requires_grad_(True)
        output = model(X_adv)
        loss = F.cross_entropy(output, y_tensor)
        loss.backward()

        with torch.no_grad():
            X_adv = X_adv + step_size * X_adv.grad.sign()
            # Project back to L-inf epsilon-ball
            delta = (X_adv - X_tensor).clamp(-epsilon, epsilon)
            X_adv = (X_tensor + delta).clamp(0, 1)

            if constraint_projector is not None:
                X_adv = constraint_projector(X_adv, X_tensor, epsilon)
        X_adv = X_adv.detach()

    return X_adv.cpu().numpy().astype(np.float32)


def generate_adversarial_examples(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    config: AttackConfig,
    device: torch.device,
    constraint_projector: Optional[Callable] = None,
) -> dict:
    """Generate adversarial examples using FGSM and PGD at multiple epsilon values.

    Args:
        constraint_projector: Optional callable(X_adv, X_orig, epsilon) -> X_adv
            that projects perturbations onto the valid feature manifold.
            When None, only standard [0,1] clamping is applied.

    Returns:
        dict with keys:
            "fgsm": {epsilon: X_adv_array, ...}
            "pgd": {epsilon: X_adv_array, ...}
            "indices": sample indices used
            "X_clean": clean samples
            "y_clean": clean labels
            "fgsm_success": {epsilon: rate, ...}
            "pgd_success": {epsilon: rate, ...}
    """
    # Subsample
    n = min(config.num_attack_samples, len(X))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(X), size=n, replace=False)
    X_sub = X[indices]
    y_sub = y[indices]

    # Unwrap DataParallel
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    base_model.eval()

    batch_size = config.attack_batch_size

    # Clean predictions
    with torch.no_grad():
        preds_clean = []
        for i in range(0, n, batch_size):
            batch = torch.tensor(X_sub[i : i + batch_size], dtype=torch.float32, device=device)
            preds_clean.append(base_model(batch).argmax(1).cpu().numpy())
        preds_clean = np.concatenate(preds_clean)

    results = {
        "fgsm": {},
        "pgd": {},
        "indices": indices,
        "X_clean": X_sub,
        "y_clean": y_sub,
        "preds_clean": preds_clean,
        "fgsm_success": {},
        "pgd_success": {},
    }

    # --- FGSM at each epsilon ---
    for eps in config.fgsm_epsilons:
        logger.info(f"  FGSM attack (epsilon={eps})")
        start = time.time()
        adv_batches = []
        for i in range(0, n, batch_size):
            X_batch = X_sub[i : i + batch_size]
            y_batch = y_sub[i : i + batch_size]
            X_adv = fgsm_attack(base_model, X_batch, y_batch, eps, device, constraint_projector)
            adv_batches.append(X_adv)
        X_adv_full = np.concatenate(adv_batches, axis=0)
        results["fgsm"][eps] = X_adv_full
        elapsed = time.time() - start

        # Success rate: fraction where prediction changed
        with torch.no_grad():
            preds_adv = []
            for i in range(0, n, batch_size):
                batch = torch.tensor(
                    X_adv_full[i : i + batch_size], dtype=torch.float32, device=device
                )
                preds_adv.append(base_model(batch).argmax(1).cpu().numpy())
            preds_adv = np.concatenate(preds_adv)
        success_rate = float((preds_clean != preds_adv).mean())
        results["fgsm_success"][eps] = success_rate

        # L-inf perturbation magnitude
        linf = float(np.max(np.abs(X_adv_full - X_sub)))
        logger.info(
            f"    FGSM eps={eps}: success={success_rate:.4f}, "
            f"L-inf={linf:.4f} ({elapsed:.1f}s)"
        )

    # --- PGD at each epsilon ---
    for eps in config.pgd_epsilons:
        step_size = config.pgd_step_size_factor * eps / config.pgd_num_steps
        logger.info(
            f"  PGD attack (epsilon={eps}, steps={config.pgd_num_steps}, "
            f"step_size={step_size:.5f})"
        )
        start = time.time()
        adv_batches = []
        for i in range(0, n, batch_size):
            X_batch = X_sub[i : i + batch_size]
            y_batch = y_sub[i : i + batch_size]
            X_adv = pgd_attack(
                base_model, X_batch, y_batch, eps, step_size, config.pgd_num_steps, device,
                constraint_projector,
            )
            adv_batches.append(X_adv)
        X_adv_full = np.concatenate(adv_batches, axis=0)
        results["pgd"][eps] = X_adv_full
        elapsed = time.time() - start

        with torch.no_grad():
            preds_adv = []
            for i in range(0, n, batch_size):
                batch = torch.tensor(
                    X_adv_full[i : i + batch_size], dtype=torch.float32, device=device
                )
                preds_adv.append(base_model(batch).argmax(1).cpu().numpy())
            preds_adv = np.concatenate(preds_adv)
        success_rate = float((preds_clean != preds_adv).mean())
        results["pgd_success"][eps] = success_rate

        linf = float(np.max(np.abs(X_adv_full - X_sub)))
        logger.info(
            f"    PGD eps={eps}: success={success_rate:.4f}, "
            f"L-inf={linf:.4f} ({elapsed:.1f}s)"
        )

    return results
