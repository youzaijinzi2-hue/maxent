from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from maxent.operators.transform import forward_gtau_from_A


def build_tau_observation_from_A(
    *,
    A_true: torch.Tensor,
    omega: torch.Tensor,
    tau: torch.Tensor,
    beta: float,
    sigma_tau: torch.Tensor,
    prep_cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ROUTE-A: build τ-observation from A->Gτ forward (no finite-sum truncation bias)
      G_tau_true = forward_gtau_from_A(A_true,...)
      G_tau_noisy = G_tau_true + N(0, sigma_tau*tau_noise_scale)
    """
    G_tau_true = forward_gtau_from_A(A_true, omega, tau, beta).to(torch.float64)
    tau_noise_scale = float(prep_cfg.get("tau_noise_scale", 1.0))
    noise = torch.randn_like(G_tau_true) * (sigma_tau * tau_noise_scale)
    G_tau_noisy = (G_tau_true + noise).to(torch.float64)
    print(f"[info] τ-observation: built from forward_gtau_from_A + N(0, sigma_tau*{tau_noise_scale:g})")
    return G_tau_true, G_tau_noisy