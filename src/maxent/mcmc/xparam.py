from __future__ import annotations

import torch
import torch.nn.functional as F


def y_to_x(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Unconstrained y (R,D) or (D,) -> monotone x in [0,1] with x[...,0]=0, x[...,-1]=1.
    """
    y = y.to(torch.float64)
    d = F.softplus(y) + eps                          # positive increments
    s = torch.cumsum(d, dim=-1)                      # strictly increasing
    s0 = s[..., :1]
    sL = s[..., -1:]
    denom = (sL - s0).clamp_min(eps)
    x = (s - s0) / denom                             # x0=0, xL=1
    return x


def softplus_inverse(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Inverse of softplus for z>0: y = log(exp(z)-1), stabilized.
    """
    z = z.clamp_min(eps)
    # For large z, exp(z) overflows; use approximation y≈z
    return torch.where(z > 20.0, z, torch.log(torch.expm1(z)))


def x_to_y(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Monotone x in [0,1] -> y such that y_to_x(y) ≈ x.
    Used only for initialization.
    """
    x = x.to(torch.float64)
    # enforce monotone roughly for safety (but don't sort; just clamp)
    x = x.clamp(0.0, 1.0)
    # delta_1 = x1-0, delta_i = x_i - x_{i-1}
    x_prev = torch.cat([torch.zeros_like(x[..., :1]), x[..., :-1]], dim=-1)
    delta = (x - x_prev).clamp_min(eps)
    y = softplus_inverse(delta)
    return y