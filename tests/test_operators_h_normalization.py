# tests/test_operators_h_normalization.py
from __future__ import annotations

import torch

from maxent.operators.forward import Operators


def test_normalize_makes_unit_mass_and_h_is_finite() -> None:
    torch.manual_seed(0)

    # small shapes
    R = 2
    T = 5
    D = 101

    beta = 10.0
    tau = torch.linspace(0.0, beta, T, dtype=torch.float64)
    x = torch.linspace(0.0, 1.0, D, dtype=torch.float64)

    # linear phi_inv: ω = w_min + x (w_max - w_min)
    w_min, w_max = -3.0, 5.0
    phi_inv = lambda xx: w_min + xx * (w_max - w_min)

    # Use exact Jacobian for linear map
    jacobian = lambda xx: (w_max - w_min) * torch.ones_like(xx)

    # Simple positive D_w (R,D)
    D_w = torch.exp(-((vmap_phi_inv := torch.vmap(phi_inv))(x))[None, :] ** 2).repeat(R, 1)
    D_w = D_w.to(torch.float64)

    # dummy observed G_tau and sigma
    G_tau = torch.zeros((R, T), dtype=torch.float64)
    sigma = 0.1 * torch.ones((R, T), dtype=torch.float64)

    ops = Operators(
        phi_inv=phi_inv,
        beta=beta,
        tau=tau,
        x=x,
        D_w=D_w,
        sigma=sigma,
        G_tau=G_tau,
        jacobian=jacobian,
    )

    # random positive n
    n = torch.rand((R, D), dtype=torch.float64)

    # normalize should enforce ∫ n dx = 1
    n_norm = ops.normalize(n)
    mass = torch.trapz(n_norm, x, dim=1)
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-12, rtol=0.0)

    # K and h should be finite
    K = ops._K_tau_x(x.repeat(R, 1))
    h = ops._h(n_norm, K)

    assert h.shape == (R, T)
    assert torch.isfinite(h).all()
