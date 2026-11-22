# tests/test_green_forward_consistency.py
from __future__ import annotations

import torch

from maxent.preprocess.xas_preprocess import Prepare


def test_green_forward_matches_trapz_on_nonuniform_grid() -> None:
    """
    Ensure Prepare forward integration matches torch.trapz even on NON-uniform omega grid.

    This test guards future refactors of:
      - kernel construction
      - integration discretization (must not assume uniform grid)
      - dtype/shape handling
    """
    prep = Prepare()

    # Non-uniform omega grid (fixed, deterministic)
    omega_u = torch.linspace(-5.0, 5.0, 401, dtype=torch.float64)
    omega_t = omega_u + 0.02 * torch.sin(3.0 * omega_u)  # small deterministic jitter
    omega_t = torch.unique_consecutive(omega_t)           # safety
    omega_t, _ = torch.sort(omega_t)


    # Nonnegative "spectrum-like" test function
    D_t = torch.exp(-omega_t**2).to(torch.float64)

    beta = 10.0
    Nn = 8  # small, but enough

    wn = prep._make_matsubara_wn(beta=beta, Nn=Nn, device=omega_t.device)  # complex128
    K = prep._kernel_iwn(wn=wn, omega_t=omega_t)  # (2Nn, Nw)

    # Implementation under test (general trapz)
    G_impl = prep._integrate_trapz(K=K, D=D_t, omega_t=omega_t)  # (2Nn,)

    # Reference: direct trapz over omega axis
    integrand = K * D_t.to(torch.complex128)  # broadcast to (2Nn, Nw)
    G_ref = torch.trapz(integrand, omega_t, dim=1)

    # Basic shape / dtype checks
    assert G_impl.shape == G_ref.shape
    assert torch.is_complex(G_impl)
    assert G_impl.dtype == torch.complex128

    # Relative error check (avoid division by tiny numbers)
    denom = torch.maximum(torch.abs(G_ref), torch.tensor(1e-12, dtype=torch.float64))
    rel = torch.abs(G_impl - G_ref) / denom

    # Should be extremely close even on non-uniform grid
    assert torch.max(rel).item() < 1e-10
