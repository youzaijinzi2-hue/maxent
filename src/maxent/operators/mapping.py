import torch

def _interp1d(xq: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    Vectorized 1D linear interpolation.
    xp: (N,) increasing grid
    fp: (N,) values on xp
    xq: (...) query points
    returns: (...) interpolated values
    """
    # dtype/device unify
    xq = xq.to(dtype=xp.dtype, device=xp.device)

    # indices i such that xp[i-1] <= xq < xp[i]
    idx = torch.bucketize(xq, xp, right=False)

    # clamp to [1, N-1] so that i-1 and i are valid
    N = xp.numel()
    idx = idx.clamp(1, N - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]

    # avoid division by zero (shouldn't happen if xp strictly increasing)
    denom = (x1 - x0).clamp_min(torch.finfo(xp.dtype).tiny)
    t = (xq - x0) / denom
    return y0 + t * (y1 - y0)

def build_phi_phi_inv_jacobian_torch(
    omega: torch.Tensor,
    D_w: torch.Tensor,
    eps: float = 1e-30,
):
    """
    omega: (K,) strictly increasing
    D_w  : (K,) nonnegative density (not including Δω)
    eps  : floor for D when computing jacobian

    returns:
      phi(omega_query) -> x in [0,1]
      phi_inv(x_query) -> omega
      jacobian(x_query) -> dω/dx (= Z / D(ω(x))) with safe floor
      plus: x_grid, Z
    """
    if omega.ndim != 1 or D_w.ndim != 1 or omega.shape != D_w.shape:
        raise ValueError("omega and D_w must be 1D and same shape.")
    if not torch.all(torch.isfinite(omega)) or not torch.all(torch.isfinite(D_w)):
        raise ValueError("omega and D_w must be finite.")
    if not torch.all(omega[1:] > omega[:-1]):
        raise ValueError("omega must be strictly increasing.")
    if torch.any(D_w < 0):
        raise ValueError("D_w must be nonnegative.")

    omega = omega.to(dtype=torch.float64)
    D = D_w.to(dtype=torch.float64)
    
    # Z = ∫ D(ω) dω (trapezoid)
    dω = omega[1:] - omega[:-1]
    inc = 0.5 * (D[:-1] + D[1:]) * dω  # (K-1,)
    Z = inc.sum()
    if not torch.isfinite(Z) or Z <= 0:
        raise ValueError("Integral Z must be positive. Check D_w.")

    # cumulative integral C(ω_j) = ∫_{ω0}^{ωj} D dω
    C = torch.empty_like(D)
    C[0] = 0.0
    C[1:] = torch.cumsum(inc, dim=0)

    # x_grid = C / Z in [0,1]
    x_grid = C / Z
    # numerical safety: enforce endpoints
    x_grid[0] = 0.0
    x_grid[-1] = 1.0
    # enforce monotone (should hold for D>=0)
    x_grid = torch.maximum(x_grid, torch.cummax(x_grid, dim=0).values)

    def phi(omega_query: torch.Tensor) -> torch.Tensor:
        wq = omega_query.to(dtype=torch.float64)
        # interpolate ω -> x
        return _interp1d(wq, omega, x_grid).clamp(0.0, 1.0)

    def phi_inv(x_query: torch.Tensor) -> torch.Tensor:
        xq = x_query.to(dtype=torch.float64).clamp(0.0, 1.0)
        # interpolate x -> ω (x_grid monotone)
        return _interp1d(xq, x_grid, omega)

    def jacobian(x_query: torch.Tensor) -> torch.Tensor:
        """
        J(x)=dω/dx = Z / D(ω(x))
        """
        xq = x_query.to(dtype=torch.float64).clamp(0.0, 1.0)
        wq = phi_inv(xq)
        Dq = _interp1d(wq, omega, D)
        Dq = torch.clamp(Dq, min=eps)
        return Z / Dq

    return phi, phi_inv, jacobian, x_grid, Z