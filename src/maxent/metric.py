import torch
from maxent.operators.transform import giw_from_A

import torch

def white_iw(G_iw_rec: torch.Tensor, G_iw_noisy: torch.Tensor, sigma_iw: torch.Tensor | float):
    """
    Compute whitened chi^2 in Matsubara (iω) space.

    Accepts:
      - sigma_iw: scalar (float or 0-d tensor)
      - sigma_iw: 1D tensor of shape (Nw,) matching r.numel()

    Returns:
      sqrt_chi2 = sqrt(chi2)
      chi2_per_n = chi2 / Nw   (expected ~ 1 if noise model is correct)
    """
    r = (G_iw_rec - G_iw_noisy)

    # sigma_iw -> tensor on same device/dtype (real)
    sigma = torch.as_tensor(sigma_iw, device=r.device, dtype=torch.float64)
    if sigma.ndim == 0:
        # scalar -> broadcast
        denom2 = (sigma * sigma)
    elif sigma.ndim == 1:
        if sigma.numel() != r.numel():
            raise ValueError(f"sigma_iw length mismatch: sigma={sigma.numel()} vs r={r.numel()}")
        denom2 = sigma * sigma  # (Nw,)
    else:
        raise ValueError("sigma_iw must be scalar or 1D tensor")

    # (Re^2 + Im^2) / sigma^2
    # If denom2 is (Nw,), broadcasting works elementwise.
    w2 = torch.sum((r.real.to(torch.float64)**2 + r.imag.to(torch.float64)**2) / denom2)

    return torch.sqrt(w2), w2 / r.numel()


def estimate_c_scalar(A_est, omega, wn, G_iw_target):
    # compute Giw with c=1
    G_iw_c1 = giw_from_A(A_est, omega, wn)  # complex
    # Find scalar c minimizing ||c*G_iw_c1 - G_iw_target||_2
    # Use complex inner product
    num = torch.sum(torch.conj(G_iw_c1) * G_iw_target)
    den = torch.sum(torch.conj(G_iw_c1) * G_iw_c1) + 1e-30
    c = (num / den).real  # take real scalar
    return c, G_iw_c1