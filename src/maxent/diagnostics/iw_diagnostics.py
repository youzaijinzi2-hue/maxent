from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from maxent.metric import white_iw, estimate_c_scalar
from typing import Any, Dict, Optional
import logging

def run_iw_diagnostics(
    *,
    A_est: torch.Tensor,
    omega: torch.Tensor,
    wn: torch.Tensor,
    G_obs: torch.Tensor,
    sigma_iw: torch.Tensor,
    log: Optional[logging.Logger] = None,
    eps: float = 1e-300,
) -> Dict[str, Any]:
    """
    Optional iω-space diagnostics (reference/debug).

    What it does (same as old inline code):
      - reconstruct G(iω) from A_est
      - best-fit scale c (amplitude alignment)
      - whitened residual stats: chi2/Nw, z_rms, mean/std of real/imag
      - norms for sanity

    Returns a dict with key stats (useful for saving later if you want).
    """
    # logger fallback: if not provided, use a module logger (won't duplicate handlers)
    if log is None:
        log = logging.getLogger("maxent")
        
    # reconstruct Giw from A_est
    A = A_est.detach().to(torch.float64)
    om = omega.detach().to(torch.float64)

    c, G_iw_c1 = estimate_c_scalar(A, om, wn, G_obs)
    G_iw_rec = (c * G_iw_c1).to(torch.complex128)

    log.info("[iw] best-fit c = %.6g", float(c))

    sqrt_chi2, chi2_per_n = white_iw(G_iw_rec, G_obs, sigma_iw)
    log.info("[iw-whiten] chi2/Nw = %.6g", float(chi2_per_n))
    log.info("[iw-whiten] z_rms   = %.6g  (sqrt(chi2))", float(sqrt_chi2))

    # Compute whitened residual z = (G_rec - G_obs)/sigma_iw
    if torch.is_tensor(sigma_iw) and sigma_iw.ndim == 1:
        z_iw = (G_iw_rec - G_obs) / (sigma_iw.to(G_iw_rec.device) + eps)
    else:
        z_iw = (G_iw_rec - G_obs) / (float(sigma_iw) + eps)

    z_mean_real = torch.mean(z_iw.real)
    z_mean_imag = torch.mean(z_iw.imag)
    z_std_real = torch.std(z_iw.real, unbiased=False)
    z_std_imag = torch.std(z_iw.imag, unbiased=False)
    log.info("[iw-whiten] z mean/std (real) = %.6g / %.6g", float(z_mean_real), float(z_std_real))
    log.info("[iw-whiten] z mean/std (imag) = %.6g / %.6g", float(z_mean_imag), float(z_std_imag))

    n_obs = float(torch.linalg.norm(G_obs).real)
    n_rec = float(torch.linalg.norm(G_iw_rec).real)
    log.info("[iw] ||G_iw_noisy||_2  = %.6g", n_obs)
    log.info("[iw] ||G_iw_rec||_2    = %.6g", n_rec)
    log.info("[metric-iw] sqrt(chi2) = %.6g", float(sqrt_chi2))
    log.info("[metric-iw] chi2/Nw    = %.6g", float(chi2_per_n))

    return {
        "c_iw": float(c),
        "chi2_iw_per_point": float(chi2_per_n),
        "z_rms_iw": float(sqrt_chi2),
        "z_mean_real_iw": float(z_mean_real),
        "z_std_real_iw": float(z_std_real),
        "z_mean_imag_iw": float(z_mean_imag),
        "z_std_imag_iw": float(z_std_imag),
        "norm_G_obs": n_obs,
        "norm_G_iw_rec": n_rec,
    }