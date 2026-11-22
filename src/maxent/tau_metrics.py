from __future__ import annotations

from typing import Dict, Tuple

import torch


@torch.no_grad()
def compute_tau_metrics(
    *,
    G_tau_rec: torch.Tensor,
    G_tau_noisy: torch.Tensor,
    tau: torch.Tensor,
    beta: float,
    sigma_tau: torch.Tensor,
    tau_margin: float = 5.0,
    topk: int = 8,
) -> Dict[str, object]:
    """
    Compute τ-space metrics used in run_maxent.py.

    Returns a dict containing:
      - deltaGtau_abs / rel / white (global, full τ-grid)
      - core_abs / core_rel / core_white / core_N (masked central τ region)
      - chi2_tau / chi2_tau_per_point (global, whitened chi^2)
      - z_mean_*/z_std_* (global whitened residual stats)
      - core_chi2_tau / core_chi2_tau_per_point (core, whitened chi^2)
      - core_z_mean_*/core_z_std_* (core whitened residual stats)
      - topk_idx / topk_tau / topk_abs_r / topk_abs_r_over_mean_sigma (debug)
    """
    diff_tau = (G_tau_rec - G_tau_noisy)

    # ensure float64 for stable metrics
    diff_tau64 = diff_tau.to(torch.complex128) if torch.is_complex(diff_tau) else diff_tau.to(torch.float64)
    noisy_tau64 = G_tau_noisy.to(torch.complex128) if torch.is_complex(G_tau_noisy) else G_tau_noisy.to(torch.float64)
    sigma64 = sigma_tau.to(torch.float64)

    # global
    delta_abs = torch.linalg.norm(diff_tau64)
    delta_rel = delta_abs / (torch.linalg.norm(noisy_tau64) + 1e-30)
    r = diff_tau64 / (sigma64 + 1e-30)  # whitened residual
    delta_white = torch.linalg.norm(r)

    # chi^2 in tau space (whitened)
    # If r is complex, use |r|^2.
    abs_r2 = torch.abs(r) ** 2
    Nt = int(abs_r2.numel())
    chi2_tau = torch.sum(abs_r2)
    chi2_tau_per_point = chi2_tau / max(Nt, 1)

    # z-statistics: mean/std (report real/imag separately if complex)
    if torch.is_complex(r):
        z_mean_real = torch.mean(r.real)
        z_mean_imag = torch.mean(r.imag)
        z_std_real = torch.std(r.real, unbiased=False)
        z_std_imag = torch.std(r.imag, unbiased=False)
    else:
        z_mean_real = torch.mean(r)
        z_mean_imag = torch.tensor(0.0, dtype=torch.float64, device=r.device)
        z_std_real = torch.std(r, unbiased=False)
        z_std_imag = torch.tensor(0.0, dtype=torch.float64, device=r.device)

    # also provide RMS(z) = sqrt(chi2/N)
    z_rms = torch.sqrt(chi2_tau_per_point)

    # debug top residuals
    abs_r = torch.sqrt(abs_r2)
    k = min(int(topk), int(abs_r.numel()))
    vals, idx = torch.topk(abs_r, k=k)

    # core region mask
    mask = (tau >= float(tau_margin)) & (tau <= float(beta) - float(tau_margin))
    diff_core = diff_tau64[mask]
    noisy_core = noisy_tau64[mask]
    sigma_core = sigma64[mask]
    N = int(diff_core.numel())

    core_abs = torch.linalg.norm(diff_core)
    core_rel = core_abs / (torch.linalg.norm(noisy_core) + 1e-30)
    # core white should also be properly whitened elementwise
    r_core = diff_core / (sigma_core + 1e-30)
    core_white = torch.linalg.norm(r_core)

    # core chi^2 and z stats
    abs_r2_core = torch.abs(r_core) ** 2
    core_Nt = int(abs_r2_core.numel())
    core_chi2_tau = torch.sum(abs_r2_core)
    core_chi2_tau_per_point = core_chi2_tau / max(core_Nt, 1)
    if torch.is_complex(r_core):
        core_z_mean_real = torch.mean(r_core.real)
        core_z_mean_imag = torch.mean(r_core.imag)
        core_z_std_real = torch.std(r_core.real, unbiased=False)
        core_z_std_imag = torch.std(r_core.imag, unbiased=False)
    else:
        core_z_mean_real = torch.mean(r_core)
        core_z_mean_imag = torch.tensor(0.0, dtype=torch.float64, device=r_core.device)
        core_z_std_real = torch.std(r_core, unbiased=False)
        core_z_std_imag = torch.tensor(0.0, dtype=torch.float64, device=r_core.device)
    core_z_rms = torch.sqrt(core_chi2_tau_per_point)

    return {
        "deltaGtau_abs": float(delta_abs.detach().cpu().item()),
        "deltaGtau_rel": float(delta_rel.detach().cpu().item()),
        "deltaGtau_white": float(delta_white.detach().cpu().item()),
        "chi2_tau": float(chi2_tau.detach().cpu().item()),
        "chi2_tau_per_point": float(chi2_tau_per_point.detach().cpu().item()),
        "z_mean_real": float(z_mean_real.detach().cpu().item()),
        "z_mean_imag": float(z_mean_imag.detach().cpu().item()),
        "z_std_real": float(z_std_real.detach().cpu().item()),
        "z_std_imag": float(z_std_imag.detach().cpu().item()),
        "z_rms": float(z_rms.detach().cpu().item()),
        "core_abs": float(core_abs.detach().cpu().item()),
        "core_rel": float(core_rel.detach().cpu().item()),
        "core_white": float(core_white.detach().cpu().item()),
        "core_N": N,
        "core_chi2_tau": float(core_chi2_tau.detach().cpu().item()),
        "core_chi2_tau_per_point": float(core_chi2_tau_per_point.detach().cpu().item()),
        "core_z_mean_real": float(core_z_mean_real.detach().cpu().item()),
        "core_z_mean_imag": float(core_z_mean_imag.detach().cpu().item()),
        "core_z_std_real": float(core_z_std_real.detach().cpu().item()),
        "core_z_std_imag": float(core_z_std_imag.detach().cpu().item()),
        "core_z_rms": float(core_z_rms.detach().cpu().item()),
        "topk_idx": idx.detach().cpu().to(torch.int64).numpy(),
        "topk_tau": tau[idx].detach().cpu().to(torch.float64).numpy(),
        "topk_abs_r": vals.detach().cpu().to(torch.float64).numpy(),
        # keep the old debug field for continuity, but interpret it explicitly
        "topk_abs_r_over_mean_sigma": (abs_r[idx] / (torch.mean(sigma64) + 1e-30)).detach().cpu().to(torch.float64).numpy(),
    }
