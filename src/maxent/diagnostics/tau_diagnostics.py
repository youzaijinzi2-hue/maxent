from __future__ import annotations

import math
from typing import Dict, Tuple, Optional

import torch
import logging

from maxent.tau_metrics import compute_tau_metrics


def fit_c_tau(G_tau_rec: torch.Tensor, G_tau_noisy: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """
    best-fit scale in τ:
     c_tau = argmin_c || c*G_tau_rec - G_tau_noisy ||_2
    returns: scalar tensor (float64)
    """
    rec_flat = G_tau_rec.reshape(-1).to(torch.complex128)
    obs_flat = G_tau_noisy.reshape(-1).to(torch.complex128)
    num = torch.vdot(rec_flat, obs_flat).real
    den = torch.vdot(rec_flat, rec_flat).real
    return (num / (den + eps)).to(torch.float64)


def compute_tau_metrics_pair(
    *,
    G_tau_rec: torch.Tensor,
    G_tau_rec_scaled: torch.Tensor,
    G_tau_noisy: torch.Tensor,
    tau: torch.Tensor,
    beta: float,
    sigma_tau: torch.Tensor,
    tau_margin: float = 5.0,
    topk: int = 8,
) -> Tuple[Dict, Dict]:
    """
    Compute τ-metrics for both raw and scaled reconstructions.
    Returns: (tau_m_raw, tau_m_scaled)
    """
    tau_m_raw = compute_tau_metrics(
        G_tau_rec=G_tau_rec,
        G_tau_noisy=G_tau_noisy,
        tau=tau,
        beta=beta,
        sigma_tau=sigma_tau,
        tau_margin=tau_margin,
        topk=topk,
    )
    tau_m = compute_tau_metrics(
        G_tau_rec=G_tau_rec_scaled,
        G_tau_noisy=G_tau_noisy,
        tau=tau,
        beta=beta,
        sigma_tau=sigma_tau,
        tau_margin=tau_margin,
        topk=topk,
    )
    return tau_m_raw, tau_m


def print_tau_diagnostics_summary(
    *,
    tau_m_raw: Dict,
    tau_m: Dict,
    G_tau_rec: torch.Tensor,
    G_tau_rec_scaled: torch.Tensor,
    G_tau_noisy: torch.Tensor,
    N_tau: int,
    log: Optional[logging.Logger] = None,
)-> None:
    """
    Print τ diagnostics in the same style as the runner previously did.
    Keep this in one place so the runner stays readable.
    """
    # logger fallback: if not provided, use a module logger (won't duplicate handlers)
    if log is None:
        log = logging.getLogger("maxent")
        
    # quick comparison: raw vs scaled chi2/Nt
    log.info(f"[tau-raw]    chi2/Nt = {tau_m_raw['chi2_tau_per_point']:.6g}  z_rms={tau_m_raw['z_rms']:.6g}  z_mean={tau_m_raw['z_mean_real']:.6g}")
    log.info(f"[tau-scaled] chi2/Nt = {tau_m['chi2_tau_per_point']:.6g}  z_rms={tau_m['z_rms']:.6g}  z_mean={tau_m['z_mean_real']:.6g}")

    # empirical sigma (debug)
    log.debug(f"[DEBUG] sigma_tau^emp (raw):    {float(torch.linalg.norm((G_tau_rec - G_tau_noisy)) / math.sqrt(N_tau)):.6g}")
    log.debug(f"[DEBUG] sigma_tau^emp (scaled): {float(torch.linalg.norm((G_tau_rec_scaled - G_tau_noisy)) / math.sqrt(N_tau)):.6g}")
 
    # legacy L2-style metrics
    # print(f"[metric] ΔG_tau_abs   = {tau_m['deltaGtau_abs']:.6e}  (L2 in τ space)")
    # print(f"[metric] ΔG_tau_rel   = {tau_m['deltaGtau_rel']:.6e}  (relative L2 in τ space)")
    # print(f"[metric] ΔG_tau_white = {tau_m['deltaGtau_white']:.6e}  (||z||_2 where z=(ΔG/σ))")

    # PASS/FAIL: chi2/Nt ~ 1 and z mean~0, std~1
    log.info(f"[tau-whiten-scaled] chi2/Nt = {tau_m['chi2_tau_per_point']:.6g}")
    log.info(f"[tau-whiten-scaled] z_rms   = {tau_m['z_rms']:.6g}  (sqrt(chi2/Nt))")
    log.info(f"[tau-whiten-scaled] z mean/std (real) = {tau_m['z_mean_real']:.6g} / {tau_m['z_std_real']:.6g}")
    if abs(tau_m.get("z_mean_imag", 0.0)) > 0.0 or abs(tau_m.get("z_std_imag", 0.0)) > 0.0:
        log.info(f"[tau-whiten-scaled] z mean/std (imag) = {tau_m['z_mean_imag']:.6g} / {tau_m['z_std_imag']:.6g}")

    # core (exclude edges in τ)
    log.info(f"[tau-core-scaled] chi2/Nt = {tau_m['core_chi2_tau_per_point']:.6g}  N={tau_m['core_N']}")
    log.info(f"[tau-core-scaled] z_rms   = {tau_m['core_z_rms']:.6g}")
    log.info(f"[tau-core-scaled] z mean/std (real) = {tau_m['core_z_mean_real']:.6g} / {tau_m['core_z_std_real']:.6g}")
    if abs(tau_m.get("core_z_mean_imag", 0.0)) > 0.0 or abs(tau_m.get("core_z_std_imag", 0.0)) > 0.0:
        log.info(f"[tau-core-scaled] z mean/std (imag) = {tau_m['core_z_mean_imag']:.6g} / {tau_m['core_z_std_imag']:.6g}")

    log.info(
        f"[metric-core] abs, rel, white = {tau_m['core_abs']} {tau_m['core_rel']} {tau_m['core_white']} N={tau_m['core_N']}"
    )
