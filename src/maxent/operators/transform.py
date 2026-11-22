from __future__ import annotations
from typing import Optional
import torch

# ============================================================
# ΔGτ = || Gτ[A_rec] - Gτ_noisy ||  (consistent with forward.py)
#
# forward kernel (fermionic):
#   G(τ) = ∫ dω  exp(-τ ω) / (1 + exp(-β ω)) * A(ω)
#
# and compute Gτ_noisy from Matsubara data:
#   G(τ) ≈ (1/β) Σ_n exp(-i ω_n τ) G(iω_n)
#
# Then report:
#   deltaGtau_abs   = ||Gτ_rec - Gτ_noisy||_2
#   deltaGtau_rel   = deltaGtau_abs / ||Gτ_noisy||_2
#   deltaGtau_white = ||(Gτ_rec - Gτ_noisy)/σ(τ)||_2
# ============================================================
def giw_to_gtau_pos(G_iw_pos: torch.Tensor, wn_pos: torch.Tensor, tau: torch.Tensor, beta: float) -> torch.Tensor:
    """
    G_iw_pos: (Nw,) complex  (positive fermionic Matsubara only)
    wn_pos  : (Nw,) real
    tau     : (Nt,) real
    """
    G_iw_pos = G_iw_pos.to(torch.complex128)
    wn_pos = wn_pos.to(torch.float64)
    tau = tau.to(torch.float64)

    phase = tau[:, None] * wn_pos[None, :]  # (Nt, Nw)
    K_cos = (2.0 / beta) * torch.cos(phase)
    K_sin = (2.0 / beta) * torch.sin(phase)

    # G(tau) = sum_n [K_cos * ReG + K_sin * ImG]
    return (K_cos * G_iw_pos.real + K_sin * G_iw_pos.imag).sum(dim=1).to(torch.float64)

def forward_gtau_from_A(A_w: torch.Tensor, omega_w: torch.Tensor, tau: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Implements the same physics as forward.py (but directly integrates over ω grid):
    G(τ) = ∫ dω K(τ,ω) A(ω),  K(τ,ω)=exp(-τω)/(1+exp(-βω))
    """
    A_w = A_w.to(torch.float64)
    omega_w = omega_w.to(torch.float64)
    tau = tau.to(torch.float64)
    tau_b = tau[:, None]              # (Nt, 1)
    omega_b = omega_w[None, :]        # (1, Nw)
    beta_f = float(beta)

    # Numerically stable fermionic kernel:
    #   exp(-tau*omega) / (1 + exp(-beta*omega))
    # = exp(-tau*omega + logsigmoid(beta*omega))
    log_kernel = -tau_b * omega_b + torch.nn.functional.logsigmoid(beta_f * omega_b)
    K = torch.exp(log_kernel)  # (Nt, Nw)
    # Trapz over ω -> (Nt,)
    G_tau = torch.trapz(K * A_w[None, :], omega_w, dim=1)
    return (-G_tau).to(torch.float64)   # ← マイナス & 実数

def giw_from_A(A_w: torch.Tensor, omega: torch.Tensor, wn: torch.Tensor) -> torch.Tensor:
    """
    Spectral representation:
        G(iω_n) = ∫ dω A(ω) / (iω_n - ω)
    Assumes:
        - A_w real-valued (>=0 typically)
        - omega real grid (same shape as A_w)
        - wn real positive Matsubara frequencies (or any real list)
    Returns:
        G_iw complex tensor of shape (Nw,)
    """
    # Ensure dtypes/devices
    A_w = A_w.to(dtype=torch.float64)
    omega = omega.to(dtype=torch.float64)
    wn = wn.to(dtype=torch.float64)

    # (Nw, Nω) integrand: A(ω) / (iω_n - ω)
    denom = (1j * wn[:, None]).to(dtype=torch.complex128) - omega[None, :].to(dtype=torch.complex128)
    integrand = A_w[None, :].to(dtype=torch.complex128) / denom
    G_iw = torch.trapz(integrand, omega, dim=1)  # integrate over ω
    return G_iw  # complex128


def kernel_tau_iw(wn_pos: torch.Tensor, tau: torch.Tensor, beta: float, *, dtype=torch.float64):
    """
    Build kernels to map positive-fermionic Matsubara Green's function G(iw_n) (n>=0)
    to real-imaginary-time G(tau) using conjugation symmetry:

        G(tau) = (2/beta) * Re[ sum_{n>=0} exp(-i wn tau) G(iwn) ]
              = (2/beta) * sum_{n>=0} [ cos(wn tau) Re G_n + sin(wn tau) Im G_n ].

    Returns:
        K_cos: (N_tau, Nw) real
        K_sin: (N_tau, Nw) real
    """
    wn_pos = wn_pos.to(dtype=dtype)
    tau = tau.to(dtype=dtype)
    phase = tau[:, None] * wn_pos[None, :]  # (N_tau, Nw)

    K_cos = (2.0 / beta) * torch.cos(phase)
    K_sin = (2.0 / beta) * torch.sin(phase)
    return K_cos, K_sin

def sigma_tau_from_sigma_iw(
    wn_pos: torch.Tensor,
    tau: torch.Tensor,
    beta: float,
    sigma_iw: torch.Tensor | float,
    *,
    cov_iw: Optional[torch.Tensor] = None,
    eps: float = 1e-300,
) -> torch.Tensor:
    r"""
    Propagate iω-space noise to τ-space.

    We assume a linear transform (for positive Matsubara freqs):
        G_tau = K @ G_iw

    If iω noise is independent with std vector σ_iw (diagonal covariance),
        Var[G_tau(i)] = sum_j |K_{ij}|^2 * σ_iw(j)^2

    If a full covariance cov_iw is provided (Hermitian, shape (Nw,Nw)),
        Cov_tau = K cov_iw K^H, and we return sqrt(diag(Cov_tau)).

    Args
    ----
    wn_pos : (Nw,) positive Matsubara frequencies
    tau    : (Nt,) tau grid (typically in (0,beta))
    beta   : inverse temperature
    sigma_iw: scalar or (Nw,) tensor
    cov_iw : optional full covariance in iω, shape (Nw,Nw)
    eps    : numerical floor
    """
    # kernel_tau_iw returns (K_cos, K_sin) real tensors: (Nt, Nw), (Nt, Nw)
    K_cos, K_sin = kernel_tau_iw(tau=tau, wn_pos=wn_pos, beta=beta)
        
    # --- full covariance path ---
    if cov_iw is not None:
        raise NotImplementedError(
            "cov_iw path is not implemented for (K_cos, K_sin) representation. "
            "Use diagonal sigma_iw (scalar or vector) for now."
        )

    # --- diagonal (independent) path ---
    if not isinstance(sigma_iw, torch.Tensor):
        sigma_vec = torch.full((K_cos.shape[1],), float(sigma_iw), device=K_cos.device, dtype=torch.float64)
    else:
        sigma_vec = sigma_iw.to(device=K_cos.device, dtype=torch.float64)
        if sigma_vec.ndim == 0:
            sigma_vec = torch.full((K_cos.shape[1],), float(sigma_vec.item()), device=K_cos.device, dtype=torch.float64)
        if sigma_vec.shape != (K_cos.shape[1],):
            raise ValueError(
                f"sigma_iw must be scalar or shape (Nw,), got {tuple(sigma_vec.shape)} vs Nw={K_cos.shape[1]}"
            )

    # Your noise injection uses: noise = (randn + i randn) * (sigma / sqrt(2))
    # => std(Re)=std(Im)=sigma/sqrt(2)
    # Thus Var contribution uses (sigma_vec^2)/2 for both Re and Im parts.
    sig2_half = (sigma_vec ** 2) * 0.5
    var_tau = (K_cos.to(torch.float64) ** 2) @ sig2_half \
            + (K_sin.to(torch.float64) ** 2) @ sig2_half
    return torch.sqrt(torch.clamp(var_tau, min=0.0) + eps).to(torch.float64)

def sigma_tau_from_Sigma_tau(Sigma_tau: torch.Tensor) -> torch.Tensor:
    """Return per-tau std vector from covariance matrix."""
    return torch.sqrt(torch.clamp(torch.diag(Sigma_tau), min=0.0))
