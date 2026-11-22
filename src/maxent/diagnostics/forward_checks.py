import torch
from maxent.operators.transform import (
    giw_from_A,
    forward_gtau_from_A,
)
import math

def giw_to_gtau_fullsum(G_iw_pos: torch.Tensor, wn_pos: torch.Tensor, tau: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Build G(τ) from Matsubara sum using both +wn and -wn.
    If A(ω) is real, then G(-iω_n) = conj(G(iω_n)). We exploit this.

    G(τ) = (1/β) Σ_n exp(-iω_n τ) G(iω_n)

    Inputs:
        - G_iw_pos: G(iω_n) for wn_pos (assumed positive frequencies), shape (Nw,)
        - wn_pos: positive Matsubara frequencies, shape (Nw,)
        - tau: τ grid, shape (Nτ,)
        - beta: scalar
    Returns:
        G_tau real64 tensor (Nτ,)  (we take real part at the end, consistent with your current metric style)
    """
    G_iw_pos = G_iw_pos.to(dtype=torch.complex128)
    wn_pos = wn_pos.to(dtype=torch.float64)
    tau = tau.to(dtype=torch.float64)

    # Construct negative frequencies using conjugation symmetry
    wn_neg = -torch.flip(wn_pos, dims=[0])
    G_iw_neg = torch.conj(torch.flip(G_iw_pos, dims=[0]))

    wn_full = torch.cat([wn_neg, wn_pos], dim=0).to(dtype=torch.float64)                  # (2Nw,)
    G_full  = torch.cat([G_iw_neg, G_iw_pos], dim=0).to(dtype=torch.complex128)           # (2Nw,)

    phase = torch.exp((-1j) * wn_full[None, :].to(dtype=torch.complex128) * tau[:, None].to(dtype=torch.complex128))
    G_tau = (phase * G_full[None, :]).sum(dim=1) / beta  # complex128
    return G_tau.real.to(dtype=torch.float64)

def forward_self_consistency_test(A_w: torch.Tensor, omega: torch.Tensor, tau: torch.Tensor, beta: float, wn_pos: torch.Tensor):
    """
    Compare:
      (1) G_tau_from_omega = forward_gtau_from_A(A_w, omega, tau, beta)
      (2) G_tau_from_iw    = giw_to_gtau_fullsum( giw_from_A(A_w, omega, wn_pos), wn_pos, tau, beta )

    Prints diagnostic metrics and top residual locations.
    """
    # Path (1): omega -> tau
    G_tau_omega = forward_gtau_from_A(A_w, omega, tau, beta).to(dtype=torch.float64)

    # Path (2): omega -> iw -> tau
    G_iw_pos = giw_from_A(A_w, omega, wn_pos)  # complex128
    G_tau_iw = giw_to_gtau_fullsum(G_iw_pos, wn_pos, tau, beta).to(dtype=torch.float64)
    
    r = (G_tau_omega - G_tau_iw)
    abs_L2 = torch.linalg.norm(r).item()
    denom = (torch.linalg.norm(G_tau_omega) + 1e-30).item()
    rel_L2 = abs_L2 / denom

    print("\n===== Forward Self-Consistency Test (MaxEnt-free) =====")
    print(f"[self] ||G_tau(omega) - G_tau(iw)||_2      = {abs_L2:.6e}")
    print(f"[self] relative L2 (vs ||G_tau(omega)||_2)  = {rel_L2:.6e}")
    print(f"[self] ||G_tau(omega)||_2                   = {torch.linalg.norm(G_tau_omega).item():.6e}")
    print(f"[self] ||G_tau(iw)||_2                      = {torch.linalg.norm(G_tau_iw).item():.6e}")

    # Where is the mismatch concentrated?
    abs_r = torch.abs(r)
    k = min(10, abs_r.numel())
    idx = torch.topk(abs_r, k=k).indices
    print("[self] top |r(tau)|:")
    for i in idx.tolist():
        print(f"  {i:3d}  tau={float(tau[i]):.6g}  |r|={float(abs_r[i]):.6e}")

    return G_tau_omega, G_tau_iw, r
