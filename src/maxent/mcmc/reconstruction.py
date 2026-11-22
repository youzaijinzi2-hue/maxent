from __future__ import annotations

from typing import Optional, Tuple

import torch


def reconstruct_posterior_mean(
    *,
    device: torch.device,
    D_w: torch.Tensor,
    omega: torch.Tensor,
    n_means: torch.Tensor,
    H_means: torch.Tensor,
    w_alpha_cached: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct posterior mean spectrum from replica means."""
    if w_alpha_cached is not None:
        w = w_alpha_cached.to(device=device, dtype=torch.float64)
        w = w / (w.sum() + 1e-300)
    else:
        w = torch.softmax(-0.5 * H_means.to(torch.float64), dim=0)

    n_post = torch.matmul(w, n_means)
    D_vals = D_w.to(device=device, dtype=torch.float64)
    A_est = n_post.to(torch.float64) * D_vals
    area = torch.trapz(A_est, omega)
    return A_est, w, area


def credible_interval_from_samples(
    *,
    n_samples: torch.Tensor,
    D_w: torch.Tensor,
    omega: torch.Tensor,
    device: torch.device,
    w: torch.Tensor,
    adapt_end: int,
    q_low: float = 0.025,
    q_high: float = 0.975,
    max_samples: int = 20000,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Estimate weighted credible interval by resampling post-burnin chains."""
    assert w.ndim == 1, "w must be (R,)"
    R, L, _ = n_samples.shape

    if adapt_end >= L:
        adapt_end = 0

    post_len = L - adapt_end
    total_available = int(R * post_len)
    if total_available <= 0:
        raise RuntimeError("No post-burnin samples available for CI estimation.")

    M = min(int(max_samples), total_available)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    w = w.to(device=device, dtype=torch.float64)
    w = w / (w.sum() + 1e-300)
    ridx = torch.multinomial(w, num_samples=M, replacement=True, generator=gen)
    tidx = torch.randint(low=adapt_end, high=L, size=(M,), device=device, generator=gen)

    n_samp = n_samples[ridx, tidx, :]
    D_vals = D_w.to(device=device, dtype=torch.float64)
    A = n_samp.to(torch.float64) * D_vals[None, :]

    Z = torch.trapz(A, omega, dim=1)
    A_low = torch.quantile(A, q_low, dim=0)
    A_high = torch.quantile(A, q_high, dim=0)

    qs = torch.tensor([0.025, 0.5, 0.975], device=device, dtype=torch.float64)
    Zq = torch.quantile(Z.to(torch.float64), qs)

    ci_diag = {
        "ci_method": "mixture_resample_quantile",
        "ci_samples": int(M),
        "adapt_end": int(adapt_end),
        "Z_q025": float(Zq[0].detach().cpu()),
        "Z_q50": float(Zq[1].detach().cpu()),
        "Z_q975": float(Zq[2].detach().cpu()),
        "Z_min": float(Z.min().detach().cpu()),
        "Z_max": float(Z.max().detach().cpu()),
    }
    return A_low, A_high, ci_diag
