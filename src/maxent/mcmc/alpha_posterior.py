from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch


def trapz_weights(grid: torch.Tensor) -> torch.Tensor:
    """Return trapezoidal-rule weights matching torch.trapz on a 1D grid."""
    g = grid.to(torch.float64)
    n = g.numel()
    if n == 0:
        return g
    if n == 1:
        return torch.ones_like(g, dtype=torch.float64)
    dx = g[1:] - g[:-1]
    w = torch.empty_like(g, dtype=torch.float64)
    w[0] = dx[0] * 0.5
    w[-1] = dx[-1] * 0.5
    if n > 2:
        w[1:-1] = (dx[:-1] + dx[1:]) * 0.5
    return w


def alpha_quadrature_widths(alpha_sorted: torch.Tensor, eps: float = 1e-300) -> torch.Tensor:
    """Bin widths for approximating integral f(alpha) d alpha on a sorted alpha grid."""
    a = alpha_sorted.to(torch.float64)
    n = a.numel()
    if n == 0:
        return a
    if n == 1:
        return torch.ones_like(a, dtype=torch.float64)
    da = torch.empty_like(a, dtype=torch.float64)
    da[0] = 0.5 * (a[1] - a[0])
    da[-1] = 0.5 * (a[-1] - a[-2])
    if n > 2:
        da[1:-1] = 0.5 * (a[2:] - a[:-2])
    return da.clamp_min(eps)


def compute_lambdas_jerrell(
    *,
    operators: Any,
    x_grid: torch.Tensor,
    device: torch.device,
    A_hat: torch.Tensor,
    chunk_size: int = 256,
    eps: float = 1e-300,
) -> torch.Tensor:
    """Compute Occam-factor eigenvalues in Jerrell/Bryan form."""
    K_tau_x = operators._K_tau_x(x_grid.unsqueeze(0))[0].to(
        device=device, dtype=torch.float64
    )  # (T,D)
    T, D = K_tau_x.shape
    wx = trapz_weights(x_grid).to(device=device, dtype=torch.float64)  # (D,)

    A = A_hat.to(device=device, dtype=torch.float64).clamp_min(eps)
    sqrtA = torch.sqrt(A)

    whiten_name = None
    for cand in ("whiten_tau", "whiten", "_whiten_tau"):
        if hasattr(operators, cand) and callable(getattr(operators, cand)):
            whiten_name = cand
            break

    if whiten_name is not None:
        whiten = getattr(operators, whiten_name)
        Kx = K_tau_x * wx[None, :]  # (T,D)
        WKx_BT = whiten(Kx.transpose(0, 1).contiguous())  # (D,T)
        if WKx_BT.shape != (D, T):
            raise ValueError(
                f"{whiten_name} returned {tuple(WKx_BT.shape)}, expected {(D, T)}"
            )
        B_TD = WKx_BT.transpose(0, 1).contiguous()  # (T,D)
    else:
        B_cols = torch.empty((D, T), device=device, dtype=torch.float64)  # (D,T)
        for j0 in range(0, D, chunk_size):
            j1 = min(D, j0 + chunk_size)
            m = j1 - j0

            K_rep = K_tau_x[None, :, :].repeat(m, 1, 1)  # (m,T,D)
            n0 = torch.zeros((m, D), device=device, dtype=torch.float64)
            h0 = operators._h(n0, K_rep)  # (m,T)

            nb = torch.zeros((m, D), device=device, dtype=torch.float64)
            idx = torch.arange(m, device=device)
            nb[idx, j0 + idx] = 1.0
            hb = operators._h(nb, K_rep)  # (m,T)

            WKx = (hb - h0).to(torch.float64)  # (m,T)
            B_cols[j0:j1, :] = WKx
        B_TD = B_cols.transpose(0, 1).contiguous()  # (T,D)

    C = B_TD * sqrtA[None, :]
    s = torch.linalg.svdvals(C)
    return (s * s).contiguous()


def alpha_posterior_weights_bryan_occam(
    *,
    alphas: torch.Tensor,
    H_term: torch.Tensor,
    S_term: torch.Tensor,
    n_means: torch.Tensor,
    compute_lambdas: Callable[[torch.Tensor], torch.Tensor],
    prior: str = "jeffreys",
    eps: float = 1e-300,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bryan/Laplace-style alpha posterior weights with Occam factor."""
    a = alphas.to(torch.float64)
    Hm = H_term.to(torch.float64)
    Sm = S_term.to(torch.float64)
    Ahat = n_means.to(torch.float64)

    order = torch.argsort(a)
    a = a[order]
    Hm = Hm[order]
    Sm = Sm[order]
    Ahat = Ahat[order]

    if prior == "jeffreys":
        logprior = -torch.log(a.clamp_min(eps))
    elif prior == "flat":
        logprior = torch.zeros_like(a)
    else:
        raise ValueError(f"unknown prior: {prior}")

    occam = torch.empty_like(a)
    with torch.no_grad():
        for k in range(a.numel()):
            alpha_k = a[k].clamp_min(eps)
            lambdas = compute_lambdas(Ahat[k]).to(torch.float64).clamp_min(0.0)
            occam[k] = 0.5 * torch.sum(
                torch.log(alpha_k) - torch.log((alpha_k + lambdas).clamp_min(eps))
            )

    log_like = a * Sm - Hm
    logP = logprior + log_like + occam
    log_measure = torch.log(alpha_quadrature_widths(a, eps=eps))
    log_mass = logP + log_measure
    logZ_alpha = torch.logsumexp(log_mass, dim=0)
    w_sorted = torch.softmax(log_mass, dim=0)

    w = torch.zeros_like(w_sorted)
    w[order] = w_sorted
    logP_unsorted = torch.zeros_like(log_mass)
    logP_unsorted[order] = log_mass
    return (
        w.to(torch.float64),
        logP_unsorted.to(torch.float64),
        logZ_alpha.to(torch.float64),
    )


def alpha_posterior_weights_ti(
    *,
    alphas: torch.Tensor,
    S_mean: torch.Tensor,
    prior: str = "jeffreys",
    eps: float = 1e-300,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TI-based alpha posterior weights."""
    a = alphas.to(torch.float64)
    Sm = S_mean.to(torch.float64)
    order = torch.argsort(a)
    a = a[order]
    Sm = Sm[order]

    da = a[1:] - a[:-1]
    integrand = 0.5 * (Sm[1:] + Sm[:-1]) * da
    logZ = torch.zeros_like(a)
    logZ[1:] = torch.cumsum(integrand, dim=0)

    if prior == "jeffreys":
        logprior = -torch.log(a.clamp_min(eps))
    elif prior == "flat":
        logprior = torch.zeros_like(a)
    else:
        raise ValueError(f"unknown prior: {prior}")

    logP = logZ + logprior
    log_measure = torch.log(alpha_quadrature_widths(a, eps=eps))
    log_mass = logP + log_measure
    logZ_alpha = torch.logsumexp(log_mass, dim=0)
    w_sorted = torch.softmax(log_mass, dim=0)

    w = torch.zeros_like(w_sorted)
    w[order] = w_sorted
    logP_unsorted = torch.zeros_like(log_mass)
    logP_unsorted[order] = log_mass
    return (
        w.to(torch.float64),
        logP_unsorted.to(torch.float64),
        logZ_alpha.to(torch.float64),
    )


def validate_alpha_weights(
    w: torch.Tensor,
    logp: torch.Tensor,
    *,
    label: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate and normalize alpha weights."""
    w = w.to(torch.float64)
    logp = logp.to(torch.float64)
    if (not torch.isfinite(w).all()) or (not torch.isfinite(logp).all()):
        raise FloatingPointError(f"non-finite alpha weights/logP from {label}")
    s = w.sum()
    if (not torch.isfinite(s)) or (s <= 0):
        raise FloatingPointError(f"invalid alpha-weight sum from {label}: {float(s)}")
    w = w / (s + 1e-300)
    return w, logp


def build_alpha_reference_spectra(
    *,
    config: Any,
    adapt_end: int,
    n_means: torch.Tensor,
    n_samples: torch.Tensor,
    H_samples: torch.Tensor,
    S_samples: torch.Tensor,
    alphas: torch.Tensor,
    device: torch.device,
    diagnostics: dict,
) -> torch.Tensor:
    """
    Build reference spectra A_hat(alpha) used in Occam-factor evaluation.
    """
    mode = str(getattr(config, "alpha_reference_spectrum", "map")).lower()
    diagnostics["alpha_ref_mode"] = mode

    if mode == "mean":
        return n_means.to(torch.float64)

    post_n = n_samples[:, adapt_end:, :].to(torch.float64)  # (R, Lp, D)
    post_H = H_samples[:, adapt_end:].to(torch.float64)  # (R, Lp)
    post_S = S_samples[:, adapt_end:].to(torch.float64)  # (R, Lp)
    if post_n.shape[1] == 0:
        diagnostics["alpha_ref_warning"] = "no_postburnin_samples_fallback_to_mean"
        return n_means.to(torch.float64)

    logp = alphas.to(torch.float64)[:, None] * post_S - post_H  # (R, Lp)
    map_idx = torch.argmax(logp, dim=1)  # (R,)
    ridx = torch.arange(post_n.shape[0], device=device)
    n_map = post_n[ridx, map_idx, :]  # (R, D)
    map_lp = logp[ridx, map_idx]  # (R,)

    diagnostics["alpha_ref_map_idx"] = map_idx.detach().cpu().numpy().astype(np.int64)
    diagnostics["alpha_ref_map_logpost"] = map_lp.detach().cpu().numpy()
    diagnostics["alpha_ref_map_to_mean_rel_l2"] = (
        torch.linalg.norm((n_map - n_means.to(torch.float64)).reshape(n_map.shape[0], -1), dim=1)
        / torch.linalg.norm(n_means.to(torch.float64).reshape(n_map.shape[0], -1), dim=1).clamp_min(1e-300)
    ).detach().cpu().numpy()

    if mode == "map":
        return n_map

    if mode != "map_local_mc":
        diagnostics["alpha_ref_warning"] = f"unknown_mode_{mode}_fallback_to_map"
        return n_map

    n_mc = max(1, int(getattr(config, "alpha_map_local_samples", 64)))
    lp_window = float(getattr(config, "alpha_map_local_logp_window", 5.0))
    temp = max(1e-12, float(getattr(config, "alpha_map_local_temperature", 1.0)))
    seed = int(getattr(config, "alpha_map_local_seed", 0))
    diagnostics["alpha_ref_map_local_samples"] = int(n_mc)
    diagnostics["alpha_ref_map_local_window"] = float(lp_window)
    diagnostics["alpha_ref_map_local_temperature"] = float(temp)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    R = post_n.shape[0]
    n_ref = torch.empty_like(n_map)
    support_sizes = torch.empty((R,), device=device, dtype=torch.int64)

    with torch.no_grad():
        for k in range(R):
            lp_k = logp[k]  # (Lp,)
            lp_max = map_lp[k]
            keep = lp_k >= (lp_max - lp_window)
            idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                idx = map_idx[k : k + 1]
            support_sizes[k] = idx.numel()

            lp_sel = lp_k[idx]
            logits = (lp_sel - torch.max(lp_sel)) / temp
            probs = torch.softmax(logits, dim=0)

            if idx.numel() == 1:
                draw_idx = idx.repeat(n_mc)
            else:
                draw_local = torch.multinomial(
                    probs, num_samples=n_mc, replacement=True, generator=gen
                )
                draw_idx = idx[draw_local]
            n_ref[k] = post_n[k, draw_idx, :].mean(dim=0)

    diagnostics["alpha_ref_map_local_support_size"] = (
        support_sizes.detach().cpu().numpy().astype(np.int64)
    )
    diagnostics["alpha_ref_local_to_map_rel_l2"] = (
        torch.linalg.norm((n_ref - n_map).reshape(R, -1), dim=1)
        / torch.linalg.norm(n_map.reshape(R, -1), dim=1).clamp_min(1e-300)
    ).detach().cpu().numpy()
    return n_ref


def select_alpha_weights(
    *,
    config: Any,
    alphas: torch.Tensor,
    H_means: torch.Tensor,
    S_means: torch.Tensor,
    n_means: torch.Tensor,
    n_samples: torch.Tensor,
    H_samples: torch.Tensor,
    S_samples: torch.Tensor,
    adapt_end: int,
    device: torch.device,
    compute_lambdas: Callable[[torch.Tensor], torch.Tensor],
    diagnostics: dict,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Compute/compare alpha posteriors and choose final weights."""
    requested_method = str(getattr(config, "alpha_weight_method", "auto")).lower()
    prior_bryan = str(getattr(config, "alpha_prior_bryan", "flat")).lower()
    prior_ti = str(getattr(config, "alpha_prior_ti", "flat")).lower()
    bryan_main_term = str(getattr(config, "alpha_bryan_main_term", "map")).lower()

    diagnostics["alpha_weight_method_requested"] = requested_method
    diagnostics["alpha_weight_prior_bryan"] = prior_bryan
    diagnostics["alpha_weight_prior_ti"] = prior_ti
    diagnostics["alpha_weight_measure"] = "dalpha_quadrature"
    diagnostics["alpha_bryan_main_term_requested"] = bryan_main_term

    H_bryan = H_means.to(torch.float64)
    S_bryan = S_means.to(torch.float64)
    if bryan_main_term == "mean":
        diagnostics["alpha_bryan_main_term"] = "mean"
    else:
        post_H = H_samples[:, adapt_end:].to(torch.float64)
        post_S = S_samples[:, adapt_end:].to(torch.float64)
        if post_H.shape[1] == 0:
            diagnostics["alpha_bryan_main_term"] = "mean_fallback_no_postburnin"
        else:
            logp_post = alphas.to(torch.float64)[:, None] * post_S - post_H
            ridx = torch.arange(post_H.shape[0], device=device)
            map_idx_main = torch.argmax(logp_post, dim=1)
            H_map = post_H[ridx, map_idx_main]
            S_map = post_S[ridx, map_idx_main]
            H_bryan = H_map
            S_bryan = S_map
            diagnostics["alpha_bryan_main_term"] = "map"
            diagnostics["alpha_bryan_main_term_map_idx"] = (
                map_idx_main.detach().cpu().numpy().astype(np.int64)
            )
            diagnostics["alpha_bryan_main_term_rel_l2_H_map_vs_mean"] = float(
                (
                    torch.linalg.norm(H_map - H_means.to(torch.float64))
                    / torch.linalg.norm(H_means.to(torch.float64)).clamp_min(1e-300)
                ).detach().cpu()
            )
            diagnostics["alpha_bryan_main_term_rel_l2_S_map_vs_mean"] = float(
                (
                    torch.linalg.norm(S_map - S_means.to(torch.float64))
                    / torch.linalg.norm(S_means.to(torch.float64)).clamp_min(1e-300)
                ).detach().cpu()
            )

    n_ref_for_occam = build_alpha_reference_spectra(
        config=config,
        adapt_end=adapt_end,
        n_means=n_means,
        n_samples=n_samples,
        H_samples=H_samples,
        S_samples=S_samples,
        alphas=alphas,
        device=device,
        diagnostics=diagnostics,
    )

    w_bryan = None
    logP_bryan = None
    logZ_bryan = None
    w_ti = None
    logP_ti = None
    logZ_ti = None

    try:
        wb, lpb, lzb = alpha_posterior_weights_bryan_occam(
            alphas=alphas,
            H_term=H_bryan,
            S_term=S_bryan,
            n_means=n_ref_for_occam,
            compute_lambdas=compute_lambdas,
            prior=prior_bryan,
        )
        w_bryan, logP_bryan = validate_alpha_weights(wb, lpb, label="bryan_laplace")
        logZ_bryan = lzb.to(torch.float64)
        diagnostics["w_alpha_bryan"] = w_bryan
        diagnostics["logP_alpha_bryan"] = logP_bryan
        diagnostics["logZ_alpha_bryan"] = float(logZ_bryan.detach().cpu())
    except Exception as e:
        diagnostics["alpha_weight_error_bryan"] = f"{type(e).__name__}: {e}"

    try:
        wt, lpt, lzt = alpha_posterior_weights_ti(
            alphas=alphas,
            S_mean=S_means,
            prior=prior_ti,
        )
        w_ti, logP_ti = validate_alpha_weights(wt, lpt, label="ti")
        logZ_ti = lzt.to(torch.float64)
        diagnostics["w_alpha_ti"] = w_ti
        diagnostics["logP_alpha_ti"] = logP_ti
        diagnostics["logZ_alpha_ti"] = float(logZ_ti.detach().cpu())
    except Exception as e:
        diagnostics["alpha_weight_error_ti"] = f"{type(e).__name__}: {e}"

    if (w_bryan is not None) and (w_ti is not None):
        diagnostics["alpha_weight_l1_bryan_vs_ti"] = float(
            torch.sum(torch.abs(w_bryan - w_ti)).detach().cpu()
        )
        diagnostics["alpha_weight_argmax_alpha_bryan"] = float(
            alphas[torch.argmax(w_bryan)].detach().cpu()
        )
        diagnostics["alpha_weight_argmax_alpha_ti"] = float(
            alphas[torch.argmax(w_ti)].detach().cpu()
        )

    weights = None
    logP_alpha = None
    logZ_alpha = None
    method = requested_method
    if method in ("bryan", "bryan_laplace"):
        if w_bryan is not None:
            weights, logP_alpha = w_bryan, logP_bryan
            logZ_alpha = logZ_bryan
            diagnostics["alpha_weight_method"] = "bryan_laplace"
        elif w_ti is not None:
            weights, logP_alpha = w_ti, logP_ti
            logZ_alpha = logZ_ti
            diagnostics["alpha_weight_method"] = "ti"
            diagnostics["alpha_weight_selection_note"] = "fallback_from_bryan_request"
        else:
            diagnostics["alpha_weight_fallback_reason"] = "both_bryan_and_ti_failed"
    elif method == "ti":
        if w_ti is not None:
            weights, logP_alpha = w_ti, logP_ti
            logZ_alpha = logZ_ti
            diagnostics["alpha_weight_method"] = "ti"
        elif w_bryan is not None:
            weights, logP_alpha = w_bryan, logP_bryan
            logZ_alpha = logZ_bryan
            diagnostics["alpha_weight_method"] = "bryan_laplace"
            diagnostics["alpha_weight_selection_note"] = "fallback_from_ti_request"
        else:
            diagnostics["alpha_weight_fallback_reason"] = "both_ti_and_bryan_failed"
    else:
        if w_bryan is not None:
            weights, logP_alpha = w_bryan, logP_bryan
            logZ_alpha = logZ_bryan
            diagnostics["alpha_weight_method"] = "bryan_laplace"
            diagnostics["alpha_weight_selection_note"] = "auto_prefer_bryan"
        elif w_ti is not None:
            weights, logP_alpha = w_ti, logP_ti
            logZ_alpha = logZ_ti
            diagnostics["alpha_weight_method"] = "ti"
            diagnostics["alpha_weight_selection_note"] = "auto_fallback_ti"
        else:
            diagnostics["alpha_weight_fallback_reason"] = "both_bryan_and_ti_failed"

    if weights is None or logP_alpha is None:
        weights = torch.full(
            (alphas.numel(),),
            1.0 / float(alphas.numel()),
            device=device,
            dtype=torch.float64,
        )
        logP_alpha = torch.log(weights)
        diagnostics["alpha_weight_method"] = "uniform_fallback"
        diagnostics["logZ_alpha"] = float("nan")
        diagnostics["alpha_evidence_note"] = "unavailable_due_to_uniform_fallback"
    else:
        if logZ_alpha is not None:
            diagnostics["logZ_alpha"] = float(logZ_alpha.detach().cpu())
        else:
            diagnostics["logZ_alpha"] = float("nan")
        diagnostics["alpha_evidence_note"] = "relative_log_evidence_up_to_additive_constant"

    m_used = str(diagnostics.get("alpha_weight_method", "uniform_fallback"))
    if m_used.startswith("bryan"):
        diagnostics["alpha_weight_prior"] = prior_bryan
    elif m_used.startswith("ti"):
        diagnostics["alpha_weight_prior"] = prior_ti
    else:
        diagnostics["alpha_weight_prior"] = "none"

    return weights, logP_alpha, logZ_alpha
