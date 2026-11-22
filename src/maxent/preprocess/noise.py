from __future__ import annotations

from typing import Any, Dict

import torch

from maxent.operators.transform import sigma_tau_from_sigma_iw, kernel_tau_iw


def _sigma_iw_guard_options(prep_cfg: Dict[str, Any]) -> Dict[str, float | bool]:
    """
    Read sigma_iw guard options with backward-compatible fallbacks.
    """
    guard_cfg = prep_cfg.get("sigma_iw_guard", {})
    if not isinstance(guard_cfg, dict):
        guard_cfg = {}

    def _opt(name: str, default: float | bool):
        if name in guard_cfg:
            return guard_cfg[name]
        return prep_cfg.get(f"sigma_iw_guard_{name}", default)

    return {
        "enabled": bool(_opt("enabled", True)),
        "max_rel": float(_opt("max_rel", 3.0e-3)),
        "cap_quantile": float(_opt("cap_quantile", 0.90)),
        "amp_floor_rel": float(_opt("amp_floor_rel", 1.0e-3)),
        "sigma_min": float(_opt("sigma_min", 1.0e-12)),
    }


def _apply_sigma_iw_guard(
    *,
    sigma_iw: torch.Tensor,
    G_thm: torch.Tensor,
    wn: torch.Tensor,
    prep_cfg: Dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """
    Prevent structurally over-large sigma_iw by clipping against a relative cap.

    cap_pos(n) = max_rel * max(|G_thm(iw_n)|, amp_floor), for wn>0.
    """
    opts = _sigma_iw_guard_options(prep_cfg)
    sigma = torch.as_tensor(sigma_iw, device=device, dtype=torch.float64)
    if sigma.ndim > 1:
        raise ValueError("prepare.sigma_iw must be scalar or 1D array.")

    sigma = torch.clamp(sigma, min=max(opts["sigma_min"], 0.0))
    info: Dict[str, Any] = {
        "enabled": bool(opts["enabled"]),
        "clipped": False,
        "max_rel": float(opts["max_rel"]),
        "cap_quantile": float(opts["cap_quantile"]),
        "amp_floor_rel": float(opts["amp_floor_rel"]),
        "sigma_min": float(opts["sigma_min"]),
    }
    if not opts["enabled"] or opts["max_rel"] <= 0.0:
        return sigma, info

    pos_mask = wn.to(torch.float64) > 0
    n_pos = int(pos_mask.sum().item())
    if n_pos == 0:
        return sigma, info

    abs_g_pos = torch.abs(G_thm[pos_mask].to(torch.complex128)).to(torch.float64)
    amp_floor = float(torch.median(abs_g_pos).item()) * max(opts["amp_floor_rel"], 0.0)
    amp_floor = max(amp_floor, 1e-30)
    cap_pos = float(opts["max_rel"]) * torch.clamp(abs_g_pos, min=amp_floor)

    if sigma.ndim == 0:
        q = min(max(float(opts["cap_quantile"]), 0.0), 1.0)
        cap_scalar = torch.quantile(cap_pos, q) if cap_pos.numel() > 1 else cap_pos.reshape(())
        before = float(sigma.item())
        after = float(torch.clamp(sigma, max=cap_scalar).item())
        sigma = torch.as_tensor(after, device=device, dtype=torch.float64)
        info["cap_scalar"] = float(cap_scalar.item())
        info["clipped"] = after < before - 1e-18
        info["before_scalar"] = before
        info["after_scalar"] = after
        return sigma, info

    sig_pos = _as_sigma_pos(sigma_iw=sigma, n_pos=n_pos, device=device)
    before = sig_pos.clone()
    sig_pos = torch.minimum(sig_pos, cap_pos)
    info["clipped"] = bool(torch.any(sig_pos < before))
    info["cap_pos_min"] = float(cap_pos.min().item())
    info["cap_pos_med"] = float(cap_pos.median().item())
    info["cap_pos_max"] = float(cap_pos.max().item())
    if info["clipped"]:
        info["before_pos_min"] = float(before.min().item())
        info["before_pos_med"] = float(before.median().item())
        info["before_pos_max"] = float(before.max().item())
        info["after_pos_min"] = float(sig_pos.min().item())
        info["after_pos_med"] = float(sig_pos.median().item())
        info["after_pos_max"] = float(sig_pos.max().item())
    return sig_pos, info


def build_sigma_iw(*, G_thm: torch.Tensor, wn: torch.Tensor, prep_cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """
    Priority:
      1) external sigma_iw given in config (scalar or vector)
      2) fallback: sigma_iw = amp(G_thm)/snr (legacy behavior)
    Returns: tensor (float64), scalar or 1D
    """
    sigma_iw_external = prep_cfg.get("sigma_iw", None)
    sigma_source = "snr"
    if sigma_iw_external is not None:
        sigma_iw = torch.as_tensor(sigma_iw_external, device=device, dtype=torch.float64)
        sigma_source = "external"
    else:
        snr = float(prep_cfg["snr"])
        sigma_iw_mode = str(prep_cfg.get("sigma_iw_mode", "rms")).lower()

        absG = torch.abs(G_thm)
        amp_rms = torch.sqrt(torch.mean(absG ** 2))
        amp_max = torch.max(absG)

        if sigma_iw_mode in ("rms", "amp_rms", "rms_amp"):
            amp_used = amp_rms
        elif sigma_iw_mode in ("max", "amp_max", "peak"):
            amp_used = amp_max
        else:
            raise ValueError(
                f"Unknown prepare.sigma_iw_mode={sigma_iw_mode!r}. "
                "Use one of: 'rms', 'max'."
            )
        sigma_iw = (amp_used / (snr + 1e-300)).to(torch.float64)
        print(
            "[iw-noise] "
            f"snr={snr:.6g} mode={sigma_iw_mode} "
            f"amp_rms={float(amp_rms):.6g} amp_max={float(amp_max):.6g} "
            f"sigma_iw={float(sigma_iw):.6g}"
        )

    sigma_iw, guard = _apply_sigma_iw_guard(
        sigma_iw=sigma_iw,
        G_thm=G_thm,
        wn=wn,
        prep_cfg=prep_cfg,
        device=device,
    )
    if sigma_iw.ndim == 1:
        print(
            f"[iw-noise] sigma_iw: {sigma_source} (shape={tuple(sigma_iw.shape)}) "
            f"min/med/max={float(sigma_iw.min()):.6g}/{float(sigma_iw.median()):.6g}/{float(sigma_iw.max()):.6g}"
        )
    else:
        print(f"[iw-noise] sigma_iw: {sigma_source} (scalar={float(sigma_iw):.6g})")

    if bool(guard.get("enabled", False)) and bool(guard.get("clipped", False)):
        if sigma_iw.ndim == 0:
            print(
                "[iw-noise] sigma_iw guard clipped scalar: "
                f"{guard.get('before_scalar', float('nan')):.6g} -> "
                f"{guard.get('after_scalar', float('nan')):.6g} "
                f"(cap={guard.get('cap_scalar', float('nan')):.6g}, "
                f"max_rel={guard.get('max_rel', float('nan')):.6g}, "
                f"q={guard.get('cap_quantile', float('nan')):.3f})"
            )
        else:
            print(
                "[iw-noise] sigma_iw guard clipped vector: "
                f"before max={guard.get('before_pos_max', float('nan')):.6g} -> "
                f"after max={guard.get('after_pos_max', float('nan')):.6g} "
                f"(cap max={guard.get('cap_pos_max', float('nan')):.6g}, "
                f"max_rel={guard.get('max_rel', float('nan')):.6g})"
            )
    return sigma_iw


def build_sigma_tau(
   *,
    wn_pos: torch.Tensor,
    tau: torch.Tensor,
    beta: float,
    sigma_iw: torch.Tensor,
    prep_cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    If prepare.calc_sigma_tau:
      sigma_tau = sigma_tau_from_sigma_iw(...)
    else:
      sigma_tau = prepare.sigma_tau * ones_like(tau)
    """
    calc_sigma_tau = bool(prep_cfg.get("calc_sigma_tau", False))
    if calc_sigma_tau:
        return sigma_tau_from_sigma_iw(wn_pos, tau, beta, sigma_iw)
    s = float(prep_cfg.get("sigma_tau", 0.0))
    return s * torch.ones_like(tau, dtype=torch.float64, device=tau.device)


def build_sigma_tau_cov(
    *,
    wn_pos: torch.Tensor,
    tau: torch.Tensor,
    beta: float,
    sigma_iw: torch.Tensor,
    prep_cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Build full tau-space covariance induced from independent iω noise.
    """
    calc_sigma_tau = bool(prep_cfg.get("calc_sigma_tau", False))
    jitter = float(prep_cfg.get("tau_cov_jitter", 1e-12))

    if not calc_sigma_tau:
        s = float(prep_cfg.get("sigma_tau", 0.0))
        T = int(tau.numel())
        I = torch.eye(T, device=tau.device, dtype=torch.float64)
        return (s * s) * I + jitter * I

    K_cos, K_sin = kernel_tau_iw(wn_pos=wn_pos, tau=tau, beta=beta, dtype=torch.float64)
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

    sig2_half = 0.5 * (sigma_vec ** 2)  # Var(Re)=Var(Im)=sigma^2/2
    C_cos = (K_cos * sig2_half[None, :]) @ K_cos.T
    C_sin = (K_sin * sig2_half[None, :]) @ K_sin.T
    Sigma_tau = C_cos + C_sin
    Sigma_tau = 0.5 * (Sigma_tau + Sigma_tau.T)
    T = Sigma_tau.shape[0]
    I = torch.eye(T, device=Sigma_tau.device, dtype=Sigma_tau.dtype)
    return Sigma_tau + jitter * I


def _as_sigma_pos(*, sigma_iw: torch.Tensor, n_pos: int, device: torch.device) -> torch.Tensor:
    """
    Normalize sigma_iw to a positive-frequency std vector (N_pos,).
    """
    sig = torch.as_tensor(sigma_iw, device=device, dtype=torch.float64)
    if sig.ndim == 0:
        return torch.full((n_pos,), float(sig.item()), device=device, dtype=torch.float64)
    if sig.ndim != 1:
        raise ValueError(f"sigma_iw must be scalar or 1D, got ndim={sig.ndim}")
    if sig.numel() != n_pos:
        raise ValueError(f"sigma_iw length mismatch: got {sig.numel()} vs N_pos={n_pos}")
    return sig


def sigma_iw_full_from_pos(
    *,
    wn: torch.Tensor,
    sigma_iw: torch.Tensor,
) -> torch.Tensor:
    """
    Expand scalar/positive-frequency sigma_iw to full wn ordering.
    """
    wn = wn.to(torch.float64)
    device = wn.device
    pos_mask = wn > 0
    neg_mask = wn < 0
    pos_idx = torch.where(pos_mask)[0]
    neg_idx = torch.where(neg_mask)[0]
    n_pos = int(pos_idx.numel())
    n_neg = int(neg_idx.numel())
    if n_pos != n_neg:
        raise ValueError(f"Expected equal #positive/#negative wn, got {n_pos} and {n_neg}")
    if n_pos == 0:
        raise ValueError("No positive Matsubara frequencies found.")
    if not torch.allclose(
        torch.abs(wn[pos_idx]),
        torch.abs(torch.flip(wn[neg_idx], dims=[0])),
        rtol=1e-8,
        atol=1e-12,
    ):
        raise ValueError("wn positive/negative pairing mismatch; expected symmetric ordering.")

    sig_pos = _as_sigma_pos(sigma_iw=sigma_iw, n_pos=n_pos, device=device)
    sig_full = torch.empty_like(wn, dtype=torch.float64)
    sig_full[pos_idx] = sig_pos
    sig_full[neg_idx] = torch.flip(sig_pos, dims=[0])
    return sig_full


def build_noisy_giw_observation(
    *,
    G_thm: torch.Tensor,
    wn: torch.Tensor,
    sigma_iw: torch.Tensor,
    prep_cfg: Dict[str, Any],
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """
    Build iω observation by injecting conjugate-symmetric complex Gaussian noise.
    """
    G_thm = G_thm.to(torch.complex128)
    wn = wn.to(torch.float64)
    device = G_thm.device

    add_iw_noise = bool(prep_cfg.get("add_iw_noise", True))
    iw_noise_scale = float(prep_cfg.get("iw_noise_scale", 1.0))
    if not add_iw_noise or iw_noise_scale == 0.0:
        return G_thm, {
            "enabled": False,
            "iw_noise_scale": iw_noise_scale,
            "note": "iω noise disabled",
        }

    pos_mask = wn > 0
    neg_mask = wn < 0
    pos_idx = torch.where(pos_mask)[0]
    neg_idx = torch.where(neg_mask)[0]
    n_pos = int(pos_idx.numel())
    n_neg = int(neg_idx.numel())
    if n_pos != n_neg:
        raise ValueError(f"Expected equal #positive/#negative wn, got {n_pos} and {n_neg}")
    if n_pos == 0:
        raise ValueError("No positive Matsubara frequencies found.")
    if not torch.allclose(
        torch.abs(wn[pos_idx]),
        torch.abs(torch.flip(wn[neg_idx], dims=[0])),
        rtol=1e-8,
        atol=1e-12,
    ):
        raise ValueError("wn positive/negative pairing mismatch; expected symmetric ordering.")

    sig_pos = _as_sigma_pos(sigma_iw=sigma_iw, n_pos=n_pos, device=device) * iw_noise_scale

    # Preserve G(-iwn)=conj(G(iwn)) by mirrored noise construction.
    eps_half = sig_pos / torch.sqrt(torch.tensor(2.0, device=device, dtype=torch.float64))
    noise_re = torch.randn((n_pos,), device=device, dtype=torch.float64) * eps_half
    noise_im = torch.randn((n_pos,), device=device, dtype=torch.float64) * eps_half
    noise_pos = noise_re.to(torch.complex128) + 1j * noise_im.to(torch.complex128)

    noise_full = torch.zeros_like(G_thm, dtype=torch.complex128)
    noise_full[pos_idx] = noise_pos
    noise_full[neg_idx] = torch.conj(torch.flip(noise_pos, dims=[0]))

    G_obs = G_thm + noise_full
    info = {
        "enabled": True,
        "iw_noise_scale": iw_noise_scale,
        "sigma_iw_pos_min": float(sig_pos.min().item()),
        "sigma_iw_pos_med": float(sig_pos.median().item()),
        "sigma_iw_pos_max": float(sig_pos.max().item()),
    }
    return G_obs, info
