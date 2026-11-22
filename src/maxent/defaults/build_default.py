from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

def _rel_l2_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-300) -> float:
    """Relative L2 distance ||x-y||/||y|| as python float."""
    num = torch.linalg.norm((x - y).to(torch.float64))
    den = torch.linalg.norm(y.to(torch.float64)).clamp_min(eps)
    return float(num / den)

def _gaussian_smooth_on_omega(
    y: torch.Tensor,
    omega: torch.Tensor,
    sigma: float,
    eps: float = 1e-300,
) -> torch.Tensor:
    """
    Gaussian smoothing on possibly non-uniform omega grid.
    Uses an NxN kernel; fine for current grid sizes.
    """
    sigma = float(sigma)
    if sigma <= 0.0:
        return y.clone()

    d = omega[:, None] - omega[None, :]
    w = torch.exp(-0.5 * (d / sigma) ** 2)
    w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)
    return w @ y

def _build_similar_default_from_dw(
    *,
    omega: torch.Tensor,
    D_w: torch.Tensor,
    sim_cfg: Dict[str, Any],
    floor: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Build a default model close to D_w but not identical.

    Strategy:
      1) Smooth D_w on omega.
      2) Blend D_w and the smoothed spectrum.
      3) Enforce relative-L2 distance to stay in [min_rel_l2, max_rel_l2].
    """
    sigma = float(sim_cfg.get("smooth_sigma", 0.35))
    mix_smoothed = float(sim_cfg.get("mix_smoothed", 0.30))
    min_rel_l2 = float(sim_cfg.get("min_rel_l2", 0.02))
    max_rel_l2 = float(sim_cfg.get("max_rel_l2", 0.15))

    if sigma <= 0.0:
        raise ValueError(f"default_model.similar_to_Dw.smooth_sigma must be > 0, got {sigma}")
    if not (0.0 <= mix_smoothed <= 1.0):
        raise ValueError(
            "default_model.similar_to_Dw.mix_smoothed must be in [0,1], "
            f"got {mix_smoothed}"
        )
    if min_rel_l2 < 0.0:
        raise ValueError(f"default_model.similar_to_Dw.min_rel_l2 must be >= 0, got {min_rel_l2}")
    if max_rel_l2 <= 0.0:
        raise ValueError(f"default_model.similar_to_Dw.max_rel_l2 must be > 0, got {max_rel_l2}")
    if min_rel_l2 > max_rel_l2:
        raise ValueError(
            "default_model.similar_to_Dw requires min_rel_l2 <= max_rel_l2, "
            f"got {min_rel_l2} > {max_rel_l2}"
        )

    smooth = _gaussian_smooth_on_omega(D_w, omega, sigma=sigma)
    candidate = (1.0 - mix_smoothed) * D_w + mix_smoothed * smooth
    candidate = torch.clamp(candidate, min=max(float(floor), 1e-300))

    diff_smooth = smooth - D_w
    rel_smooth = _rel_l2_distance(smooth, D_w)
    rel = _rel_l2_distance(candidate, D_w)

    # Too close: move further toward the smoothed shape.
    if rel < min_rel_l2:
        if rel_smooth > 1e-12:
            t = min(1.0, min_rel_l2 / rel_smooth)
            candidate = D_w + t * diff_smooth
        else:
            # Fallback for near-constant D_w: tiny deterministic tilt.
            span = (omega.max() - omega.min()).clamp_min(1e-12)
            x = (omega - omega.mean()) / span
            candidate = D_w * (1.0 + min_rel_l2 * x)
        candidate = torch.clamp(candidate, min=max(float(floor), 1e-300))
        rel = _rel_l2_distance(candidate, D_w)

    # Too far: scale residual back.
    if rel > max_rel_l2:
        scale = max_rel_l2 / max(rel, 1e-300)
        candidate = D_w + scale * (candidate - D_w)
        candidate = torch.clamp(candidate, min=max(float(floor), 1e-300))
        rel = _rel_l2_distance(candidate, D_w)

    info = {
        "enabled": True,
        "smooth_sigma": sigma,
        "mix_smoothed": mix_smoothed,
        "min_rel_l2": min_rel_l2,
        "max_rel_l2": max_rel_l2,
        "actual_rel_l2": rel,
        "smooth_rel_l2": rel_smooth,
    }
    return candidate, info

def add_gaussian_peaks_to_default(D_w: torch.Tensor,
                                w: torch.Tensor,
                                peaks,              # list of (mu, sigma, amp)
                                eps: float = 0.2,
                                renorm: bool = True,
                                floor: float = 1e-300):
        """
        D_w: (Nw,) default model (must be >=0 ideally)
        w  : (Nw,) frequency grid
        peaks: [(mu, sigma, amp), ...] all floats
        eps: mixing ratio
        renorm: match area of original D_w
        """
        D0 = torch.clamp(D_w, min=floor)

        p = torch.zeros_like(D0)
        for (mu, sigma, amp) in peaks:
            sigma = float(sigma)
            amp = float(amp)
            mu = float(mu)
            p = p + amp * torch.exp(-0.5 * ((w - mu) / sigma) ** 2)

        p = torch.clamp(p, min=floor)

        # add noise peak with controlled area

        # normalize p to unit area
        p_area = torch.trapz(p, w)
        p_unit = p / (p_area + 1e-300)   # ∫ p_unit dω = 1

        # eps = added spectral weight (area)
        eps = float(eps)  # ← 追加
        D_new = D0 + eps * p_unit
        D_new = torch.clamp(D_new, min=floor)

        print("delta_area =", torch.trapz(D_new - D0, w).item())
        
        if renorm:
            area0 = torch.trapz(D0, w)
            area1 = torch.trapz(D_new, w)
            D_new = D_new * (area0 / (area1 + 1e-300))
        return D_new
    
def _as_float_tuple(x: Any, *, name: str) -> Tuple[float, float]:
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        raise ValueError(f"{name} must be length-2 list/tuple, got {type(x)}: {x}")
    return float(x[0]), float(x[1])

def _gaussian_profile(
    *,
    omega: torch.Tensor,
    mu: float,
    sigma: float,
    floor: float,
) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError(f"Gaussian sigma must be > 0, got {sigma}")
    mu = float(mu)
    g = torch.exp(-0.5 * ((omega - mu) / sigma) ** 2)
    return torch.clamp(g, min=max(float(floor), 1e-300))

def _build_gaussian_default(
    *,
    omega: torch.Tensor,
    dcfg: Dict[str, Any],
    floor: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Build a single-Gaussian default model:
      m(omega) ~ exp(-(omega-mu)^2 / (2*sigma^2))
    """
    gcfg_raw = dcfg.get("gaussian", {})
    gcfg = gcfg_raw if isinstance(gcfg_raw, dict) else {}
    mu = float(gcfg.get("mu", gcfg.get("mean", 0.0)))
    sigma = float(gcfg.get("sigma", gcfg.get("std", 1.0)))
    amp = float(gcfg.get("amp", gcfg.get("weight", 1.0)))
    if amp <= 0.0:
        raise ValueError(f"default_model.gaussian.amp must be > 0, got {amp}")

    m = amp * _gaussian_profile(omega=omega, mu=mu, sigma=sigma, floor=floor)
    m = torch.clamp(m, min=max(float(floor), 1e-300))
    info = {"mu": mu, "sigma": sigma, "amp": amp}
    return m, info

def _build_gaussian_mixture_default(
    *,
    omega: torch.Tensor,
    dcfg: Dict[str, Any],
    floor: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Build Gaussian-mixture default model:
      m(omega) ~ sum_k w_k * N(omega; mu_k, sigma_k)
    """
    mcfg_raw = dcfg.get("gaussian_mixture", {})
    mcfg = mcfg_raw if isinstance(mcfg_raw, dict) else {}
    comps = mcfg.get("components", dcfg.get("components", None))
    if not isinstance(comps, list) or len(comps) == 0:
        raise ValueError(
            "default_model.gaussian_mixture.components must be a non-empty list of "
            "{mu, sigma, weight/amp} objects."
        )

    m = torch.zeros_like(omega, dtype=torch.float64)
    comp_info = []
    for i, c in enumerate(comps):
        if not isinstance(c, dict):
            raise ValueError(f"gaussian_mixture.components[{i}] must be a mapping.")
        mu = float(c.get("mu", c.get("mean", 0.0)))
        sigma = float(c.get("sigma", c.get("std", 1.0)))
        weight = float(c.get("weight", c.get("amp", 1.0)))
        if weight <= 0.0:
            raise ValueError(f"gaussian_mixture.components[{i}].weight must be > 0, got {weight}")
        g = _gaussian_profile(omega=omega, mu=mu, sigma=sigma, floor=floor)
        m = m + weight * g
        comp_info.append({"mu": mu, "sigma": sigma, "weight": weight})

    m = torch.clamp(m, min=max(float(floor), 1e-300))
    w_sum = float(sum(ci["weight"] for ci in comp_info))
    info = {
        "n_components": int(len(comp_info)),
        "components": comp_info,
        "weight_sum": w_sum,
    }
    return m, info

def build_default_model(
    *,
    omega: torch.Tensor,
    D_w: torch.Tensor,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Build MaxEnt default model m(ω) on the same ω-grid.

    Expected behavior:
      - base: m0(ω) = D_w (or something close; keep it simple and explicit)
      - optional: add Gaussian peaks (for toy/default tweaking)
      - optional: floor / renormalization

   Returns:
      m_w: (K,) float64 tensor on omega grid
      info: dict describing what was applied (for logging / artifacts)
    """
    if omega.ndim != 1 or D_w.ndim != 1 or omega.shape != D_w.shape:
        raise ValueError("omega and D_w must be 1D and same shape.")
    omega = omega.to(dtype=torch.float64)
    D_w = D_w.to(dtype=torch.float64)

    dcfg = cfg.get("default_model", {}) if isinstance(cfg, dict) else {}
    floor = float(dcfg.get("floor", 0.0))

    # ---- base default ----
    base = str(dcfg.get("base", "Dw")).lower()
    if base in ("dw", "d_w", "measure", "density"):
        m = D_w.clone()
    elif base in ("flat", "uniform"):
        # uniform over ω-grid (not over measure); keep as explicit baseline
        m = torch.ones_like(D_w)
    elif base in ("gaussian", "gauss", "normal"):
        m, g_info = _build_gaussian_default(omega=omega, dcfg=dcfg, floor=floor)
    elif base in ("gaussian_mixture", "gaussian-mixture", "gmm", "mixture"):
        m, gm_info = _build_gaussian_mixture_default(omega=omega, dcfg=dcfg, floor=floor)
    else:
       raise ValueError(
           f"Unknown default_model.base={base!r}. "
           "Use one of: 'Dw', 'flat', 'gaussian', 'gaussian_mixture'."
       )

    info: Dict[str, Any] = {"base": base}
    if base in ("gaussian", "gauss", "normal"):
        info["gaussian"] = g_info
    if base in ("gaussian_mixture", "gaussian-mixture", "gmm", "mixture"):
        info["gaussian_mixture"] = gm_info

    # ---- optional: build a "similar but not identical" model from D_w ----
    sim_cfg = dcfg.get("similar_to_Dw", dcfg.get("similar_to_dw", {}))
    if bool(sim_cfg.get("enabled", False)):
        if base not in ("dw", "d_w", "measure", "density"):
            raise ValueError("default_model.similar_to_Dw.enabled=true requires base='Dw'.")
        m, sim_info = _build_similar_default_from_dw(
            omega=omega,
            D_w=D_w,
            sim_cfg=sim_cfg,
            floor=floor,
        )
        info["similar_to_Dw"] = sim_info
    else:
        info["similar_to_Dw"] = {"enabled": False}

    # ---- optional: gaussian peaks ----
    # config example:
    # default_model:
    #   add_gaussian_peaks: true
    #   peaks:
    #     - {pos: -0.2, width: 0.15, amp: 0.5}
    #     - {pos:  0.8, width: 0.10, amp: 1.0}
    add_peaks = bool(dcfg.get("add_gaussian_peaks", False))
    peaks = dcfg.get("peaks", None)
    if add_peaks:
        if peaks is None:
            # backward-compatible keys (if you used them)
            # e.g. default_model.peak_pos, peak_width, peak_amp
            ppos = dcfg.get("peak_pos", None)
            pwid = dcfg.get("peak_width", None)
            pamp = dcfg.get("peak_amp", None)
            if ppos is not None and pwid is not None and pamp is not None:
                peaks = [{"pos": float(ppos), "width": float(pwid), "amp": float(pamp)}]
            else:
                peaks = []

        # Apply peaks one by one
        for p in peaks:
            pos = float(p["pos"])
            width = float(p.get("width", dcfg.get("peak_width", 0.1)))
            amp = float(p.get("amp", dcfg.get("peak_amp", 1.0)))
            m = add_gaussian_peaks_to_default(m, omega, [(pos, width, amp)])

        info["add_gaussian_peaks"] = True
        info["peaks"] = peaks
    else:
        info["add_gaussian_peaks"] = False

    # ---- floor to avoid zeros ----
    if floor > 0:
        m = torch.clamp(m, min=floor)
        info["floor"] = floor

    # ---- renormalize (optional) ----
    # normalize so that ∫ m(ω) dω equals ∫ D_w dω (or 1 if base=flat)
    renorm = bool(dcfg.get("renormalize", True))
    if renorm:
        # trapezoid ∫ f dω
        dω = omega[1:] - omega[:-1]
        def trapz(y: torch.Tensor) -> torch.Tensor:
            inc = 0.5 * (y[:-1] + y[1:]) * dω
            return inc.sum()

        target = trapz(D_w) if base in ("dw", "d_w", "measure", "density") else torch.tensor(1.0, dtype=torch.float64, device=omega.device)
        z = trapz(m)
        if not torch.isfinite(z) or float(z) <= 0.0:
            raise ValueError("Default model integral is non-positive; check configuration.")
        m = m * (target / z)
        info["renormalize"] = True
        info["integral_target"] = float(target)
        info["integral_before"] = float(z)
        info["integral_after"] = float(trapz(m))
    else:
        info["renormalize"] = False

    
    return m.to(torch.float64), info

def make_noisy_default(
    omega: torch.Tensor,
    default: torch.Tensor,
    *,
    variance: float,
    seed: int | None = None,
    clip_min: float = 1e-12,
) -> torch.Tensor:
    """
    Construct a default model m(ω)=1 + Gaussian noise with given variance.
    """
    device = omega.device
    dtype = omega.dtype

    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    else:
        g = None

    # # 無情報 default
    # m = torch.ones_like(omega, dtype=dtype, device=device)

    # ガウスノイズ（平均0、分散variance）
    sigma = variance ** 0.5
    noise = sigma * torch.max(default) * torch.randn(
        omega.shape,
        dtype=dtype,
        device=device,
        generator=g,
    )

    m_noisy = default + noise

    # MaxEnt 必須：非負制約
    m_noisy = torch.clamp(m_noisy, min=clip_min)

    return m_noisy
