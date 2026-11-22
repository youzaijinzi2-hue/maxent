from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import re
import numpy as np
import torch


def tensor_to_np(t: torch.Tensor, *, dtype: torch.dtype) -> np.ndarray:
    """
    Convert torch.Tensor -> numpy.ndarray with explicit dtype and CPU transfer.
    """
    return t.detach().to("cpu").to(dtype).numpy()

def _is_alpha_related_key(k: str) -> bool:
    """
    Heuristic: keys likely related to alpha posterior weights / evidence terms.
    Saved with 'diag_' prefix to avoid collisions.
    """
    kl = k.lower()
    # common names used across iterations
    if kl in {"w_alpha", "logp_alpha", "logz_alpha", "logz", "logp", "weights"}:
        return True
    # pattern-based catch-all (safe because we prefix in output)
    if "alpha" in kl and ("w" in kl or "weight" in kl or "logp" in kl or "logz" in kl or "post" in kl):
        return True
    return False

def save_inputs_npz(
    *,
    run_dir: Path,
    save_dtype: torch.dtype,
    wn: torch.Tensor,
    G_obs: torch.Tensor,
    omega: torch.Tensor,
    D_w: torch.Tensor,
    tau: torch.Tensor,
    sigma_tau: torch.Tensor,
    alphas: torch.Tensor,
) -> Path:
    """
    Save input artifacts to run_dir/artifacts/inputs.npz
    """
    art_dir = run_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    npz_path = art_dir / "inputs.npz"
    np.savez(
        npz_path,
        wn=tensor_to_np(wn, dtype=save_dtype),
        G_obs_real=tensor_to_np(torch.real(G_obs), dtype=save_dtype),
        G_obs_imag=tensor_to_np(torch.imag(G_obs), dtype=save_dtype),
        omega=tensor_to_np(omega, dtype=save_dtype),
        D_w=tensor_to_np(D_w, dtype=save_dtype),
        tau=tensor_to_np(tau, dtype=save_dtype),
        sigma_tau=tensor_to_np(sigma_tau, dtype=save_dtype),
        alphas=tensor_to_np(alphas, dtype=save_dtype),
    )
    return npz_path


def _as_np_any(x: Any, *, save_dtype: torch.dtype) -> Any:
    """
    Convert torch -> numpy; keep python/numpy as-is.
    """
    if isinstance(x, torch.Tensor):
        return tensor_to_np(x, dtype=save_dtype)
    return x


def build_result_dict(
    *,
    save_dtype: torch.dtype,
    omega: torch.Tensor,
    A_est: torch.Tensor,
    A_low: torch.Tensor,
    A_high: torch.Tensor,
    mcse: Any,
    diagnostics: Dict[str, Any],
    tau_metrics: Dict[str, Any],
    tau_metrics_raw: Optional[Dict[str, Any]] = None,
    G_tau_noisy: torch.Tensor,
    G_tau_rec: torch.Tensor,
    peak_pos_roi: Optional[torch.Tensor] = None,  # ★追加
    c_tau: float,
) -> Dict[str, Any]:
    """
    Create a flat dict for result.npz including diagnostics and key debug signals.
    """
    out: Dict[str, Any] = {
        "omega": tensor_to_np(omega, dtype=save_dtype),
        "A_est": tensor_to_np(A_est, dtype=save_dtype),
        "A_low": tensor_to_np(A_low, dtype=save_dtype),
        "A_high": tensor_to_np(A_high, dtype=save_dtype),
        "mcse": _as_np_any(mcse, save_dtype=save_dtype),
         # scalar metric: ensure it is saved deterministically as a numpy scalar
        "c_tau": np.asarray(float(c_tau), dtype=np.float64),
        # τ-space debug signals
        "G_tau_noisy_real": tensor_to_np(
            (G_tau_noisy.real if torch.is_complex(G_tau_noisy) else G_tau_noisy),
            dtype=save_dtype
        ),
        "G_tau_noisy_imag": tensor_to_np(
            (G_tau_noisy.imag if torch.is_complex(G_tau_noisy) else torch.zeros_like(G_tau_noisy)),
            dtype=save_dtype
        ),
        "G_tau_rec_real": tensor_to_np(
            (G_tau_rec.real if torch.is_complex(G_tau_rec) else G_tau_rec),
            dtype=save_dtype
        ),
        "G_tau_rec_imag": tensor_to_np(
            (G_tau_rec.imag if torch.is_complex(G_tau_rec) else torch.zeros_like(G_tau_rec)),
            dtype=save_dtype
        ),
    }
    # Optional: ROI peak position distribution (1D)
    if peak_pos_roi is not None:
        out["peak_pos_roi"] = tensor_to_np(peak_pos_roi, dtype=save_dtype)

    # tau metrics (scalars or arrays)
    for k, v in tau_metrics.items():
       out[k] = _as_np_any(v, save_dtype=save_dtype)

    # raw tau metrics (pre/post scalingなどの「生値」用). key衝突回避のため raw_ prefix
    if tau_metrics_raw is not None:
        for k, v in tau_metrics_raw.items():
            out[f"raw_{k}"] = _as_np_any(v, save_dtype=save_dtype)

    # diagnostics (selected arrays/scalars)
    for k in [
        "alphas", "H_means", "S_means", "w", "n_means",
        # alpha posterior artifacts (if present)
        "w_alpha", "logP_alpha", "logZ_alpha", "alpha_weight_method",
        "logZ_alpha_bryan", "logZ_alpha_ti", "alpha_evidence_note",
        "w_alpha_bryan", "logP_alpha_bryan", "w_alpha_ti", "logP_alpha_ti",
        "alpha_weight_method_requested", "alpha_weight_prior",
        "alpha_weight_prior_bryan", "alpha_weight_prior_ti",
        "alpha_weight_selection_note", "alpha_weight_l1_bryan_vs_ti",
        "alpha_weight_argmax_alpha_bryan", "alpha_weight_argmax_alpha_ti",
        "alpha_ref_mode", "alpha_ref_warning",
        "alpha_ref_map_idx", "alpha_ref_map_logpost", "alpha_ref_map_to_mean_rel_l2",
        "alpha_ref_map_local_samples", "alpha_ref_map_local_window",
        "alpha_ref_map_local_temperature", "alpha_ref_map_local_support_size",
        "alpha_ref_local_to_map_rel_l2",
        "tau_ints", "ess_per_chain", "total_ess", "ess_per_sec", "elapsed_sec",
        # MCMC phase metadata for convergence diagnostics.
        "n_steps", "n_step", "burnin", "hot_start", "adapt_end",
        "hot_start_ratio", "adapt_end_ratio",
        "rhat_max", "rhat_mean", "mcse_mean", "mcse_max",
        "rhat_method", "rhat_split_segments", "rhat_per_replica_max", "rhat_per_replica_mean",
        "pca_accept_rate", "ws_accept_rate", "cg_accept_rate",
        "swap_trial_count", "swap_accept_count", "swap_call_count", "swap_accept_rate",
        "target_accept_pca", "target_accept_ws", "target_accept_cg",
        "prob_adapt_metric",
        "move_prob_pca_final", "move_prob_ws_final", "move_prob_cg_final",
        "prob_history_steps",
        "move_prob_pca_history", "move_prob_ws_history", "move_prob_cg_history",
        "prob_utility_obs_pca_history", "prob_utility_obs_ws_history", "prob_utility_obs_cg_history",
        "prob_utility_est_pca_history", "prob_utility_est_ws_history", "prob_utility_est_cg_history",
        "prob_history_length",
        "mcse_threshold", "mcse_achieved_step", "mcse_achieved_time",
        # Optional thinned energy traces for convergence visualization.
        "H_samples_thin", "H_thin_stride", "H_thin_start", "H_thin_indices", "H_samples_thin_error",
    ]:
        if k in diagnostics:
            out[k] = _as_np_any(diagnostics[k], save_dtype=save_dtype)

    # Backward/forward compatibility:
    # - Keep existing 'w' if present (many scripts may rely on it)
    # - Also store an alias 'w_mix' so the meaning is explicit going forward.
    if "w" in diagnostics and "w_mix" not in out:
        out["w_mix"] = _as_np_any(diagnostics["w"], save_dtype=save_dtype)

    # If alpha posterior was stored as a nested dict, flatten it.
    # e.g., diagnostics["alpha_posterior"] = {"w_alpha":..., "logP_alpha":..., "logZ_alpha":...}
    if "alpha_posterior" in diagnostics and isinstance(diagnostics["alpha_posterior"], dict):
        ap = diagnostics["alpha_posterior"]
        for k2 in ["w_alpha", "logP_alpha", "logZ_alpha", "logZ", "logP"]:
            if k2 in ap and k2 not in out:
                out[k2] = _as_np_any(ap[k2], save_dtype=save_dtype)

    # Safety net:
    # Save any alpha-related diagnostic keys with a prefix to avoid name collisions.
    # This prevents "I swear it's in diagnostics" issues from reappearing.
    for k, v in diagnostics.items():
        if _is_alpha_related_key(k):
            pref = f"diag_{k}"
            # don't overwrite explicit saved keys
            if pref not in out and k not in out:
                out[pref] = _as_np_any(v, save_dtype=save_dtype)


    # expand move_diag if present
    if "move_diag" in diagnostics:
        for move, md in diagnostics["move_diag"].items():
            for k, v in md.items():
                out[f"move_diag_{move}_{k}"] = _as_np_any(v, save_dtype=save_dtype)

    return out


def save_result_npz(
    *,
    run_dir: Path,
    save_dtype: torch.dtype,
    result_dict: Dict[str, Any],
) -> Path:
    """
    Save result artifacts to run_dir/artifacts/result.npz
    """
    art_dir = run_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    npz_path = art_dir / "result.npz"
    np.savez(npz_path, **result_dict)
    return npz_path
