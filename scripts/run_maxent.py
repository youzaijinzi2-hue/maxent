from __future__ import annotations

import argparse
import math
import datetime
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import logging

import numpy as np
import torch
import yaml
import sys

from maxent.preprocess.xas_preprocess import Prepare
from maxent.mcmc.sampler import MaxEnt, MaxEntConfig
from typing import Any

from maxent.utils.provenance import (
    sha256_file,
    get_git_info,
    get_env_info,
    make_run_id,
)
from maxent.operators.transform import (
    giw_from_A,
)
from maxent.diagnostics.forward_checks import forward_self_consistency_test
from maxent.metric import white_iw, estimate_c_scalar
from maxent.defaults.build_default import build_default_model, make_noisy_default
from maxent.tau_metrics import compute_tau_metrics
from maxent.runner.artifacts import save_run_artifacts
from maxent.operators.mapping import build_phi_phi_inv_jacobian_torch
from maxent.diagnostics.peak_position import (
    _compute_peak_pos_roi_from_sampler,
    _compute_peak_pos_thin_from_sampler
)
from maxent.diagnostics.tau_diagnostics import (
    fit_c_tau,
    compute_tau_metrics_pair,
    print_tau_diagnostics_summary,
)
from maxent.runner.setup import (
    save_reproducibility_files,
    prepare_run_dir,
    write_resolved_config,
)
from maxent.io.manifest import write_input_manifest_and_meta
from maxent.preprocess.noise import (
    build_sigma_iw,
    build_sigma_tau,
    build_sigma_tau_cov,
    build_noisy_giw_observation,
    sigma_iw_full_from_pos,
)
from maxent.postprocess.run_postprocess import (
    run_postprocess,
)
from maxent.runner.logging import (
    setup_logger,
)
from maxent.diagnostics.iw_diagnostics import (
    run_iw_diagnostics,
)
from maxent.diagnostics.mcmc_convergence import save_mcmc_convergence_figure

def torch_dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported save_float: {name}")


def giw_to_gtau_full(*, G_iw: torch.Tensor, wn: torch.Tensor, tau: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Full Matsubara sum (same convention as sampler._compute_G_tau).
    """
    G_iw = G_iw.to(torch.complex128)
    wn = wn.to(torch.float64)
    tau = tau.to(torch.float64)
    exp_term = torch.exp((-1j) * wn[:, None].to(torch.complex128) * tau[None, :].to(torch.complex128))
    G_tau = (1.0 / float(beta)) * torch.sum(exp_term * G_iw[:, None], dim=0)
    return G_tau.real.to(torch.float64)

def main():
    # ---- diagnostics flags ----
    # Policy:
    #   - τ-space diagnostics are the primary acceptance criteria in this runner.
    #   - iω-space diagnostics are optional (debug/reference) and can be disabled.
    # Usage in YAML:
    # diagnostics:
    #   do_iw: false
    #   do_forward_self_test: true
    #
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    diag_cfg: dict[str, Any] = cfg.get("diagnostics", {}) or {}
    do_iw_diag = bool(diag_cfg.get("do_iw", False))
    do_forward_self_test = bool(diag_cfg.get("do_forward_self_test", True))
    mcmc_plot_cfg = diag_cfg.get("mcmc_plot", {}) or {}
    if not isinstance(mcmc_plot_cfg, dict):
        raise ValueError("diagnostics.mcmc_plot must be a mapping.")
    do_mcmc_plot = bool(mcmc_plot_cfg.get("enabled", True))

    # ---- seeds ----
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- run dir ----
    out_root = Path(cfg["output"]["root"])
    run_id = make_run_id(cfg["output"].get("tag", "run"))
    run_dir = prepare_run_dir(out_root=out_root, run_id=run_id)

    # ---- logging ----
    log_cfg = cfg.get("logging", {}) or {}
    log_level = str(log_cfg.get("level", "INFO"))
    log = setup_logger(run_dir=run_dir, level=log_level)

    # ---- reproducibility files (git sha, pip freeze, python version) ----
    repro = save_reproducibility_files(run_dir)
    log.info(f"[meta] git sha saved: {repro.get('git_sha','unknown')}")

    # ---- save resolved config ----
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_resolved_config(run_dir=run_dir, cfg=cfg)
    
    # ---- input manifest ----
    raw_path = cfg["dataset"]["file_name"]
    write_input_manifest_and_meta(
        run_dir=run_dir,
        run_id=run_id,
        cfg=cfg,
        cfg_path=cfg_path,
        raw_path=raw_path,
        repro=repro,
    )
    
    # ---- device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # デバイス設定
    
    # ---- prepare ----
    beta = float(cfg["prepare"]["beta"])
    Nn = int(cfg["prepare"]["Nn"])

    prepare = Prepare()
    G_obs, G_thm, wn, omega_sub_t, D_norm_sub_t = prepare.prepare(
        file_name=str(raw_path),
        cfg=cfg,
        beta=beta,
        Nn=Nn,
        device=device,
    )
    device = G_obs.device

    # ---- grids ----
    N_tau = int(cfg["grid"]["N_tau"])
    # tau = torch.linspace(0.0, beta, N_tau, device=device, dtype=torch.float64)
    # endpoints excluded: tau in (0, beta)
    tau = (torch.arange(N_tau, device=device, dtype=torch.float64) + 1.0) * (beta / (N_tau + 1.0))
    
    # ---- noise scale in iω ----
    prep_cfg = cfg["prepare"]
    sigma_iw = build_sigma_iw(G_thm=G_thm, wn=wn, prep_cfg=prep_cfg, device=device)
    wn_pos = wn[wn.to(torch.float64) > 0]
    
    # after wn_pos computed and sigma_iw constructed
    if torch.is_tensor(sigma_iw) and sigma_iw.ndim == 1:
        if sigma_iw.numel() != wn_pos.numel():
            raise ValueError(
                f"prepare.sigma_iw length must match wn_pos: "
                f"sigma_iw={sigma_iw.numel()} vs wn_pos={wn_pos.numel()}"
            )
    sigma_tau = build_sigma_tau(wn_pos=wn_pos, tau=tau, beta=beta, sigma_iw=sigma_iw, prep_cfg=prep_cfg)
    use_tau_covariance = bool(prep_cfg.get("use_tau_covariance", False))
    sigma_model = sigma_tau
    if use_tau_covariance:
        sigma_model = build_sigma_tau_cov(
            wn_pos=wn_pos,
            tau=tau,
            beta=beta,
            sigma_iw=sigma_iw,
            prep_cfg=prep_cfg,
        )
        log.info("[tau-noise] using full tau covariance for likelihood: shape=%s", tuple(sigma_model.shape))
    else:
        log.info("[tau-noise] using diagonal tau std for likelihood: shape=%s", tuple(sigma_model.shape))
        
    # ---- iω-consistent observation path ----
    G_obs, iw_noise_info = build_noisy_giw_observation(
        G_thm=G_thm,
        wn=wn,
        sigma_iw=sigma_iw,
        prep_cfg=prep_cfg,
    )
    sigma_iw_full = sigma_iw_full_from_pos(wn=wn, sigma_iw=sigma_iw)
    log.info("[iw-noise] observation path: iω-consistent, info=%s", json.dumps(iw_noise_info, ensure_ascii=False))

    # τ diagnostics are evaluated on Gτ derived from the same iω data used in inference.
    G_tau_true = giw_to_gtau_full(G_iw=G_thm, wn=wn, tau=tau, beta=beta)
    G_tau_noisy = giw_to_gtau_full(G_iw=G_obs, wn=wn, tau=tau, beta=beta)
    
    # phi / phi_inv defined by min/max of omega_sub_t (same as your main)
    w_min, w_max = omega_sub_t.min(), omega_sub_t.max()
    
    # ---- alphas ----
    a_cfg = cfg["alphas"]
    if a_cfg["kind"] != "logspace":
        raise ValueError("Only alphas.kind=logspace is supported in this runner for now.")

    alphas = torch.logspace(
        float(a_cfg["start_exp"]),
        float(a_cfg["end_exp"]),
        int(a_cfg["num"]),
        device=device,
        dtype=torch.float64,
    )
    
    # ---- default ----
    # m = torch.ones_like(omega_sub_t)
    m_default, default_info = build_default_model(omega=omega_sub_t, D_w=D_norm_sub_t, cfg=cfg)
    log.info("[default] model config: %s", json.dumps(default_info, ensure_ascii=False))
    # m_default = make_noisy_default(omega=omega_sub_t,
    #                                default=m_default, 
    #                                variance=sigma_iw, 
    #                                seed=cfg.get("seed"),
    #                                 )
    phi, phi_inv, jac, x_grid, Z = build_phi_phi_inv_jacobian_torch(omega_sub_t, m_default)
    # ---- config & run ----
    alpha_post_cfg = cfg.get("alpha_posterior", {}) or {}
    sampler_mode = str(cfg["sampler"].get("mode", "mixed"))
    sampler_mode_lc = sampler_mode.lower()
    config = MaxEntConfig(
        wn=wn,
        beta=beta,
        tau=tau,
        sigma_tau=sigma_model,
        alphas=alphas,
        sampler_mode=sampler_mode,
        n_step=int(cfg["sampler"]["n_step"]),
        burnin_ratio=float(cfg["sampler"]["burnin_ratio"]),
        hot_start_ratio=float(cfg["sampler"].get("hot_start_ratio", 0.50)),
        adapt_end_ratio=float(cfg["sampler"].get("adapt_end_ratio", 0.80)),
        adapt_hot_interval=int(cfg["sampler"].get("adapt_hot_interval", 1)),
        adapt_late_interval=int(cfg["sampler"].get("adapt_late_interval", 100)),
        step_size_mh=float(cfg["sampler"].get("step_size_mh", cfg["sampler"].get("step_size_pca", 0.05))),
        step_size_pca=float(cfg["sampler"].get("step_size_pca", 0.05)),
        step_size_ws=float(cfg["sampler"].get("step_size_ws", 0.05)),
        step_size_cg=float(cfg["sampler"].get("step_size_cg", 0.05)),
        scale_adapt_t0=float(cfg["sampler"].get("scale_adapt_t0", 10.0)),
        scale_adapt_gamma=float(cfg["sampler"].get("scale_adapt_gamma", 0.05)),
        scale_adapt_kappa=float(cfg["sampler"].get("scale_adapt_kappa", 0.75)),
        scale_min=float(cfg["sampler"].get("scale_min", 1e-3)),
        scale_max=float(cfg["sampler"].get("scale_max", 1e1)),
        accept_ema_kappa=float(cfg["sampler"].get("accept_ema_kappa", 0.80)),
        prob_adapt_kappa=float(cfg["sampler"].get("prob_adapt_kappa", 0.90)),
        prob_adapt_metric=str(cfg["sampler"].get("prob_adapt_metric", "esjd_per_sec")),
        prob_min=float(cfg["sampler"].get("prob_min", 0.10)),
        prob_utility_floor=float(cfg["sampler"].get("prob_utility_floor", 1e-12)),
        prob_min_trials_per_move=int(cfg["sampler"].get("prob_min_trials_per_move", 100)),
        prob_esjd_whitened=bool(cfg["sampler"].get("prob_esjd_whitened", True)),
        prob_whiten_jitter=float(cfg["sampler"].get("prob_whiten_jitter", 1e-6)),
        prob_whiten_shrinkage=float(cfg["sampler"].get("prob_whiten_shrinkage", 0.05)),
        prob_whiten_weight_mode=str(cfg["sampler"].get("prob_whiten_weight_mode", "count")),
        prob_swap_reward_weight=float(cfg["sampler"].get("prob_swap_reward_weight", 0.15)),
        prob_swap_warmup_updates=int(cfg["sampler"].get("prob_swap_warmup_updates", 10)),
        cg_cov_blend=float(cfg["sampler"].get("cg_cov_blend", 0.20)),
        cg_cov_jitter=float(cfg["sampler"].get("cg_cov_jitter", 1e-6)),
        swap_freq=int(cfg["sampler"]["swap_freq"]),
        use_replica_exchange=bool(
            cfg["sampler"].get("use_replica_exchange", sampler_mode_lc != "mh")
        ),
        target_accept_pca=float(cfg["sampler"].get("target_accept_pca", 0.50)),
        target_accept_ws=float(cfg["sampler"].get("target_accept_ws", 0.50)),
        target_accept_cg=float(cfg["sampler"].get("target_accept_cg", 0.50)),
        mcse_threshold=float(cfg["sampler"].get("mcse_threshold", 0.005)),
        mcse_check_interval=int(cfg["sampler"].get("mcse_check_interval", 100)),
        mcse_iat_window=int(cfg["sampler"].get("mcse_iat_window", 2000)),
        iat_final_window=int(cfg["sampler"].get("iat_final_window", 0)),
        iat_max_fft_bytes=int(cfg["sampler"].get("iat_max_fft_bytes", 134217728)),
        alpha_weight_method=str(alpha_post_cfg.get("method", "auto")),
        alpha_prior_bryan=str(alpha_post_cfg.get("prior_bryan", "flat")),
        alpha_prior_ti=str(alpha_post_cfg.get("prior_ti", "flat")),
        alpha_bryan_main_term=str(alpha_post_cfg.get("bryan_main_term", "map")),
        alpha_reference_spectrum=str(alpha_post_cfg.get("reference_spectrum", "map")),
        alpha_map_local_samples=int(alpha_post_cfg.get("map_local_samples", 64)),
        alpha_map_local_logp_window=float(alpha_post_cfg.get("map_local_logp_window", 5.0)),
        alpha_map_local_temperature=float(alpha_post_cfg.get("map_local_temperature", 1.0)),
        alpha_map_local_seed=int(alpha_post_cfg.get("map_local_seed", seed)),
        rhat_split_segments=int(diag_cfg.get("rhat_split_segments", 4)),
        rhat_exclude_edge_bins=int(diag_cfg.get("rhat_exclude_edge_bins", 1)),
    )

    sampler = MaxEnt(
        G_iw=G_obs,
        phi=phi,
        phi_inv=phi_inv,
        jacobian=jac,
        D_w=m_default,
        omega=omega_sub_t,
        config=config,
    )

    # Optional acceleration (runner responsibility)
    s_cfg = cfg.get("sampler", {})
    if bool(s_cfg.get("compile", False)):
        backend = str(s_cfg.get("compile_backend", "eager"))
        if hasattr(torch, "compile"):
            sampler.run_replica_exchange = torch.compile(
                sampler.run_replica_exchange, backend=backend
            )
            log.info(f"[info] torch.compile enabled for run_replica_exchange (backend={backend})")
        else:
            log.warning("[warn] torch.compile is not available in this PyTorch build; skipping compile")
            
    A_est, A_low, A_high, mcse, diagnostics = sampler.run_max_ent()
    if "logZ_alpha" in diagnostics:
        log.info(
            "[default-evidence] logZ_alpha(rel)=%s method=%s",
            diagnostics.get("logZ_alpha"),
            diagnostics.get("alpha_weight_method", "unknown"),
        )
    if "logZ_alpha_bryan" in diagnostics:
        log.info("[default-evidence] logZ_alpha_bryan(rel)=%s", diagnostics.get("logZ_alpha_bryan"))
    if "logZ_alpha_ti" in diagnostics:
        log.info("[default-evidence] logZ_alpha_ti(rel)=%s", diagnostics.get("logZ_alpha_ti"))

    # Compute reconstructed G_tau from recovered spectrum A_est(ω)
    G_iw_rec = giw_from_A(A_est, omega_sub_t, wn)
    G_tau_rec = giw_to_gtau_full(G_iw=G_iw_rec, wn=wn, tau=tau, beta=beta)
    
    # ---- ROI peak position distribution (for histogram) ----
    # Default ROI is aligned with plotting() call at the end.
    plot_cfg = cfg.get("plotting", {}) or {}
    roi = tuple(plot_cfg.get("roi", (-5, 5)))
    peak_max_samples = int(plot_cfg.get("peak_pos_samples", 5000))
    peak_seed = int(plot_cfg.get("peak_pos_seed", 0))
    peak_keep_per_replica = int(plot_cfg.get("peak_pos_keep_per_replica", 2000))
    try:
        peak_pos_roi = _compute_peak_pos_roi_from_sampler(
            sampler=sampler,
            diagnostics=diagnostics,
            roi=roi,
            max_samples=peak_max_samples,
            seed=peak_seed,
        )
        log.info(f"[peak] peak_pos_roi computed: M={peak_pos_roi.numel()} ROI={roi}")
        # quick stats
        q = torch.tensor([0.16, 0.5, 0.84], device=peak_pos_roi.device, dtype=torch.float64)
        qq = torch.quantile(peak_pos_roi.to(torch.float64), q)
        log.info(f"[peak] ω_peak q16/q50/q84 = {float(qq[0]):.6g} / {float(qq[1]):.6g} / {float(qq[2]):.6g}")
    except Exception as e:
        peak_pos_roi = None
        log.warning(f"[warn] peak_pos_roi skipped: {e}")

    # ---- α-conditioned peak position samples (for alpha-landscape) ----
    try:
        peak_pos_thin = _compute_peak_pos_thin_from_sampler(
            sampler=sampler,
            diagnostics=diagnostics,
            roi=roi,
            max_keep_per_replica=peak_keep_per_replica,
        )
        # Save into diagnostics so it ends up in result.npz automatically
        diagnostics["peak_pos_thin"] = peak_pos_thin.detach().cpu().numpy()
        log.info(f"[peak] peak_pos_thin computed: shape={tuple(peak_pos_thin.shape)} ROI={roi}")
    except Exception as e:
        log.warning(f"[warn] peak_pos_thin skipped: {e}")

    # --- τ diagnostics: raw + scale-aligned (factored out) ---
    c_tau = fit_c_tau(G_tau_rec=G_tau_rec, G_tau_noisy=G_tau_noisy)
    c_tau_val = float(c_tau)
    G_tau_rec_scaled = (G_tau_rec * c_tau).to(G_tau_rec.dtype)
    log.info(f"[tau] best-fit c_tau = {float(c_tau):.6g}")

    tau_m_raw, tau_m = compute_tau_metrics_pair(
        G_tau_rec=G_tau_rec,
        G_tau_rec_scaled=G_tau_rec_scaled,
        G_tau_noisy=G_tau_noisy,
        tau=tau,
        beta=beta,
        sigma_tau=sigma_tau,
        tau_margin=5.0,
        topk=8,
    )

    print_tau_diagnostics_summary(
        tau_m_raw=tau_m_raw,
        tau_m=tau_m,
        G_tau_rec=G_tau_rec,
        G_tau_rec_scaled=G_tau_rec_scaled,
        G_tau_noisy=G_tau_noisy,
        N_tau=N_tau,
        log=log,
    )

    # ---- Optional iω diagnostics (reference/debug only) ----
    # NOTE: G_obs is generated by iω-noise injection in this runner.
    # iω diagnostics are now directly aligned with the inference observation model.
    if do_iw_diag:
        log.info("[diag] iω diagnostics: ENABLED (reference/debug)")
        run_iw_diagnostics(
            A_est=A_est,
            omega=omega_sub_t,
            wn=wn,
            G_obs=G_obs,
            sigma_iw=sigma_iw_full,
            log=log,
        )
    else:
        log.info("[diag] iω diagnostics: DISABLED (τ diagnostics only)")
        
    log.debug("[DEBUG] sigma_tau min/med/max: %s %s %s",
              float(sigma_tau.min()), float(sigma_tau.median()), float(sigma_tau.max()))
    log.debug("[DEBUG] G_tau_noisy dtype: %s is_complex=%s", str(G_tau_noisy.dtype), bool(torch.is_complex(G_tau_noisy)))
    log.debug("[DEBUG] induced tau-noise rms (from iomega): %s", float(torch.std(G_tau_noisy - G_tau_true, unbiased=False)))
    
    # Optional forward self-consistency test (debug)
    if do_forward_self_test:
        A_test = D_norm_sub_t  # default model でテスト（A>=0で安定）
        G_tau_omega, G_tau_iw, r_self = forward_self_consistency_test(
            A_w=A_test,
            omega=omega_sub_t,
            tau=tau,
            beta=beta,
            wn_pos=wn_pos
        )
        self_abs_l2 = float(torch.linalg.norm(r_self).item())
        self_den = float(torch.linalg.norm(G_tau_omega).item()) + 1e-30
        self_rel_l2 = self_abs_l2 / self_den
        diagnostics["forward_self_abs_l2"] = self_abs_l2
        diagnostics["forward_self_rel_l2"] = self_rel_l2
        diagnostics["forward_self_enabled"] = True
        log.info("[forward-self] abs_l2=%.6e rel_l2=%.6e", self_abs_l2, self_rel_l2)
        if self_rel_l2 > 1e-1:
            log.warning(
                "[forward-self] rel_l2 is large (%.6e). "
                "Consider increasing prepare.Nn to reduce finite Matsubara truncation error.",
                self_rel_l2,
            )
    else:
        diagnostics["forward_self_enabled"] = False
    
    save_dtype = torch_dtype_from_name(cfg["output"].get("save_float", "float64"))
    save_run_artifacts(
        run_dir=run_dir,
        save_dtype=save_dtype,
        # inputs
        wn=wn,
        G_obs=G_obs,
        omega=omega_sub_t,
        D_w=D_norm_sub_t,
        tau=tau,
        sigma_tau=sigma_tau,
        alphas=alphas,
        # results
        A_est=A_est,
        A_low=A_low,
        A_high=A_high,
        mcse=mcse,
        diagnostics=diagnostics,
        c_tau=c_tau_val,
        tau_metrics_raw=tau_m_raw,
        tau_metrics=tau_m,
        G_tau_noisy=G_tau_noisy,
        G_tau_rec=G_tau_rec,
        peak_pos_roi=peak_pos_roi,
    )

    if do_mcmc_plot:
        try:
            mcmc_out_raw = str(mcmc_plot_cfg.get("out", "mcmc_convergence.png"))
            mcmc_out = Path(mcmc_out_raw)
            if not mcmc_out.is_absolute():
                mcmc_out = run_dir / mcmc_out
            mcmc_title_raw = mcmc_plot_cfg.get("title", None)
            mcmc_title = str(mcmc_title_raw) if mcmc_title_raw not in (None, "") else None
            mcmc_out_saved, mcmc_summary = save_mcmc_convergence_figure(
                run_dir,
                out_path=mcmc_out,
                rhat_threshold=float(mcmc_plot_cfg.get("rhat_threshold", 1.05)),
                ess_min_per_chain=float(mcmc_plot_cfg.get("ess_min_per_chain", 50.0)),
                max_replicas=max(1, int(mcmc_plot_cfg.get("max_replicas", 12))),
                title=mcmc_title,
                dpi=max(72, int(mcmc_plot_cfg.get("dpi", 180))),
            )
            log.info("[mcmc] convergence figure saved: %s", mcmc_out_saved)
            log.info(
                "[mcmc] summary: rhat_max=%.6g ess_min_per_chain=%.6g total_ess=%.6g mcse_max=%.6g",
                float(mcmc_summary.get("rhat_max", float("nan"))),
                float(mcmc_summary.get("ess_min_per_chain", float("nan"))),
                float(mcmc_summary.get("total_ess", float("nan"))),
                float(mcmc_summary.get("mcse_max", float("nan"))),
            )
        except Exception as e:
            log.warning("[warn] mcmc convergence figure skipped: %s", e)

    log.info(f"[OK] run saved to: {run_dir}")
    
    run_postprocess(
        run_dir=run_dir,
        D_true=D_norm_sub_t,
        D_default=m_default,
        roi=roi,
        plot_cfg=cfg.get("plotting", {}) or {},
    )

if __name__ == "__main__":
    main()
