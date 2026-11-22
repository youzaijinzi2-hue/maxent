from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MaxEntConfig:
    """
    Configuration for the MaxEnt sampler.
    """

    wn: torch.Tensor
    beta: float
    tau: torch.Tensor
    sigma_tau: torch.Tensor
    alphas: torch.Tensor
    ws_k: int = 4
    p_ws: float = 0.316
    p_pca: float = 0.300
    sampler_mode: str = "mixed"  # "mixed" or "mh"
    n_step: int = 10000
    step_size_mh: float = 0.05
    step_size_pca: float = 0.05
    step_size_ws: float = 0.05
    step_size_cg: float = 0.05
    burnin_ratio: float = 0.3
    hot_start_ratio: float = 0.5
    adapt_end_ratio: float = 0.8
    adapt_hot_interval: int = 1
    adapt_late_interval: int = 100
    scale_adapt_t0: float = 10.0
    scale_adapt_gamma: float = 0.05
    scale_adapt_kappa: float = 0.75
    scale_min: float = 1e-3
    scale_max: float = 1e1
    accept_ema_kappa: float = 0.80
    prob_adapt_kappa: float = 0.90
    prob_adapt_metric: str = "esjd_per_sec"  # "esjd_per_sec" or "accept_rate"
    prob_min: float = 0.10
    prob_utility_floor: float = 1e-12
    prob_min_trials_per_move: int = 100
    prob_esjd_whitened: bool = True
    prob_whiten_jitter: float = 1e-6
    prob_whiten_shrinkage: float = 0.05
    prob_whiten_weight_mode: str = "count"  # "equal", "count", or "trial"
    prob_swap_reward_weight: float = 0.15
    prob_swap_warmup_updates: int = 10
    cg_cov_blend: float = 0.20
    cg_cov_jitter: float = 1e-6
    swap_freq: int = 3
    use_replica_exchange: bool = True
    target_accept_pca: float = 0.50
    target_accept_ws: float = 0.50
    target_accept_cg: float = 0.50
    # alpha posterior weighting:
    #   "auto" -> prefer Bryan/Laplace if valid, otherwise TI
    #   "bryan_laplace" -> use Bryan/Laplace (fallback TI on failure)
    #   "ti" -> use TI (fallback Bryan/Laplace on failure)
    alpha_weight_method: str = "auto"
    alpha_prior_bryan: str = "flat"
    alpha_prior_ti: str = "flat"
    # main-term proxy in Bryan/Laplace: "map" or "mean"
    alpha_bryan_main_term: str = "map"
    # reference spectrum for Occam-factor lambdas:
    #   "mean" (legacy), "map", "map_local_mc"
    alpha_reference_spectrum: str = "map"
    # used only when alpha_reference_spectrum == "map_local_mc"
    alpha_map_local_samples: int = 64
    alpha_map_local_logp_window: float = 5.0
    alpha_map_local_temperature: float = 1.0
    alpha_map_local_seed: int = 0
    # R-hat diagnostics: temporal split count per replica chain
    rhat_split_segments: int = 4
    # R-hat diagnostics: ignore this many bins on each edge for summary stats
    rhat_exclude_edge_bins: int = 1
    # MCSE/IAT diagnostics tuning (useful on low-memory runtimes like Colab)
    mcse_threshold: float = 0.005
    mcse_check_interval: int = 100
    mcse_iat_window: int = 2000
    iat_final_window: int = 0
    iat_max_fft_bytes: int = 134217728
    device: str | None = None
