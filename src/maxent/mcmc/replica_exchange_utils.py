from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Optional

import numpy as np
import torch


def target_accept_value(
    config: object,
    name: str,
    *,
    device: torch.device,
    default: float = 0.50,
) -> torch.Tensor:
    v = float(getattr(config, name, default))
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    v = min(max(v, 0.01), 0.99)
    return torch.tensor(v, dtype=torch.float64, device=device)


def obs_esjd_per_sec(
    esjd_sum: float,
    n_try: int,
    t_sum: float,
    n_eval: int,
    *,
    utility_floor: float,
) -> Optional[float]:
    if (n_try <= 0) or (n_eval <= 0):
        return None
    if (not math.isfinite(esjd_sum)) or (not math.isfinite(t_sum)):
        return None
    if t_sum <= 0.0:
        return None
    esjd_per_trial = esjd_sum / float(n_try)
    sec_per_eval = t_sum / float(n_eval)
    if sec_per_eval <= 0.0:
        return None
    val = esjd_per_trial / sec_per_eval
    if not math.isfinite(val):
        return None
    return max(val, utility_floor)


def rm_update_pos(
    est: float,
    obs: Optional[float],
    *,
    utility_floor: float,
    gamma: float,
) -> float:
    if obs is None:
        return est
    obs = max(float(obs), utility_floor)
    if (not math.isfinite(est)) or (est <= 0.0):
        return obs
    return est * math.exp(gamma * (math.log(obs) - math.log(est)))


def sum_jump_sq(
    dx_list: list[torch.Tensor],
    P_ref: Optional[torch.Tensor],
    P_ref_chol: Optional[torch.Tensor] = None,
    *,
    device: torch.device,
) -> float:
    if len(dx_list) == 0:
        return 0.0
    dx = torch.cat(dx_list, dim=0).to(device=device, dtype=torch.float64)
    if P_ref is None:
        q = torch.sum(dx * dx, dim=1)
    elif P_ref_chol is not None:
        z = dx @ P_ref_chol
        q = torch.sum(z * z, dim=1)
    else:
        q = torch.sum((dx @ P_ref) * dx, dim=1)
    q = q[torch.isfinite(q)]
    if q.numel() == 0:
        return 0.0
    return float(q.sum().item())


def _n_to_u(n_batch: torch.Tensor, w_x: torch.Tensor) -> torch.Tensor:
    p = torch.clamp(n_batch.to(torch.float64) * w_x.unsqueeze(0), min=1e-300)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-300)
    return torch.log(p)


@dataclass
class ProbAdaptState:
    utility_est_pca: float
    utility_est_ws: float
    utility_est_cg: float
    swap_est_pca: float
    swap_est_ws: float
    swap_est_cg: float
    trials_pca: int = 0
    trials_ws: int = 0
    trials_cg: int = 0
    accepts_pca: int = 0
    accepts_ws: int = 0
    accepts_cg: int = 0
    dx_buf_pca: list[torch.Tensor] = field(default_factory=list)
    dx_buf_ws: list[torch.Tensor] = field(default_factory=list)
    dx_buf_cg: list[torch.Tensor] = field(default_factory=list)
    time_sum_pca: float = 0.0
    time_sum_ws: float = 0.0
    time_sum_cg: float = 0.0
    eval_count_pca: int = 0
    eval_count_ws: int = 0
    eval_count_cg: int = 0
    swap_credit_pca: float = 0.0
    swap_credit_ws: float = 0.0
    swap_credit_cg: float = 0.0
    last_obs_pca: float = float("nan")
    last_obs_ws: float = float("nan")
    last_obs_cg: float = float("nan")
    last_swap_obs_pca: float = float("nan")
    last_swap_obs_ws: float = float("nan")
    last_swap_obs_cg: float = float("nan")
    history_steps: list[int] = field(default_factory=list)
    history_pca: list[float] = field(default_factory=list)
    history_ws: list[float] = field(default_factory=list)
    history_cg: list[float] = field(default_factory=list)
    history_utility_obs_pca: list[float] = field(default_factory=list)
    history_utility_obs_ws: list[float] = field(default_factory=list)
    history_utility_obs_cg: list[float] = field(default_factory=list)
    history_utility_est_pca: list[float] = field(default_factory=list)
    history_utility_est_ws: list[float] = field(default_factory=list)
    history_utility_est_cg: list[float] = field(default_factory=list)

    @classmethod
    def initialize(
        cls,
        *,
        p_pca: float,
        p_ws: float,
        p_cg: float,
        utility_floor: float,
    ) -> "ProbAdaptState":
        utility_est_pca = max(float(p_pca), utility_floor)
        utility_est_ws = max(float(p_ws), utility_floor)
        utility_est_cg = max(float(p_cg), utility_floor)
        return cls(
            utility_est_pca=utility_est_pca,
            utility_est_ws=utility_est_ws,
            utility_est_cg=utility_est_cg,
            swap_est_pca=utility_est_pca,
            swap_est_ws=utility_est_ws,
            swap_est_cg=utility_est_cg,
        )

    def accumulate_timing(self, *, dt_pca: float, dt_ws: float, dt_cg: float) -> None:
        self.time_sum_pca += float(dt_pca)
        self.time_sum_ws += float(dt_ws)
        self.time_sum_cg += float(dt_cg)
        self.eval_count_pca += 1
        self.eval_count_ws += 1
        self.eval_count_cg += 1

    def accumulate_moves(
        self,
        *,
        cond_pca: torch.Tensor,
        cond_ws: torch.Tensor,
        cond_cg: torch.Tensor,
        acc_pca: torch.Tensor,
        acc_ws: torch.Tensor,
        acc_cg: torch.Tensor,
        prop_n_pca: torch.Tensor,
        prop_n_ws: torch.Tensor,
        prop_n_cg: torch.Tensor,
        n_current: torch.Tensor,
        w_x: torch.Tensor,
    ) -> None:
        self.trials_pca += int(cond_pca.sum().item())
        self.trials_ws += int(cond_ws.sum().item())
        self.trials_cg += int(cond_cg.sum().item())
        self.accepts_pca += int(acc_pca.sum().item())
        self.accepts_ws += int(acc_ws.sum().item())
        self.accepts_cg += int(acc_cg.sum().item())
        w_x = torch.clamp(w_x.to(device=n_current.device, dtype=torch.float64), min=1e-30)
        if acc_pca.any():
            u_prop = _n_to_u(prop_n_pca[acc_pca].detach(), w_x)
            u_cur = _n_to_u(n_current[acc_pca].detach(), w_x)
            self.dx_buf_pca.append((u_prop - u_cur).to(torch.float64))
        if acc_ws.any():
            u_prop = _n_to_u(prop_n_ws[acc_ws].detach(), w_x)
            u_cur = _n_to_u(n_current[acc_ws].detach(), w_x)
            self.dx_buf_ws.append((u_prop - u_cur).to(torch.float64))
        if acc_cg.any():
            u_prop = _n_to_u(prop_n_cg[acc_cg].detach(), w_x)
            u_cur = _n_to_u(n_current[acc_cg].detach(), w_x)
            self.dx_buf_cg.append((u_prop - u_cur).to(torch.float64))

    def accumulate_swap_credit(self, swap_credit: torch.Tensor) -> None:
        self.swap_credit_pca += float(swap_credit[0])
        self.swap_credit_ws += float(swap_credit[1])
        self.swap_credit_cg += float(swap_credit[2])

    def min_trials(self) -> int:
        return min(self.trials_pca, self.trials_ws, self.trials_cg)

    def reset_window(self) -> None:
        self.trials_pca = 0
        self.trials_ws = 0
        self.trials_cg = 0
        self.accepts_pca = 0
        self.accepts_ws = 0
        self.accepts_cg = 0
        self.dx_buf_pca.clear()
        self.dx_buf_ws.clear()
        self.dx_buf_cg.clear()
        self.time_sum_pca = 0.0
        self.time_sum_ws = 0.0
        self.time_sum_cg = 0.0
        self.eval_count_pca = 0
        self.eval_count_ws = 0
        self.eval_count_cg = 0
        self.swap_credit_pca = 0.0
        self.swap_credit_ws = 0.0
        self.swap_credit_cg = 0.0


def select_move_masks(
    u: torch.Tensor,
    p_pca: float,
    p_ws: float,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cond_pca = u < p_pca
    cond_ws = (u >= p_pca) & (u < p_pca + p_ws)
    cond_cg = u >= (p_pca + p_ws)
    move_idx = torch.where(
        cond_pca,
        torch.zeros((u.shape[0],), device=device, dtype=torch.int64),
        torch.where(
            cond_ws,
            torch.ones((u.shape[0],), device=device, dtype=torch.int64),
            torch.full((u.shape[0],), 2, device=device, dtype=torch.int64),
        ),
    )
    return cond_pca, cond_ws, cond_cg, move_idx


def build_whiten_precision(
    C_pca: torch.Tensor,
    C_ws: torch.Tensor,
    C_cg: torch.Tensor,
    *,
    w_pca: float,
    w_ws: float,
    w_cg: float,
    device: torch.device,
    jitter: float,
    shrinkage: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = np.asarray([w_pca, w_ws, w_cg], dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights >= 0.0), weights, 0.0)
    w_sum = float(weights.sum())
    if w_sum <= 0.0:
        weights[:] = 1.0
        w_sum = 3.0
    C_ref = (
        weights[0] * C_pca.to(torch.float64)
        + weights[1] * C_ws.to(torch.float64)
        + weights[2] * C_cg.to(torch.float64)
    ) / w_sum
    C_ref = torch.nan_to_num(C_ref, nan=0.0, posinf=0.0, neginf=0.0)
    C_ref = 0.5 * (C_ref + C_ref.T)
    lam = float(shrinkage)
    lam = min(max(lam, 0.0), 1.0)
    I_ref = torch.eye(C_ref.shape[0], device=device, dtype=torch.float64)
    if lam > 0.0:
        tau = float(torch.trace(C_ref).item()) / float(C_ref.shape[0])
        if (not math.isfinite(tau)) or (tau <= 0.0):
            tau = 1.0
        C_ref = (1.0 - lam) * C_ref + (lam * tau) * I_ref
    jit = max(float(jitter), float(torch.finfo(C_ref.dtype).eps))
    info_ref = torch.tensor(1, device=device, dtype=torch.int32)
    for _ in range(8):
        L_cov, info_ref = torch.linalg.cholesky_ex(C_ref + jit * I_ref)
        if int(info_ref.item()) == 0:
            break
        jit *= 10.0
    if int(info_ref.item()) != 0:
        evals, evecs = torch.linalg.eigh(C_ref)
        evals = torch.clamp(evals, min=jit)
        C_spd = evecs @ torch.diag(evals) @ evecs.T
        L_cov = torch.linalg.cholesky(C_spd)
    P_ref = torch.cholesky_inverse(L_cov)
    P_ref = 0.5 * (P_ref + P_ref.T)
    jit_prec = max(float(jitter), float(torch.finfo(P_ref.dtype).eps))
    info_prec = torch.tensor(1, device=device, dtype=torch.int32)
    for _ in range(8):
        L_prec, info_prec = torch.linalg.cholesky_ex(P_ref + jit_prec * I_ref)
        if int(info_prec.item()) == 0:
            break
        jit_prec *= 10.0
    if int(info_prec.item()) != 0:
        evals_p, evecs_p = torch.linalg.eigh(P_ref)
        evals_p = torch.clamp(evals_p, min=jit_prec)
        P_spd = evecs_p @ torch.diag(evals_p) @ evecs_p.T
        P_ref = P_spd
        L_prec = torch.linalg.cholesky(P_spd)
    return P_ref, L_prec


def adapt_move_probabilities(
    *,
    state: ProbAdaptState,
    p_pca: float,
    p_ws: float,
    p_cg: float,
    t: int,
    gamma_prob: float,
    prob_metric: str,
    prob_min_trials_per_move: int,
    prob_esjd_whitened: bool,
    prob_whiten_jitter: float,
    prob_whiten_shrinkage: float,
    prob_whiten_weight_mode: str,
    prob_swap_reward_weight: float,
    prob_swap_warmup_updates: int,
    prob_utility_floor: float,
    C_pca: torch.Tensor,
    C_ws: torch.Tensor,
    C_cg: torch.Tensor,
    count_pca: int,
    count_ws: int,
    count_cg: int,
    device: torch.device,
) -> tuple[float, float, float]:
    if state.min_trials() < prob_min_trials_per_move:
        return p_pca, p_ws, p_cg

    if prob_metric == "accept_rate":
        obs_pca = (state.accepts_pca / state.trials_pca) if state.trials_pca > 0 else None
        obs_ws = (state.accepts_ws / state.trials_ws) if state.trials_ws > 0 else None
        obs_cg = (state.accepts_cg / state.trials_cg) if state.trials_cg > 0 else None
    else:
        P_ref = None
        P_ref_chol = None
        if prob_esjd_whitened:
            weight_mode = str(prob_whiten_weight_mode).lower()
            if weight_mode == "trial":
                w_pca = float(state.trials_pca)
                w_ws = float(state.trials_ws)
                w_cg = float(state.trials_cg)
            elif weight_mode == "equal":
                w_pca = 1.0
                w_ws = 1.0
                w_cg = 1.0
            else:
                w_pca = float(count_pca)
                w_ws = float(count_ws)
                w_cg = float(count_cg)
            P_ref, P_ref_chol = build_whiten_precision(
                C_pca,
                C_ws,
                C_cg,
                w_pca=w_pca,
                w_ws=w_ws,
                w_cg=w_cg,
                device=device,
                jitter=prob_whiten_jitter,
                shrinkage=prob_whiten_shrinkage,
            )
        esjd_sum_pca = sum_jump_sq(state.dx_buf_pca, P_ref, P_ref_chol=P_ref_chol, device=device)
        esjd_sum_ws = sum_jump_sq(state.dx_buf_ws, P_ref, P_ref_chol=P_ref_chol, device=device)
        esjd_sum_cg = sum_jump_sq(state.dx_buf_cg, P_ref, P_ref_chol=P_ref_chol, device=device)
        obs_pca = obs_esjd_per_sec(
            esjd_sum_pca,
            state.trials_pca,
            state.time_sum_pca,
            state.eval_count_pca,
            utility_floor=prob_utility_floor,
        )
        obs_ws = obs_esjd_per_sec(
            esjd_sum_ws,
            state.trials_ws,
            state.time_sum_ws,
            state.eval_count_ws,
            utility_floor=prob_utility_floor,
        )
        obs_cg = obs_esjd_per_sec(
            esjd_sum_cg,
            state.trials_cg,
            state.time_sum_cg,
            state.eval_count_cg,
            utility_floor=prob_utility_floor,
        )

    swap_obs_pca = (state.swap_credit_pca / state.trials_pca) if state.trials_pca > 0 else None
    swap_obs_ws = (state.swap_credit_ws / state.trials_ws) if state.trials_ws > 0 else None
    swap_obs_cg = (state.swap_credit_cg / state.trials_cg) if state.trials_cg > 0 else None

    state.last_obs_pca = float(obs_pca) if obs_pca is not None else float("nan")
    state.last_obs_ws = float(obs_ws) if obs_ws is not None else float("nan")
    state.last_obs_cg = float(obs_cg) if obs_cg is not None else float("nan")
    state.last_swap_obs_pca = float(swap_obs_pca) if swap_obs_pca is not None else float("nan")
    state.last_swap_obs_ws = float(swap_obs_ws) if swap_obs_ws is not None else float("nan")
    state.last_swap_obs_cg = float(swap_obs_cg) if swap_obs_cg is not None else float("nan")

    state.utility_est_pca = rm_update_pos(
        state.utility_est_pca, obs_pca, utility_floor=prob_utility_floor, gamma=gamma_prob
    )
    state.utility_est_ws = rm_update_pos(
        state.utility_est_ws, obs_ws, utility_floor=prob_utility_floor, gamma=gamma_prob
    )
    state.utility_est_cg = rm_update_pos(
        state.utility_est_cg, obs_cg, utility_floor=prob_utility_floor, gamma=gamma_prob
    )
    state.swap_est_pca = rm_update_pos(
        state.swap_est_pca, swap_obs_pca, utility_floor=prob_utility_floor, gamma=gamma_prob
    )
    state.swap_est_ws = rm_update_pos(
        state.swap_est_ws, swap_obs_ws, utility_floor=prob_utility_floor, gamma=gamma_prob
    )
    state.swap_est_cg = rm_update_pos(
        state.swap_est_cg, swap_obs_cg, utility_floor=prob_utility_floor, gamma=gamma_prob
    )

    utility_sum = state.utility_est_pca + state.utility_est_ws + state.utility_est_cg + prob_utility_floor
    utility_norm_pca = state.utility_est_pca / utility_sum
    utility_norm_ws = state.utility_est_ws / utility_sum
    utility_norm_cg = state.utility_est_cg / utility_sum
    target_pca = utility_norm_pca
    target_ws = utility_norm_ws
    target_cg = utility_norm_cg
    if prob_swap_reward_weight > 0.0:
        swap_sum = state.swap_est_pca + state.swap_est_ws + state.swap_est_cg + prob_utility_floor
        swap_norm_pca = state.swap_est_pca / swap_sum
        swap_norm_ws = state.swap_est_ws / swap_sum
        swap_norm_cg = state.swap_est_cg / swap_sum
        w_swap = float(prob_swap_reward_weight)
        n_warm = max(0, int(prob_swap_warmup_updates))
        if n_warm > 0:
            n_updates = len(state.history_steps) + 1
            w_swap *= min(1.0, float(n_updates) / float(n_warm))
        target_pca = (1.0 - w_swap) * utility_norm_pca + w_swap * swap_norm_pca
        target_ws = (1.0 - w_swap) * utility_norm_ws + w_swap * swap_norm_ws
        target_cg = (1.0 - w_swap) * utility_norm_cg + w_swap * swap_norm_cg

    p_pca = p_pca + gamma_prob * (target_pca - p_pca)
    p_ws = p_ws + gamma_prob * (target_ws - p_ws)
    p_cg = p_cg + gamma_prob * (target_cg - p_cg)

    state.history_steps.append(int(t))
    state.history_pca.append(float(p_pca))
    state.history_ws.append(float(p_ws))
    state.history_cg.append(float(p_cg))
    state.history_utility_obs_pca.append(float(obs_pca) if obs_pca is not None else float("nan"))
    state.history_utility_obs_ws.append(float(obs_ws) if obs_ws is not None else float("nan"))
    state.history_utility_obs_cg.append(float(obs_cg) if obs_cg is not None else float("nan"))
    state.history_utility_est_pca.append(float(state.utility_est_pca))
    state.history_utility_est_ws.append(float(state.utility_est_ws))
    state.history_utility_est_cg.append(float(state.utility_est_cg))
    state.reset_window()
    return p_pca, p_ws, p_cg


def clamp_step_sizes(
    s_pca: float,
    s_ws: float,
    s_cg: float,
    *,
    step_min: float,
    step_max: float,
) -> tuple[float, float, float]:
    def _sanitize(v: float) -> float:
        if (not math.isfinite(v)) or (v <= 0.0):
            v = step_min
        return min(max(v, step_min), step_max)

    return _sanitize(s_pca), _sanitize(s_ws), _sanitize(s_cg)


def clamp_and_normalize_probs(
    p_pca: float,
    p_ws: float,
    p_cg: float,
    *,
    prob_min: float,
) -> tuple[float, float, float]:
    eps = float(prob_min)
    p_pca = max(eps, min(p_pca, 1.0))
    p_ws = max(eps, min(p_ws, 1.0))
    p_cg = max(eps, min(p_cg, 1.0))
    p_sum = p_pca + p_ws + p_cg
    if (not math.isfinite(p_sum)) or (p_sum <= 0.0):
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return p_pca / p_sum, p_ws / p_sum, p_cg / p_sum


def flush_cov_buffers(
    *,
    cov_buf_pca: list[torch.Tensor],
    cov_buf_ws: list[torch.Tensor],
    cov_buf_cg: list[torch.Tensor],
    mu_pca: torch.Tensor,
    C_pca: torch.Tensor,
    count_pca: int,
    mu_ws: torch.Tensor,
    C_ws: torch.Tensor,
    count_ws: int,
    mu_cg: torch.Tensor,
    C_cg: torch.Tensor,
    count_cg: int,
    online_stats_update: Callable[
        [torch.Tensor, torch.Tensor, int, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, int],
    ],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    if len(cov_buf_pca) > 0:
        x_accept_pca = torch.cat(cov_buf_pca, dim=0)
        mu_pca, C_pca, count_pca = online_stats_update(mu_pca, C_pca, count_pca, x_accept_pca)
        cov_buf_pca.clear()
    if len(cov_buf_ws) > 0:
        x_accept_ws = torch.cat(cov_buf_ws, dim=0)
        mu_ws, C_ws, count_ws = online_stats_update(mu_ws, C_ws, count_ws, x_accept_ws)
        cov_buf_ws.clear()
    if len(cov_buf_cg) > 0:
        x_accept_cg = torch.cat(cov_buf_cg, dim=0)
        mu_cg, C_cg, count_cg = online_stats_update(mu_cg, C_cg, count_cg, x_accept_cg)
        cov_buf_cg.clear()
    return (
        mu_pca,
        C_pca,
        count_pca,
        mu_ws,
        C_ws,
        count_ws,
        mu_cg,
        C_cg,
        count_cg,
    )


def populate_run_diagnostics(
    *,
    diagnostics: dict,
    adapt_end: int,
    hot_start: int,
    burnin: int,
    hot_start_ratio: float,
    adapt_end_ratio: float,
    n_steps: int,
    check_interval: int,
    mcse_iat_window: int,
    iat_final_window: int,
    iat_max_fft_bytes: int,
    periodic_iat_error: str,
    final_iat_error: str,
    trials: dict,
    accepts: dict,
    swap_trials: int,
    swap_accepts: int,
    swap_calls: int,
    alpha_star_pca: torch.Tensor,
    alpha_star_ws: torch.Tensor,
    alpha_star_cg: torch.Tensor,
    step_adapter_pca,
    step_adapter_ws,
    step_adapter_cg,
    s_pca: float,
    s_ws: float,
    s_cg: float,
    adapt_hot_interval: int,
    adapt_late_interval: int,
    scale_adapt_t0: float,
    scale_adapt_gamma: float,
    scale_adapt_kappa: float,
    step_min: float,
    step_max: float,
    cg_cov_blend: float,
    cg_cov_jitter: float,
    prob_metric: str,
    prob_min: float,
    prob_min_trials_per_move: int,
    prob_esjd_whitened: bool,
    prob_whiten_jitter: float,
    prob_whiten_shrinkage: float,
    prob_whiten_weight_mode: str,
    prob_swap_reward_weight: float,
    prob_swap_warmup_updates: int,
    prob_utility_floor: float,
    utility_est_pca: float,
    utility_est_ws: float,
    utility_est_cg: float,
    swap_est_pca: float,
    swap_est_ws: float,
    swap_est_cg: float,
    prob_last_obs_pca: float,
    prob_last_obs_ws: float,
    prob_last_obs_cg: float,
    prob_last_swap_obs_pca: float,
    prob_last_swap_obs_ws: float,
    prob_last_swap_obs_cg: float,
    prob_trials_pca: int,
    prob_trials_ws: int,
    prob_trials_cg: int,
    p_pca: float,
    p_ws: float,
    p_cg: float,
    prob_history_steps: list[int],
    prob_history_pca: list[float],
    prob_history_ws: list[float],
    prob_history_cg: list[float],
    prob_history_utility_obs_pca: list[float],
    prob_history_utility_obs_ws: list[float],
    prob_history_utility_obs_cg: list[float],
    prob_history_utility_est_pca: list[float],
    prob_history_utility_est_ws: list[float],
    prob_history_utility_est_cg: list[float],
    move_diag_summary: dict,
    tau_ints: torch.Tensor,
    ess_per_chain: torch.Tensor,
    total_ess: float,
    elapsed_sec: float,
    ess_per_sec: float,
    rhat_split_segments: int,
    rhat_exclude_edge_bins_requested: int,
    rhat_exclude_edge_bins_applied: int,
    rhat_eval_bins: int,
    rhat_max: torch.Tensor,
    rhat_mean: torch.Tensor,
    rhat_by_replica: torch.Tensor,
    mcse_d: torch.Tensor,
    mcse_threshold: float,
    mcse_achieved_step: Optional[int],
    mcse_achieved_time: Optional[float],
    H_samples: torch.Tensor,
) -> dict:
    diagnostics["adapt_end"] = int(adapt_end)
    diagnostics["hot_start"] = int(hot_start)
    diagnostics["burnin"] = int(burnin)
    diagnostics["hot_start_ratio"] = float(hot_start_ratio)
    diagnostics["adapt_end_ratio"] = float(adapt_end_ratio)
    diagnostics["n_steps"] = int(n_steps)
    diagnostics["n_step"] = int(n_steps)
    diagnostics["mcse_check_interval"] = int(check_interval)
    diagnostics["mcse_iat_window"] = int(mcse_iat_window)
    diagnostics["iat_final_window"] = int(iat_final_window)
    diagnostics["iat_max_fft_bytes"] = int(iat_max_fft_bytes)
    if periodic_iat_error:
        diagnostics["mcse_periodic_check_error"] = str(periodic_iat_error)
    if final_iat_error:
        diagnostics["iat_final_error"] = str(final_iat_error)

    for move in ["pca", "ws", "cg"]:
        rate = accepts[move] / trials[move] if trials[move] > 0 else float("nan")
        diagnostics[f"{move}_accept_rate"] = float(rate)
    diagnostics["swap_trial_count"] = int(swap_trials)
    diagnostics["swap_accept_count"] = int(swap_accepts)
    diagnostics["swap_call_count"] = int(swap_calls)
    diagnostics["swap_accept_rate"] = float(
        (swap_accepts / swap_trials) if swap_trials > 0 else float("nan")
    )
    diagnostics["target_accept_pca"] = float(alpha_star_pca.detach().cpu().item())
    diagnostics["target_accept_ws"] = float(alpha_star_ws.detach().cpu().item())
    diagnostics["target_accept_cg"] = float(alpha_star_cg.detach().cpu().item())
    diagnostics["step_size_pca_init"] = float(step_adapter_pca.step_init)
    diagnostics["step_size_ws_init"] = float(step_adapter_ws.step_init)
    diagnostics["step_size_cg_init"] = float(step_adapter_cg.step_init)
    diagnostics["step_size_pca_final"] = float(s_pca)
    diagnostics["step_size_ws_final"] = float(s_ws)
    diagnostics["step_size_cg_final"] = float(s_cg)
    diagnostics["adapt_hot_interval"] = int(adapt_hot_interval)
    diagnostics["adapt_late_interval"] = int(adapt_late_interval)
    diagnostics["scale_adapt_t0"] = float(scale_adapt_t0)
    diagnostics["scale_adapt_gamma"] = float(scale_adapt_gamma)
    diagnostics["scale_adapt_kappa"] = float(scale_adapt_kappa)
    diagnostics["scale_min"] = float(step_min)
    diagnostics["scale_max"] = float(step_max)
    diagnostics["cg_cov_blend"] = float(cg_cov_blend)
    diagnostics["cg_cov_jitter"] = float(cg_cov_jitter)
    diagnostics["prob_adapt_metric"] = str(prob_metric)
    diagnostics["prob_min"] = float(prob_min)
    diagnostics["prob_min_trials_per_move"] = int(prob_min_trials_per_move)
    diagnostics["prob_esjd_whitened"] = bool(prob_esjd_whitened)
    diagnostics["prob_whiten_jitter"] = float(prob_whiten_jitter)
    diagnostics["prob_whiten_shrinkage"] = float(prob_whiten_shrinkage)
    diagnostics["prob_whiten_weight_mode"] = str(prob_whiten_weight_mode)
    diagnostics["prob_swap_reward_weight"] = float(prob_swap_reward_weight)
    diagnostics["prob_swap_warmup_updates"] = int(prob_swap_warmup_updates)
    diagnostics["prob_utility_floor"] = float(prob_utility_floor)
    diagnostics["prob_utility_est_pca_final"] = float(utility_est_pca)
    diagnostics["prob_utility_est_ws_final"] = float(utility_est_ws)
    diagnostics["prob_utility_est_cg_final"] = float(utility_est_cg)
    diagnostics["prob_swap_est_pca_final"] = float(swap_est_pca)
    diagnostics["prob_swap_est_ws_final"] = float(swap_est_ws)
    diagnostics["prob_swap_est_cg_final"] = float(swap_est_cg)
    diagnostics["prob_utility_obs_pca_last"] = float(prob_last_obs_pca)
    diagnostics["prob_utility_obs_ws_last"] = float(prob_last_obs_ws)
    diagnostics["prob_utility_obs_cg_last"] = float(prob_last_obs_cg)
    diagnostics["prob_swap_obs_pca_last"] = float(prob_last_swap_obs_pca)
    diagnostics["prob_swap_obs_ws_last"] = float(prob_last_swap_obs_ws)
    diagnostics["prob_swap_obs_cg_last"] = float(prob_last_swap_obs_cg)
    diagnostics["prob_trials_pending_pca"] = int(prob_trials_pca)
    diagnostics["prob_trials_pending_ws"] = int(prob_trials_ws)
    diagnostics["prob_trials_pending_cg"] = int(prob_trials_cg)
    diagnostics["move_prob_pca_final"] = float(p_pca)
    diagnostics["move_prob_ws_final"] = float(p_ws)
    diagnostics["move_prob_cg_final"] = float(p_cg)
    diagnostics["prob_history_steps"] = np.asarray(prob_history_steps, dtype=np.int64)
    diagnostics["move_prob_pca_history"] = np.asarray(prob_history_pca, dtype=np.float64)
    diagnostics["move_prob_ws_history"] = np.asarray(prob_history_ws, dtype=np.float64)
    diagnostics["move_prob_cg_history"] = np.asarray(prob_history_cg, dtype=np.float64)
    diagnostics["prob_utility_obs_pca_history"] = np.asarray(
        prob_history_utility_obs_pca, dtype=np.float64
    )
    diagnostics["prob_utility_obs_ws_history"] = np.asarray(
        prob_history_utility_obs_ws, dtype=np.float64
    )
    diagnostics["prob_utility_obs_cg_history"] = np.asarray(
        prob_history_utility_obs_cg, dtype=np.float64
    )
    diagnostics["prob_utility_est_pca_history"] = np.asarray(
        prob_history_utility_est_pca, dtype=np.float64
    )
    diagnostics["prob_utility_est_ws_history"] = np.asarray(
        prob_history_utility_est_ws, dtype=np.float64
    )
    diagnostics["prob_utility_est_cg_history"] = np.asarray(
        prob_history_utility_est_cg, dtype=np.float64
    )
    diagnostics["prob_history_length"] = int(len(prob_history_steps))

    diagnostics["move_diag"] = move_diag_summary
    diagnostics["tau_ints"] = tau_ints.detach().cpu().numpy()
    diagnostics["ess_per_chain"] = ess_per_chain.detach().cpu().numpy()
    diagnostics["total_ess"] = float(total_ess)
    diagnostics["elapsed_sec"] = float(elapsed_sec)
    diagnostics["ess_per_sec"] = float(ess_per_sec)

    diagnostics["rhat_method"] = "split_within_replica"
    diagnostics["rhat_split_segments"] = int(rhat_split_segments)
    diagnostics["rhat_exclude_edge_bins_requested"] = int(rhat_exclude_edge_bins_requested)
    diagnostics["rhat_exclude_edge_bins_applied"] = int(rhat_exclude_edge_bins_applied)
    diagnostics["rhat_eval_bins"] = int(rhat_eval_bins)
    diagnostics["rhat_max"] = float(rhat_max.detach().cpu().item())
    diagnostics["rhat_mean"] = float(rhat_mean.detach().cpu().item())
    rhat_np = rhat_by_replica.detach().cpu().numpy()
    if np.isfinite(rhat_np).any():
        diagnostics["rhat_per_replica_max"] = np.nanmax(rhat_np, axis=1)
        diagnostics["rhat_per_replica_mean"] = np.nanmean(rhat_np, axis=1)
    else:
        diagnostics["rhat_per_replica_max"] = np.full((rhat_np.shape[0],), np.nan, dtype=np.float64)
        diagnostics["rhat_per_replica_mean"] = np.full((rhat_np.shape[0],), np.nan, dtype=np.float64)

    diagnostics["mcse_d"] = mcse_d.detach().cpu().numpy()
    diagnostics["mcse_mean"] = float(mcse_d.mean().detach().cpu().item())
    diagnostics["mcse_max"] = float(mcse_d.max().detach().cpu().item())
    diagnostics["mcse_threshold"] = float(mcse_threshold)
    diagnostics["mcse_achieved_step"] = int(mcse_achieved_step) if mcse_achieved_step is not None else -1
    diagnostics["mcse_achieved_time"] = (
        float(mcse_achieved_time) if mcse_achieved_time is not None else float("nan")
    )

    try:
        max_keep = 2000
        total_len = int(H_samples.shape[1])
        if total_len > 0:
            thin_stride = int(math.ceil(total_len / max_keep))
            thin_stride = max(1, thin_stride)
            thin_idx = torch.arange(0, total_len, thin_stride, device=H_samples.device)
            diagnostics["H_samples_thin"] = (
                H_samples.index_select(1, thin_idx).detach().cpu().numpy()
            )
            diagnostics["H_thin_stride"] = int(thin_stride)
            diagnostics["H_thin_start"] = int(0)
            diagnostics["H_thin_indices"] = thin_idx.detach().cpu().numpy().astype(np.int64)
    except Exception as e:
        diagnostics["H_samples_thin_error"] = str(e)

    return diagnostics
