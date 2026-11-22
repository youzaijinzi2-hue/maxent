"""Core MaxEnt sampler.

This module owns the full sampling lifecycle:
1. initialize chains and per-move proposal state
2. run replica-local proposals + periodic replica exchange
3. adapt move step sizes/probabilities during scheduled windows
4. aggregate posterior summaries and diagnostics
5. reconstruct final spectrum and credible intervals

The heavy math kernels are delegated to:
- `Operators` for forward model / objective terms
- `moves.*` for concrete proposal mechanisms
- `replica_exchange_utils` for adaptation bookkeeping helpers
"""

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import List, Tuple, Callable, Optional
import math
import time
import os
# import matplotlib.pyplot as plt
from maxent.operators.forward import Operators
# from maxent.preprocess.xas_preprocess import Prepare
from maxent.mcmc.adaptation import DualAveragingStepAdapter
from maxent.mcmc.alpha_posterior import (
    compute_lambdas_jerrell as compute_lambdas_jerrell_impl,
    select_alpha_weights,
)
from maxent.mcmc.config import MaxEntConfig
from maxent.mcmc.move_diagnostics import MoveDiagAccumulator
from maxent.mcmc.moves.base import MoveResult
from maxent.mcmc.moves.pca import PCAMove
from maxent.mcmc.moves.random_walk_mh import RandomWalkMHMove
from maxent.mcmc.moves.weight_sharing import WeightSharingMove
from maxent.mcmc.moves.mala import CGMALAMove
from maxent.mcmc.reconstruction import (
    credible_interval_from_samples,
    reconstruct_posterior_mean,
)
from maxent.mcmc.replica_exchange_utils import (
    ProbAdaptState,
    adapt_move_probabilities,
    clamp_and_normalize_probs,
    clamp_step_sizes,
    flush_cov_buffers,
    populate_run_diagnostics,
    select_move_masks,
    target_accept_value,
)
from maxent.diagnostics.ess import AutocorrTime
from maxent.diagnostics.rhat import GelmanRubin
from maxent.mcmc.xparam import y_to_x, x_to_y
from tqdm import tqdm


def _trapz_weights(grid: torch.Tensor) -> torch.Tensor:
    """Trapezoidal weights such that trapz(f, grid) == sum(w * f)."""
    g = grid.to(torch.float64)
    n = g.numel()
    if n == 0:
        return g
    if n == 1:
        return torch.ones_like(g, dtype=torch.float64)
    dx = g[1:] - g[:-1]
    w = torch.empty_like(g, dtype=torch.float64)
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    if n > 2:
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return w


class MaxEnt:
    def __init__(self,
                 G_iw: torch.Tensor,
                 phi: Callable[[torch.Tensor], torch.Tensor],
                 phi_inv: Callable[[torch.Tensor], torch.Tensor],
                 jacobian: Callable[[torch.Tensor], torch.Tensor],
                 D_w: torch.Tensor,
                 omega: torch.Tensor,
                 config: MaxEntConfig
                 ):
        """Initialize the MaxEnt sampler.

        Parameters
        ----------
        G_iw : torch.Tensor
            Matsubara-domain Green's function.
        phi, phi_inv : Callable
            Forward/inverse mappings between omega and x in [0, 1].
        jacobian : Callable
            Jacobian of the mapping.
        D_w : torch.Tensor
            Default model on omega grid.
        omega : torch.Tensor
            Frequency grid.
        config : MaxEntConfig
            Sampler and adaptation configuration.
        """
        # Runtime device policy:
        # - keep tensors on one device to avoid host/device copies in the loop
        # - keep diagnostics numerically stable with float64 where needed
        self.device = torch.device(config.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16 if self.device_type == 'cuda' else torch.bfloat16

        # Problem inputs on target device.
        self.G_iw = G_iw.to(self.device)
        self.phi = phi
        self.phi_inv = phi_inv
        self.jacobian = jacobian
        self.D_w = D_w.to(self.device)
        self.omega = omega.to(self.device)

        # User configuration and static grids.
        self.config = config
        self.wn = config.wn.to(self.device)
        self.beta = config.beta

        # Sampling hyperparameters and replica settings.
        self.tau = config.tau.to(self.device)
        self.sigma = config.sigma_tau.to(self.device)
        self.alphas = config.alphas.to(self.device)
        self.ws_k = config.ws_k
        self.sampler_mode = str(getattr(config, "sampler_mode", "mixed")).lower()
        if self.sampler_mode not in ("mixed", "mh"):
            self.sampler_mode = "mixed"

        self.p_ws = float(config.p_ws)
        self.p_pca = float(config.p_pca)
        self.p_cg = 1.0 - self.p_ws - self.p_pca
        if self.sampler_mode == "mh":
            self.p_pca, self.p_ws, self.p_cg = 1.0, 0.0, 0.0
        self.n_step = config.n_step
        self.swap_freq = config.swap_freq
        self.use_replica_exchange = bool(getattr(config, "use_replica_exchange", True))

        # Parameter dimension on the discretized x-grid.
        d = omega.numel()

        # Default random-walk scale factor used by some proposal policies.
        self.gamma = 2.38 / math.sqrt(d)
        # self.step_size_pca = config.step_size_pca * self.gamma
        # self.step_size_ws = config.step_size_ws * self.gamma
        # self.step_size_cg = config.step_size_cg * self.gamma

        self.burnin_ratio = config.burnin_ratio

        # Deterministic initialization in transformed coordinate.
        self.x = self.phi(omega).to(self.device)

        # Cache G(tau) once; reused in each CG/MALA proposal call.
        self.G_tau = self._compute_G_tau()


        self.D = self.x.shape[0]

        # Global adaptation step (kept for compatibility/inspection).
        self.adapt_step = 0

        # Per-move running covariance and mean for proposal preconditioning.
        self.C_pca = torch.eye(self.D, dtype=torch.float32, device=self.device)
        self.C_ws  = torch.eye(self.D, dtype=torch.float32, device=self.device)
        self.C_cg  = torch.eye(self.D, dtype=torch.float32, device=self.device)

        self.mu_pca = torch.zeros(self.D, dtype=torch.float32, device=self.device)
        self.mu_ws  = torch.zeros(self.D, dtype=torch.float32, device=self.device)
        self.mu_cg  = torch.zeros(self.D, dtype=torch.float32, device=self.device)

        self.count_pca = 0
        self.count_ws  = 0
        self.count_cg  = 0
        self.acc_count_pca = 0
        self.acc_count_ws = 0
        self.acc_count_cg = 0

        # Sanitize any user-provided step size values.
        def _clean_step(value: float, fallback: float = 0.05) -> float:
            v = float(value)
            if (not math.isfinite(v)) or (v <= 0.0):
                return fallback
            return v

        # Hard bounds for proposal step-size adaptation.
        self.step_min = _clean_step(getattr(config, "scale_min", 1e-3), fallback=1e-3)
        self.step_max = _clean_step(getattr(config, "scale_max", 1e1), fallback=1e1)
        if self.step_max <= self.step_min:
            self.step_max = max(self.step_min * 10.0, 1e1)

        # Initial per-move proposal scales (already clamped).
        mh_step = min(
            max(_clean_step(getattr(config, "step_size_mh", config.step_size_pca)), self.step_min),
            self.step_max,
        )
        self.s_pca = min(max(_clean_step(config.step_size_pca), self.step_min), self.step_max)
        if self.sampler_mode == "mh":
            # Reuse s_pca slot for diagnostics compatibility in mh mode.
            self.s_pca = mh_step
        self.s_ws  = min(max(_clean_step(config.step_size_ws), self.step_min), self.step_max)
        self.s_cg  = min(max(_clean_step(config.step_size_cg), self.step_min), self.step_max)

        # CG proposal uses blended covariance; keep settings bounded.
        cg_cov_blend = float(getattr(config, "cg_cov_blend", 0.20))
        if not math.isfinite(cg_cov_blend):
            cg_cov_blend = 0.20
        self.cg_cov_blend = min(max(cg_cov_blend, 0.0), 1.0)
        cg_cov_jitter = float(getattr(config, "cg_cov_jitter", 1e-6))
        if (not math.isfinite(cg_cov_jitter)) or (cg_cov_jitter <= 0.0):
            cg_cov_jitter = 1e-6
        self.cg_cov_jitter = cg_cov_jitter

        # Small positive epsilon used by numerical guards in several helpers.
        self.eps = 1e-4

        # Keep this for debugging transforms / Jacobian assumptions.
        # x grid is in [0,1], omega grid is known (self.omega)
        w_min = torch.min(self.omega).item()
        w_max = torch.max(self.omega).item()
        # jacobian = lambda xx: (w_max - w_min) * torch.ones_like(xx)

        # Core operator bundle used by every proposal type.
        self.operators = Operators(
            self.phi_inv, self.beta, self.tau, self.x, self.D_w, self.sigma, self.G_tau,
            jacobian=self.jacobian,
            x_transform=lambda y: y_to_x(y),
            )

        # Move objects are lightweight wrappers around proposal kernels.
        self.pca_move = PCAMove(self.operators, self.alphas)
        self.mh_move = RandomWalkMHMove(self.operators, self.alphas)
        self.ws_move = WeightSharingMove(self.operators, self.alphas, ws_k=self.ws_k)
        self.cg_move = CGMALAMove(self.operators, self.alphas)

    # ------------------------------------------------------------
    # Thin wrappers for proposal kernels.
    #
    # Why keep these wrappers:
    # - central place to pass current shared state
    # - easier to replace proposal implementations
    # - keeps the main loop compact and readable
    # ------------------------------------------------------------


    def _call_pca_move(self) -> MoveResult:
        """Propose a PCA-preconditioned move for all replicas."""
        return self.pca_move.suggest(
            self.x_current,
            self.n_current,
            self.H_current,
            self.C_pca,
            self.s_pca,
        )

    def _call_mh_move(self) -> MoveResult:
        """Propose a simple isotropic random-walk MH move for all replicas."""
        return self.mh_move.suggest(
            self.x_current,
            self.n_current,
            self.H_current,
            self.s_pca,
        )

    def _call_ws_move(self) -> MoveResult:
        """Propose a weight-sharing move for all replicas."""
        return self.ws_move.suggest(
            self.x_current,
            self.n_current,
            self.H_current,
            self.C_ws,
            self.s_ws,
        )

    def _call_cg_move(self, K_current: torch.Tensor) -> MoveResult:
        """Propose a CG/MALA move for all replicas."""
        return self.cg_move.suggest(
            self.x_current,
            self.n_current,
            self.H_current,
            K_current,
            self.G_tau,
            self.sigma,
            self.C_cg,
            self.s_cg,
            self.cg_cov_blend,
            self.cg_cov_jitter,
        )

    # compute G_tau from G_iw
    def _compute_G_tau(self) -> torch.Tensor:
        """Compute G(tau) from Matsubara-domain samples G(iw).

        We use the standard finite-temperature inverse Fourier transform:
            G(tau) = (1/beta) * sum_n exp(-i * wn * tau) * G(iwn)
        The real part is taken because downstream operators expect real values.
        """

        exp_term = torch.exp(- 1j * self.wn.unsqueeze(1) * self.tau.unsqueeze(0))  # (n_freq, n_tau)
        # Use real(G_iw) or absolute as needed
        G_tau = (1.0/self.beta) * torch.sum(exp_term * self.G_iw.unsqueeze(1), dim=0).real
        return G_tau

    # swap helper

    @torch._dynamo.disable
    def swap_samples(
        self,
        t: int,
        move_idx: Optional[torch.Tensor] = None,
        move_accepted: Optional[torch.Tensor] = None,
    ):
        """Attempt adjacent replica swaps and return swap diagnostics.

        Parameters
        ----------
        t:
            Current step index; controls odd/even pairing parity.
        move_idx:
            Optional per-replica move id used for swap-credit diagnostics.
        move_accepted:
            Optional per-replica accept mask at this step.

        Returns
        -------
        n_try, n_acc, swap_credit:
            Number of attempted swaps, accepted swaps, and per-move credit.
        """
        R = self.alphas.shape[0]
        swap_credit = (0.0, 0.0, 0.0)
        if R < 2:
            return 0, 0, swap_credit

        # Use disjoint adjacent pairs and alternate parity to avoid overlap.
        start = t % 2
        pair_i = torch.arange(start, R - 1, 2, device=self.device)
        if pair_i.numel() == 0:
            return 0, 0, swap_credit


        # Adjacent swap log-ratio: log r = (alpha_{i+1}-alpha_i) * (S_i - S_{i+1})
        dA = self.alphas[pair_i + 1] - self.alphas[pair_i]
        dS = self.S_current[pair_i] - self.S_current[pair_i + 1]
        log_r = dA * dS
        swap_prob = torch.exp(torch.clamp(log_r, max=0.0))
        do_swap = torch.rand_like(swap_prob) < swap_prob
        n_try = int(pair_i.numel())
        n_acc = int(do_swap.sum().item())
        if do_swap.any() and (move_idx is not None):
            mi = pair_i[do_swap]
            mj = mi + 1
            mids = torch.cat([move_idx[mi], move_idx[mj]], dim=0).to(torch.int64)
            if move_accepted is not None:
                accepted_swapped = torch.cat([move_accepted[mi], move_accepted[mj]], dim=0)
                mids = mids[accepted_swapped]
            if mids.numel() > 0:
                credit_vec = 0.5 * torch.bincount(mids, minlength=3).to(torch.float64)
                swap_credit = (
                    float(credit_vec[0].item()),
                    float(credit_vec[1].item()),
                    float(credit_vec[2].item()),
                )
        if not do_swap.any():
            return n_try, n_acc, swap_credit

        idx = torch.arange(R, device=self.device)
        i = pair_i[do_swap]
        j = i + 1
        new_idx = idx.clone()
        new_idx[i] = idx[j]
        new_idx[j] = idx[i]

        # Swap current states; history arrays are written once per step in the main loop.
        self.x_current = self.x_current[new_idx]
        self.n_current = self.n_current[new_idx]
        self.H_current = self.H_current[new_idx]
        self.S_current = self.S_current[new_idx]
        return n_try, n_acc, swap_credit

    def online_stats_update(self, mu_old: torch.Tensor,
                            C_old: torch.Tensor,
                            count_old: int,
                            x_accept: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Update running mean/covariance from accepted samples.

        Parameters
        ----------
        mu_old : Tensor[D]
            Previous running mean.
        C_old : Tensor[D, D]
            Previous covariance estimate.
        count_old : int
            Number of samples accumulated in (mu_old, C_old).
        x_new : Tensor[K, D]
            New accepted samples.

        Returns
        -------
        mu_new : Tensor[D]
            Updated running mean.
        C_new : Tensor[D, D]
            Updated covariance estimate.
        count_new : int
            Updated number of samples.
        """
        # x_accept shape: (k, D). If there are no new accepted points,
        # the running statistics stay unchanged.
        k = x_accept.shape[0]
        if k == 0:
            return mu_old, C_old, count_old

        # 1) Update count and mean.
        count_new = count_old + k
        sum_x = x_accept.sum(dim=0)                       # (D,)
        mu_new = (count_old * mu_old + sum_x) / count_new

        # 2) Convert stored covariance back to scatter matrix.
        scatter_old = count_old * C_old                   # (D, D)


        # 3) Scatter of new block around old mean.
        # This is paired with a shift correction below.
        diff_old = x_accept - mu_old.unsqueeze(0)         # (k, D)
        scatter_new = diff_old.T @ diff_old               # (D, D)


        # 4) Mean-shift correction from old to new mean.
        delta_mu = (mu_old - mu_new).unsqueeze(1)         # (D, 1)
        shift_mat = count_old * (delta_mu @ delta_mu.T)   # (D, D)


        # 5) Combine and re-normalize to covariance.
        C_new = (scatter_old + scatter_new + shift_mat) / count_new
        C_new = 0.5 * (C_new + C_new.T)

        return mu_new, C_new, count_new

    @torch._dynamo.disable
    def _q(self, x):
        """Helper for q10/q50/q90 diagnostics on a tensor."""
        x = x.detach().cpu().double()
        if x.numel() == 0:
            return None

        q = torch.tensor([0.1, 0.5, 0.9], device=x.device, dtype=x.dtype)

        return torch.quantile(x, q).detach().cpu().tolist()

    def _print_diagnostics(self, name, dH, acc_mask, alphas):
        """Print per-move diagnostics for accepted/rejected proposals."""
        # dH: signed, shape (R,)
        # acc_mask: shape (R,)
        # alphas: shape (R,)
        q_acc = self._q(dH[acc_mask])
        q_rej = self._q(dH[~acc_mask])
        q_aacc = self._q((alphas * dH)[acc_mask])
        q_arej = self._q((alphas * dH)[~acc_mask])

        print(f"[{name}] dH(signed) q10/q50/q90 acc =", q_acc, "rej =", q_rej)
        print(f"[{name}] alpha*dH   q10/q50/q90 acc =", q_aacc, "rej =", q_arej)

        # alpha quantile diagnostics
        a = alphas
        a_lo = torch.quantile(a, 0.2)
        a_hi = torch.quantile(a, 0.8)
        m_lo = a <= a_lo
        m_hi = a >= a_hi
        print(f"[{name}] low-alpha acc dH q =", self._q(dH[acc_mask & m_lo]))
        print(f"[{name}] high-alpha acc dH q =", self._q(dH[acc_mask & m_hi]))

    def compute_lambdas_jerrell(
        self,
        A_hat: torch.Tensor,
        chunk_size: int = 256,
        eps: float = 1e-300,
    ) -> torch.Tensor:
        """Compute Bryan/Jerrell Hessian eigenvalues for alpha weighting."""
        return compute_lambdas_jerrell_impl(
            operators=self.operators,
            x_grid=self.x,
            device=self.device,
            A_hat=A_hat,
            chunk_size=chunk_size,
            eps=eps,
        )

    def run_replica_exchange(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run replica-exchange MCMC and return posterior summaries.

        Returns
        -------
        n_means : torch.Tensor
            Replica-wise posterior mean of n(x) after adaptation.
        H_means : torch.Tensor
            Replica-wise posterior mean of H.
        mcse_d : torch.Tensor
            Per-bin MCSE estimate.
        diagnostics : dict
            Sampling diagnostics and adaptation traces.
        """
        # ------------------------------------------------------------------
        # Phase 0: allocate and initialize chain state.
        # ------------------------------------------------------------------
        # R: number of replicas (one per alpha).
        # L: number of MCMC steps.
        # D: discretized x-grid dimension for each spectrum.
        R = self.alphas.shape[0]
        L = self.n_step
        D = self.omega.shape[0]

        # We store unconstrained parameter y internally for proposal kernels.
        # Physical x in [0, 1] is reconstructed lazily by y_to_x.
        y0 = x_to_y(self.x)                    # (D,)
        self.x_current = y0.unsqueeze(0).repeat(R, 1)


        # Start from a simple feasible state:
        # - n initialized to positive constant then normalized
        # - H/S immediately evaluated for consistency
        self.n_current = torch.ones((R, D), device=self.device, dtype=torch.float64)
        self.n_current = self.operators.normalize(self.n_current)
        self.H_current = self.operators._H(
            self.operators._h(self.n_current, self.operators._K_tau_x(self.x_current))
        )
        self.S_current = self.operators.entropy(self.n_current)


        # Proposal buffers hold candidate states for the current step only.
        self.proposal_x = torch.zeros_like(self.x_current)
        self.proposal_n = torch.zeros_like(self.n_current)
        self.proposal_H = torch.zeros_like(self.H_current)


        # Full trajectory storage (step 0 ... L).
        # Note: this is memory-heavy but simplifies post-processing and diagnostics.
        self.x_samples = torch.zeros((R, L+1, D), device=self.device, dtype=torch.float64)
        self.n_samples = torch.zeros((R, L+1, D), device=self.device, dtype=torch.float64)
        self.H_samples = torch.zeros((R, L+1), device=self.device, dtype=torch.float64)
        self.S_samples = torch.zeros((R, L+1), device=self.device, dtype=torch.float64)


        self.x_samples[:, 0, :] = y_to_x(self.x_current)
        self.n_samples[:, 0, :] = self.n_current
        self.H_samples[:, 0]    = self.H_current
        self.S_samples[:, 0]    = self.S_current


        # Per-move trial/accept counters for final acceptance-rate diagnostics.
        trials = {'pca': 0, 'ws': 0, 'cg': 0}
        accepts = {'pca': 0, 'ws': 0, 'cg': 0}
        # Swap counters for replica exchange diagnostics.
        swap_trials = 0
        swap_accepts = 0
        swap_calls = 0

        # --- Move diagnostics accumulator (keeps loop readable) ---
        move_diag = MoveDiagAccumulator(max_store=5000)

        self.try_count_pca = 0
        self.try_count_ws = 0
        self.try_count_cg = 0
        cov_buf_pca: list[torch.Tensor] = []
        cov_buf_ws: list[torch.Tensor] = []
        cov_buf_cg: list[torch.Tensor] = []
        wx_trapz = _trapz_weights(self.operators.x).to(device=self.device, dtype=torch.float64)
        wx_trapz = torch.clamp(wx_trapz, min=1e-30)

        def _n_to_u(n_batch: torch.Tensor) -> torch.Tensor:
            p = torch.clamp(n_batch.to(torch.float64) * wx_trapz.unsqueeze(0), min=1e-300)
            p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-300)
            return torch.log(p)

        # ------------------------------------------------------------------
        # Phase 1: parse and sanitize adaptation hyperparameters.
        # ------------------------------------------------------------------
        # We keep all knobs finite and in valid ranges to avoid destabilizing
        # adaptation when configs are missing or malformed.
        accept_ema_kappa = float(getattr(self.config, "accept_ema_kappa", 0.80))
        if (not math.isfinite(accept_ema_kappa)) or (accept_ema_kappa <= 0.0):
            accept_ema_kappa = 0.80
        accept_ema_kappa = min(max(accept_ema_kappa, 0.50), 1.0)
        prob_adapt_kappa = float(getattr(self.config, "prob_adapt_kappa", 0.90))
        if (not math.isfinite(prob_adapt_kappa)) or (prob_adapt_kappa <= 0.0):
            prob_adapt_kappa = 0.90
        prob_adapt_kappa = min(max(prob_adapt_kappa, 0.50), 1.0)
        prob_metric = str(getattr(self.config, "prob_adapt_metric", "esjd_per_sec")).lower()
        if prob_metric not in ("esjd_per_sec", "accept_rate"):
            prob_metric = "esjd_per_sec"
        prob_min = float(getattr(self.config, "prob_min", 0.10))
        if not math.isfinite(prob_min):
            prob_min = 0.10
        prob_min = min(max(prob_min, 0.0), 0.32)
        prob_utility_floor = float(getattr(self.config, "prob_utility_floor", 1e-12))
        if (not math.isfinite(prob_utility_floor)) or (prob_utility_floor <= 0.0):
            prob_utility_floor = 1e-12
        prob_min_trials_per_move = int(getattr(self.config, "prob_min_trials_per_move", 100))
        if prob_min_trials_per_move < 1:
            prob_min_trials_per_move = 1
        prob_esjd_whitened = bool(getattr(self.config, "prob_esjd_whitened", True))
        prob_whiten_jitter = float(getattr(self.config, "prob_whiten_jitter", 1e-6))
        if (not math.isfinite(prob_whiten_jitter)) or (prob_whiten_jitter <= 0.0):
            prob_whiten_jitter = 1e-6
        prob_whiten_shrinkage = float(getattr(self.config, "prob_whiten_shrinkage", 0.05))
        if not math.isfinite(prob_whiten_shrinkage):
            prob_whiten_shrinkage = 0.05
        prob_whiten_shrinkage = min(max(prob_whiten_shrinkage, 0.0), 1.0)
        prob_whiten_weight_mode = str(getattr(self.config, "prob_whiten_weight_mode", "count")).lower()
        if prob_whiten_weight_mode not in ("equal", "count", "trial"):
            prob_whiten_weight_mode = "count"
        prob_swap_reward_weight = float(getattr(self.config, "prob_swap_reward_weight", 0.15))
        if not math.isfinite(prob_swap_reward_weight):
            prob_swap_reward_weight = 0.15
        prob_swap_reward_weight = min(max(prob_swap_reward_weight, 0.0), 1.0)
        prob_swap_warmup_updates = int(getattr(self.config, "prob_swap_warmup_updates", 10))
        if prob_swap_warmup_updates < 0:
            prob_swap_warmup_updates = 0
        adapt_hot_interval = max(1, int(getattr(self.config, "adapt_hot_interval", 1)))
        adapt_late_interval = max(1, int(getattr(self.config, "adapt_late_interval", 100)))
        scale_adapt_t0 = float(getattr(self.config, "scale_adapt_t0", 10.0))
        if (not math.isfinite(scale_adapt_t0)) or (scale_adapt_t0 < 0.0):
            scale_adapt_t0 = 10.0
        scale_adapt_gamma = float(getattr(self.config, "scale_adapt_gamma", 0.05))
        if (not math.isfinite(scale_adapt_gamma)) or (scale_adapt_gamma <= 0.0):
            scale_adapt_gamma = 0.05
        scale_adapt_kappa = float(getattr(self.config, "scale_adapt_kappa", 0.75))
        if (not math.isfinite(scale_adapt_kappa)) or (scale_adapt_kappa <= 0.5) or (scale_adapt_kappa > 1.0):
            scale_adapt_kappa = 0.75
        alpha_star_pca = target_accept_value(
            self.config, "target_accept_pca", default=0.50, device=self.device
        )
        alpha_star_ws = target_accept_value(
            self.config, "target_accept_ws", default=0.50, device=self.device
        )
        alpha_star_cg = target_accept_value(
            self.config, "target_accept_cg", default=0.50, device=self.device
        )
        self.alpha_est_pca = alpha_star_pca.clone()
        self.alpha_est_ws = alpha_star_ws.clone()
        self.alpha_est_cg = alpha_star_cg.clone()
        step_adapter_pca = DualAveragingStepAdapter(
            step=self.s_pca,
            target_accept=float(alpha_star_pca.detach().cpu().item()),
            step_min=self.step_min,
            step_max=self.step_max,
            gamma=scale_adapt_gamma,
            t0=scale_adapt_t0,
            kappa=scale_adapt_kappa,
        )
        step_adapter_ws = DualAveragingStepAdapter(
            step=self.s_ws,
            target_accept=float(alpha_star_ws.detach().cpu().item()),
            step_min=self.step_min,
            step_max=self.step_max,
            gamma=scale_adapt_gamma,
            t0=scale_adapt_t0,
            kappa=scale_adapt_kappa,
        )
        step_adapter_cg = DualAveragingStepAdapter(
            step=self.s_cg,
            target_accept=float(alpha_star_cg.detach().cpu().item()),
            step_min=self.step_min,
            step_max=self.step_max,
            gamma=scale_adapt_gamma,
            t0=scale_adapt_t0,
            kappa=scale_adapt_kappa,
        )
        step_adapter_frozen = False


        # Burn-in/adaptation schedule:
        # - adaptation is constrained inside burn-in
        # - hot phase: [0, hot_start]
        # - late phase: (hot_start, burnin]
        burnin = int(self.burnin_ratio * L)
        burnin = min(max(burnin, 0), max(0, L - 1))

        hot_start_ratio = float(getattr(self.config, "hot_start_ratio", 0.5))
        adapt_end_ratio = float(getattr(self.config, "adapt_end_ratio", 0.8))
        if not math.isfinite(hot_start_ratio):
            hot_start_ratio = 0.5
        if not math.isfinite(adapt_end_ratio):
            adapt_end_ratio = 0.8
        hot_start_ratio = min(max(hot_start_ratio, 0.0), 1.0)

        adapt_end = burnin
        hot_start = int(hot_start_ratio * adapt_end) if adapt_end > 0 else 0
        hot_start = min(max(hot_start, 0), adapt_end)
        # Report effective ratios over total steps for diagnostics compatibility.
        hot_start_ratio = float(hot_start) / float(max(1, L))
        adapt_end_ratio = float(adapt_end) / float(max(1, L))
        self.adapt_end = adapt_end

        # State object used by probability adaptation logic
        # (ESJD/accept-rate tracking, swap credit, history traces).
        prob_state = ProbAdaptState.initialize(
            p_pca=self.p_pca,
            p_ws=self.p_ws,
            p_cg=self.p_cg,
            utility_floor=prob_utility_floor,
        )

        mcse_threshold = float(getattr(self.config, "mcse_threshold", 0.005))
        if (not math.isfinite(mcse_threshold)) or (mcse_threshold <= 0.0):
            mcse_threshold = 0.005

        check_interval = int(getattr(self.config, "mcse_check_interval", 100))
        if check_interval < 1:
            check_interval = 0  # disable periodic MCSE/IAT check

        mcse_iat_window = int(getattr(self.config, "mcse_iat_window", 2000))
        if mcse_iat_window < 8:
            mcse_iat_window = 8

        iat_final_window = int(getattr(self.config, "iat_final_window", 0))
        iat_max_fft_bytes = int(getattr(self.config, "iat_max_fft_bytes", 134217728))
        if iat_max_fft_bytes < (8 * 1024 * 1024):
            iat_max_fft_bytes = 8 * 1024 * 1024

        periodic_iat_error = ""
        final_iat_error = ""
        achieved_time = None
        achieved_step = None

        # Wall-clock timer for throughput and stopping diagnostics.
        start = time.perf_counter()

        # Progress bar update policy is intentionally coarse to reduce overhead.
        pbar = tqdm(total=L, mininterval=2.0, miniters=100)
        for t in range(L):
            pbar.update(1)
            # if t % 500 == 0:
            #     print(f"step {t}/{L}")

            # NOTE: no clamp/sort here. Valid x range is enforced by transform y_to_x.

            K_current = self.operators._K_tau_x(self.x_current) # (R, T, D)
            u = torch.rand(R, device=self.device, dtype=torch.float64)


            # --------------------------------------------------
            # Proposal stage.
            # - mixed: evaluate all move families and sample one per replica.
            # - mh: use only a simple isotropic random-walk MH proposal.
            # --------------------------------------------------
            t_move = time.perf_counter()
            if self.sampler_mode == "mh":
                res_pca = self._call_mh_move()
            else:
                res_pca = self._call_pca_move()
            dt_pca = time.perf_counter() - t_move
            log_lr_pca = res_pca.log_lr
            prop_n_pca = res_pca.proposal_n
            prop_x_pca = res_pca.proposal_x
            prop_H_pca = res_pca.proposal_H

            if self.sampler_mode == "mh":
                cond_pca = torch.ones((R,), device=self.device, dtype=torch.bool)
                cond_ws = torch.zeros((R,), device=self.device, dtype=torch.bool)
                cond_cg = torch.zeros((R,), device=self.device, dtype=torch.bool)
                move_idx = torch.zeros((R,), device=self.device, dtype=torch.int64)

                prop_n_ws = self.n_current
                prop_x_ws = self.x_current
                prop_H_ws = self.H_current
                prop_n_cg = self.n_current
                prop_x_cg = self.x_current
                prop_H_cg = self.H_current

                self.try_count_pca += R
                log_lr = log_lr_pca
                self.proposal_x = prop_x_pca
                self.proposal_n = prop_n_pca
                self.proposal_H = prop_H_pca
            else:
                t_move = time.perf_counter()
                res_ws = self._call_ws_move()
                dt_ws = time.perf_counter() - t_move
                log_lr_ws = res_ws.log_lr
                prop_n_ws = res_ws.proposal_n
                prop_x_ws = res_ws.proposal_x
                prop_H_ws = res_ws.proposal_H

                t_move = time.perf_counter()
                res_cg = self._call_cg_move(K_current)
                dt_cg = time.perf_counter() - t_move
                log_lr_cg = res_cg.log_lr
                prop_n_cg = res_cg.proposal_n
                prop_x_cg = res_cg.proposal_x
                prop_H_cg = res_cg.proposal_H

                if (t > hot_start) and (t <= adapt_end):
                    prob_state.accumulate_timing(dt_pca=dt_pca, dt_ws=dt_ws, dt_cg=dt_cg)

                cond_pca, cond_ws, cond_cg, move_idx = select_move_masks(
                    u,
                    self.p_pca,
                    self.p_ws,
                    device=self.device,
                )
                assert abs(self.p_pca + self.p_ws + self.p_cg - 1.0) < 1e-6

                self.try_count_pca += cond_pca.sum().item()
                self.try_count_ws += cond_ws.sum().item()
                self.try_count_cg += cond_cg.sum().item()

                # Build selected log-ratio/candidate tensors by move mask.
                log_lr = torch.where(
                    u < self.p_pca,
                    log_lr_pca,
                    torch.where(cond_ws, log_lr_ws, log_lr_cg),
                )
                self.proposal_x = torch.where(
                    cond_pca[:, None],
                    prop_x_pca,
                    torch.where(cond_ws[:, None], prop_x_ws, prop_x_cg),
                )
                self.proposal_n = torch.where(
                    cond_pca[:, None],
                    prop_n_pca,
                    torch.where(cond_ws[:, None], prop_n_ws, prop_n_cg),
                )
                self.proposal_H = torch.where(
                    cond_pca,
                    prop_H_pca,
                    torch.where(cond_ws, prop_H_ws, prop_H_cg),
                )

            # Hard reject numerically invalid proposals by forcing log ratio to -inf.
            # This keeps state untouched and avoids poisoning adaptive statistics.
            invalid_prop = (
                (~torch.isfinite(self.proposal_n)).any(dim=1)
                | (self.proposal_n <= 0).any(dim=1)
                | (~torch.isfinite(self.proposal_x)).any(dim=1)
                | (~torch.isfinite(self.proposal_H))
                | (~torch.isfinite(log_lr))
            )
            if invalid_prop.any():
                log_lr = torch.where(
                    invalid_prop,
                    torch.full_like(log_lr, float("-inf")),
                    log_lr,
                )


            # Per-move energy delta diagnostics (proposal side).
            dH_pca = prop_H_pca - self.H_current
            dH_ws  = prop_H_ws  - self.H_current
            dH_cg  = prop_H_cg  - self.H_current

            # accumulate proposal-side stats (only for replicas where that move was selected)
            # NOTE: accumulator takes full (R,) tensors + mask, and internally slices/aggregates.
            # accumulate proposal-side stats (only for replicas where that move was selected)
            if cond_pca.any():
                move_diag.update_proposal("pca", dH_pca[cond_pca], prop_n_pca[cond_pca], self.n_current[cond_pca])
            if cond_ws.any():
                move_diag.update_proposal("ws",  dH_ws[cond_ws],  prop_n_ws[cond_ws],  self.n_current[cond_ws])
            if cond_cg.any():
                move_diag.update_proposal("cg",  dH_cg[cond_cg],  prop_n_cg[cond_cg],  self.n_current[cond_cg])


            # Metropolis-Hastings accept test in log-space for numerical stability.
            xi = torch.rand(R, device=self.device, dtype=torch.float64)
            accept_mask = torch.log(xi) < log_lr  # shape(R,)

            # ---- diagnostics: accepted-side stats (move-specific) ----
            acc_pca = cond_pca & accept_mask
            acc_ws  = cond_ws  & accept_mask
            acc_cg  = cond_cg  & accept_mask

            if acc_pca.any():
                move_diag.update_accepted("pca", dH_pca[acc_pca], prop_n_pca[acc_pca], self.n_current[acc_pca])
            if acc_ws.any():
                move_diag.update_accepted("ws",  dH_ws[acc_ws],  prop_n_ws[acc_ws],  self.n_current[acc_ws])
            if acc_cg.any():
                move_diag.update_accepted("cg",  dH_cg[acc_cg],  prop_n_cg[acc_cg],  self.n_current[acc_cg])

            # During adaptation windows, collect accepted u=log(w_x*n) samples
            # for per-move covariance updates used by proposal preconditioning.
            if (self.sampler_mode == "mixed") and (adapt_end > 0) and (t <= adapt_end):
                if acc_pca.any():
                    cov_buf_pca.append(_n_to_u(prop_n_pca[acc_pca].detach()).to(self.mu_pca.dtype))
                if acc_ws.any():
                    cov_buf_ws.append(_n_to_u(prop_n_ws[acc_ws].detach()).to(self.mu_ws.dtype))
                if acc_cg.any():
                    cov_buf_cg.append(_n_to_u(prop_n_cg[acc_cg].detach()).to(self.mu_cg.dtype))
            if (self.sampler_mode == "mixed") and (adapt_end > 0) and (t > hot_start) and (t <= adapt_end):
                prob_state.accumulate_moves(
                    cond_pca=cond_pca,
                    cond_ws=cond_ws,
                    cond_cg=cond_cg,
                    acc_pca=acc_pca,
                    acc_ws=acc_ws,
                    acc_cg=acc_cg,
                    prop_n_pca=prop_n_pca,
                    prop_n_ws=prop_n_ws,
                    prop_n_cg=prop_n_cg,
                    n_current=self.n_current,
                    w_x=wx_trapz,
                )

            # Verbose final-step debug dump for sanity-checking tail behavior.
            if t == L-1:
                primary_name = "mh" if self.sampler_mode == "mh" else "pca"
                self._print_diagnostics(primary_name, dH_pca, acc_pca, self.alphas)
                if self.sampler_mode == "mixed":
                    self._print_diagnostics("ws",  dH_ws,  acc_ws,  self.alphas)
                    self._print_diagnostics("cg",  dH_cg,  acc_cg,  self.alphas)
                print("alpha stats:", float(self.alphas.min()), float(self.alphas.median()), float(self.alphas.max()))
                print("dH stats:", float(dH_pca.min()), float(dH_pca.median()), float(dH_pca.max()))
                print("alpha*dH stats:", float((self.alphas*dH_pca).min()), float((self.alphas*dH_pca).median()), float((self.alphas*dH_pca).max()))

            self.x_current = torch.where(accept_mask.unsqueeze(1), self.proposal_x, self.x_current)
            self.n_current = torch.where(accept_mask.unsqueeze(1), self.proposal_n, self.n_current)
            self.H_current = torch.where(accept_mask, self.proposal_H, self.H_current)

            # Entropy is required by replica swap acceptance and alpha posterior weighting.
            self.S_current = self.operators.entropy(self.n_current)


            # Global counters used for final diagnostics and adaptation summaries.
            trials['pca'] += cond_pca.sum().item()
            trials['ws']  += cond_ws.sum().item()
            trials['cg']  += cond_cg.sum().item()
            accepts['pca'] += (cond_pca & accept_mask).sum().item()
            accepts['ws']  += (cond_ws  & accept_mask).sum().item()
            accepts['cg']  += (cond_cg  & accept_mask).sum().item()
            # Short-horizon counters consumed by Robbins-Monro / dual averaging.
            self.acc_count_pca += (cond_pca & accept_mask).sum().item()
            self.acc_count_ws  += (cond_ws  & accept_mask).sum().item()
            self.acc_count_cg  += (cond_cg  & accept_mask).sum().item()

            adapt_active = adapt_end > 0
            in_hot_phase = adapt_active and (t <= hot_start)
            in_late_phase = adapt_active and (t > hot_start) and (t <= adapt_end)
            do_update = False
            do_scale_update = False
            do_prob_update = False
            # Keep MH baseline fixed-step / fixed-probability for fair comparisons.
            if self.sampler_mode == "mixed":
                if in_hot_phase:
                    do_update = (t % adapt_hot_interval == 0)
                    do_scale_update = do_update
                elif in_late_phase:
                    do_update = ((t - hot_start) % adapt_late_interval == 0)
                    do_scale_update = do_update
                    do_prob_update = do_update

            # Adaptation step:
            # - step-size adaptation (all enabled phases)
            # - probability adaptation (late phase only)
            # - covariance flush + clamping for numerical safety
            if do_update:
                j = max(1, t + 1)
                gamma_accept = j ** (-accept_ema_kappa)
                gamma_prob = j ** (-prob_adapt_kappa)
                if do_scale_update:
                    if self.try_count_pca > 0:
                        alpha_pca = self.acc_count_pca / self.try_count_pca
                        self.alpha_est_pca = self.alpha_est_pca + gamma_accept * (alpha_pca - self.alpha_est_pca)
                        self.s_pca = step_adapter_pca.update(float(alpha_pca))
                    self.try_count_pca = 0
                    self.acc_count_pca = 0
                    if self.try_count_ws > 0:
                        alpha_ws = self.acc_count_ws / self.try_count_ws
                        self.alpha_est_ws = self.alpha_est_ws + gamma_accept * (alpha_ws - self.alpha_est_ws)
                        self.s_ws = step_adapter_ws.update(float(alpha_ws))
                    self.try_count_ws = 0
                    self.acc_count_ws = 0
                    if self.try_count_cg > 0:
                        alpha_cg = self.acc_count_cg / self.try_count_cg
                        self.alpha_est_cg = self.alpha_est_cg + gamma_accept * (alpha_cg - self.alpha_est_cg)
                        self.s_cg = step_adapter_cg.update(float(alpha_cg))
                    self.try_count_cg = 0
                    self.acc_count_cg = 0
                if do_prob_update:
                    self.p_pca, self.p_ws, self.p_cg = adapt_move_probabilities(
                        state=prob_state,
                        p_pca=self.p_pca,
                        p_ws=self.p_ws,
                        p_cg=self.p_cg,
                        t=t,
                        gamma_prob=gamma_prob,
                        prob_metric=prob_metric,
                        prob_min_trials_per_move=prob_min_trials_per_move,
                        prob_esjd_whitened=prob_esjd_whitened,
                        prob_whiten_jitter=prob_whiten_jitter,
                        prob_whiten_shrinkage=prob_whiten_shrinkage,
                        prob_whiten_weight_mode=prob_whiten_weight_mode,
                        prob_swap_reward_weight=prob_swap_reward_weight,
                        prob_swap_warmup_updates=prob_swap_warmup_updates,
                        prob_utility_floor=prob_utility_floor,
                        C_pca=self.C_pca,
                        C_ws=self.C_ws,
                        C_cg=self.C_cg,
                        count_pca=self.count_pca,
                        count_ws=self.count_ws,
                        count_cg=self.count_cg,
                        device=self.device,
                    )

                self.s_pca, self.s_ws, self.s_cg = clamp_step_sizes(
                    self.s_pca,
                    self.s_ws,
                    self.s_cg,
                    step_min=self.step_min,
                    step_max=self.step_max,
                )

                if self.sampler_mode == "mh":
                    self.p_pca, self.p_ws, self.p_cg = 1.0, 0.0, 0.0
                else:
                    self.p_pca, self.p_ws, self.p_cg = clamp_and_normalize_probs(
                        self.p_pca,
                        self.p_ws,
                        self.p_cg,
                        prob_min=prob_min,
                    )

                (
                    self.mu_pca,
                    self.C_pca,
                    self.count_pca,
                    self.mu_ws,
                    self.C_ws,
                    self.count_ws,
                    self.mu_cg,
                    self.C_cg,
                    self.count_cg,
                ) = flush_cov_buffers(
                    cov_buf_pca=cov_buf_pca,
                    cov_buf_ws=cov_buf_ws,
                    cov_buf_cg=cov_buf_cg,
                    mu_pca=self.mu_pca,
                    C_pca=self.C_pca,
                    count_pca=self.count_pca,
                    mu_ws=self.mu_ws,
                    C_ws=self.C_ws,
                    count_ws=self.count_ws,
                    mu_cg=self.mu_cg,
                    C_cg=self.C_cg,
                    count_cg=self.count_cg,
                    online_stats_update=self.online_stats_update,
                )

            # Freeze dual-averaging step-size adapters once adaptation window closes.
            if (not step_adapter_frozen) and (t >= adapt_end):
                (
                    self.mu_pca,
                    self.C_pca,
                    self.count_pca,
                    self.mu_ws,
                    self.C_ws,
                    self.count_ws,
                    self.mu_cg,
                    self.C_cg,
                    self.count_cg,
                ) = flush_cov_buffers(
                    cov_buf_pca=cov_buf_pca,
                    cov_buf_ws=cov_buf_ws,
                    cov_buf_cg=cov_buf_cg,
                    mu_pca=self.mu_pca,
                    C_pca=self.C_pca,
                    count_pca=self.count_pca,
                    mu_ws=self.mu_ws,
                    C_ws=self.C_ws,
                    count_ws=self.count_ws,
                    mu_cg=self.mu_cg,
                    C_cg=self.C_cg,
                    count_cg=self.count_cg,
                    online_stats_update=self.online_stats_update,
                )
                self.s_pca = step_adapter_pca.freeze()
                self.s_ws = step_adapter_ws.freeze()
                self.s_cg = step_adapter_cg.freeze()
                step_adapter_frozen = True

            # Periodic replica-exchange step.
            if self.use_replica_exchange and (t % self.swap_freq == 0):
                n_swap_try, n_swap_acc, swap_credit = self.swap_samples(
                    t,
                    move_idx=move_idx,
                    move_accepted=accept_mask,
                )
                swap_trials += int(n_swap_try)
                swap_accepts += int(n_swap_acc)
                swap_calls += 1
                if (t > hot_start) and (t <= adapt_end):
                    prob_state.accumulate_swap_credit(swap_credit)

            # Optional periodic convergence check using a trailing window.
            # This avoids unbounded FFT memory use for very long runs.
            if check_interval > 0 and t > adapt_end and (t % check_interval == 0) and achieved_time is None:
                # Use a trailing window to keep FFT memory bounded on low-memory runtimes.
                iat_start = max(adapt_end, t - mcse_iat_window)
                chain_chunk = self.n_samples[:, iat_start:t, :]
                chunk_len = int(chain_chunk.shape[1])
                if chunk_len >= 8:
                    try:
                        samples = chain_chunk.reshape(-1, D)  # (R*chunk_len, D)
                        var_d = samples.var(dim=0, unbiased=True)  # (D,)
                        tau_ints_chk = AutocorrTime.integrated_autocorr_time(
                            chain_chunk,
                            max_fft_bytes=iat_max_fft_bytes,
                        )
                        tau_ints_chk = torch.clamp(tau_ints_chk.to(torch.float64), min=1.0)
                        ess_per_chain = float(chunk_len) / tau_ints_chk
                        total_ess = max(float(ess_per_chain.sum().item()), 1e-12)
                        mcse_d = torch.sqrt(var_d / total_ess)  # (D,)
                        if mcse_d.max().item() < mcse_threshold:
                            achieved_step = t
                            achieved_time = time.perf_counter() - start
                    except Exception as e:
                        # Keep sampling alive if periodic diagnostics fail.
                        periodic_iat_error = f"{type(e).__name__}: {e}"
                        check_interval = 0

            # Persist current state at t+1 (t=0 slot was already initialized).
            self.x_samples[:, t+1, :] = y_to_x(self.x_current)
            self.n_samples[:, t+1, :] = self.n_current
            self.H_samples[:, t+1]    = self.H_current
            self.S_samples[:, t+1]    = self.S_current

        # ------------------------------------------------------------------
        # Phase 2: summarize chains and compute diagnostics.
        # ------------------------------------------------------------------
        end = time.perf_counter()
        n_means = self.n_samples[:, adapt_end:, :].mean(dim=1)
        H_means = self.H_samples[:, adapt_end:].mean(dim=1)
        S_means = self.S_samples[:, adapt_end:].mean(dim=1)
        summary_moves = ["pca"] if self.sampler_mode == "mh" else ["pca", "ws", "cg"]
        summary_labels = {
            "pca": "mh" if self.sampler_mode == "mh" else "pca",
            "ws": "ws",
            "cg": "cg",
        }

        # Acceptance summary over the whole run.
        for move in summary_moves:
            rate = accepts[move] / trials[move] if trials[move]>0 else float('nan')
            print(f"{summary_labels[move]} acceptance rate = {rate:.3f} ({accepts[move]}/{trials[move]})")
        if self.use_replica_exchange:
            swap_rate = (swap_accepts / swap_trials) if swap_trials > 0 else float('nan')
            print(f"swap acceptance rate = {swap_rate:.3f} ({swap_accepts}/{swap_trials}), calls={swap_calls}")
        else:
            print("swap disabled (use_replica_exchange=false)")

        # Condensed per-move diagnostics accumulated online.
        move_diag_summary = move_diag.finalize(trials=trials, accepts=accepts)
        for move in summary_moves:
            md = move_diag_summary.get(move, {})
            print(
                 f"[{summary_labels[move]}] |dH| mean(prop)={md.get('mean_abs_dH_prop', float('nan')):.3e}, "
                 f"mean(acc)={md.get('mean_abs_dH_acc', float('nan')):.3e}, "
                 f"q10/q50/q90(acc)={md.get('q10_abs_dH_acc', float('nan')):.3e}/"
                 f"{md.get('q50_abs_dH_acc', float('nan')):.3e}/"
                 f"{md.get('q90_abs_dH_acc', float('nan')):.3e} ; "
                 f"dN L1 mean(prop)={md.get('mean_dn_l1_prop', float('nan')):.3e}, "
                 f"mean(acc)={md.get('mean_dn_l1_acc', float('nan')):.3e} ; "
                 f"dN L2 mean(prop)={md.get('mean_dn_l2_prop', float('nan')):.3e}, "
                 f"mean(acc)={md.get('mean_dn_l2_acc', float('nan')):.3e}"
                )
        # Select post-adaptation region for convergence diagnostics.
        post_chain = self.n_samples[:, adapt_end:, :]
        if iat_final_window > 0 and post_chain.shape[1] > iat_final_window:
            # Optional truncation to cap FFT cost/memory.
            post_chain = post_chain[:, -iat_final_window:, :]
        post_len = int(post_chain.shape[1])
        try:
            tau_ints = AutocorrTime.integrated_autocorr_time(
                post_chain,
                max_fft_bytes=iat_max_fft_bytes,
            )
        except Exception as e:
            final_iat_error = f"{type(e).__name__}: {e}"
            tau_ints = torch.ones((R,), device=self.device, dtype=torch.float64)
        tau_ints = torch.clamp(tau_ints.to(torch.float64), min=1.0)
        print("IAT per replica:", tau_ints)

        # Effective sample size per replica and run throughput.
        ess_per_chain = float(post_len) / tau_ints
        print("ESS per replica:", ess_per_chain)
        total_ess = ess_per_chain.sum().item()
        elapsed = end - start
        ess_per_sec = total_ess / elapsed

        print(f"Total ESS: {total_ess:.1f}")
        print(f"ESS/sec: {ess_per_sec:.3f}")
        # R-hat must compare chains that target the same distribution.
        # Each replica has a different alpha, so we compute split-Rhat within
        # each replica chain and then aggregate.
        rhat_split_segments = max(2, int(getattr(self.config, "rhat_split_segments", 4)))
        rhat_by_replica = GelmanRubin.split_rhat_per_replica(
            self.n_samples[:, adapt_end:, :],
            n_split=rhat_split_segments,
        ).to(device=self.device, dtype=torch.float64)  # (R, D)
        rhat_exclude_edge_bins_requested = max(
            0, int(getattr(self.config, "rhat_exclude_edge_bins", 1))
        )
        rhat_exclude_edge_bins_applied = 0
        rhat_eval_by_replica = rhat_by_replica
        D_vals = int(rhat_by_replica.shape[1])
        if (
            rhat_exclude_edge_bins_requested > 0
            and D_vals > 2 * rhat_exclude_edge_bins_requested
        ):
            lo = rhat_exclude_edge_bins_requested
            hi = D_vals - rhat_exclude_edge_bins_requested
            rhat_eval_by_replica = rhat_by_replica[:, lo:hi]
            rhat_exclude_edge_bins_applied = rhat_exclude_edge_bins_requested
        rhat_eval_bins = int(rhat_eval_by_replica.shape[1])

        rhat_flat = rhat_eval_by_replica.reshape(-1)
        finite = torch.isfinite(rhat_flat)
        if finite.any():
            rhat_max = torch.max(rhat_flat[finite])
            rhat_mean = torch.mean(rhat_flat[finite])
        else:
            rhat_max = torch.tensor(float("nan"), device=self.device, dtype=torch.float64)
            rhat_mean = torch.tensor(float("nan"), device=self.device, dtype=torch.float64)
        print("rhat_max", rhat_max)
        print("rhat_mean", rhat_mean)
        if rhat_exclude_edge_bins_applied > 0:
            print(
                "R-hat summary excludes "
                f"{rhat_exclude_edge_bins_applied} edge bins per side "
                f"({rhat_eval_bins} bins evaluated)."
            )
        elif rhat_exclude_edge_bins_requested > 0:
            print(
                "R-hat edge-bin exclusion requested "
                f"({rhat_exclude_edge_bins_requested}) but not applied (D={D_vals})."
            )
        print(f"Elapsed time: {end - start:.6f} s")

        # MCSE on n(x) bins using post-adaptation samples.
        samples = self.n_samples[:, adapt_end:, :].reshape(-1, self.n_samples.size(-1))
        var_d = samples.var(dim=0, unbiased=True)  # shape (D,)
        mcse_d = torch.sqrt(var_d / total_ess)      # shape (D,)

        print("MCSE for first 5 bins:", mcse_d[:5].tolist())
        print(f"Mean MCSE across bins: {mcse_d.mean():.5f}")
        if achieved_time is not None and mcse_d.max() < mcse_threshold:
            print(f"MCSE < {mcse_threshold} first achieved at step {achieved_step}")
            print(f"Reached time: {achieved_time:.2f} s")
        else:
            print(f"MCSE target not reached. MCSE max = {mcse_d.max():.4f}")
        diagnostics = populate_run_diagnostics(
            diagnostics={},
            adapt_end=adapt_end,
            hot_start=hot_start,
            burnin=burnin,
            hot_start_ratio=hot_start_ratio,
            adapt_end_ratio=adapt_end_ratio,
            n_steps=L,
            check_interval=check_interval,
            mcse_iat_window=mcse_iat_window,
            iat_final_window=iat_final_window,
            iat_max_fft_bytes=iat_max_fft_bytes,
            periodic_iat_error=periodic_iat_error,
            final_iat_error=final_iat_error,
            trials=trials,
            accepts=accepts,
            swap_trials=swap_trials,
            swap_accepts=swap_accepts,
            swap_calls=swap_calls,
            alpha_star_pca=alpha_star_pca,
            alpha_star_ws=alpha_star_ws,
            alpha_star_cg=alpha_star_cg,
            step_adapter_pca=step_adapter_pca,
            step_adapter_ws=step_adapter_ws,
            step_adapter_cg=step_adapter_cg,
            s_pca=self.s_pca,
            s_ws=self.s_ws,
            s_cg=self.s_cg,
            adapt_hot_interval=adapt_hot_interval,
            adapt_late_interval=adapt_late_interval,
            scale_adapt_t0=scale_adapt_t0,
            scale_adapt_gamma=scale_adapt_gamma,
            scale_adapt_kappa=scale_adapt_kappa,
            step_min=self.step_min,
            step_max=self.step_max,
            cg_cov_blend=self.cg_cov_blend,
            cg_cov_jitter=self.cg_cov_jitter,
            prob_metric=prob_metric,
            prob_min=prob_min,
            prob_min_trials_per_move=prob_min_trials_per_move,
            prob_esjd_whitened=prob_esjd_whitened,
            prob_whiten_jitter=prob_whiten_jitter,
            prob_whiten_shrinkage=prob_whiten_shrinkage,
            prob_whiten_weight_mode=prob_whiten_weight_mode,
            prob_swap_reward_weight=prob_swap_reward_weight,
            prob_swap_warmup_updates=prob_swap_warmup_updates,
            prob_utility_floor=prob_utility_floor,
            utility_est_pca=prob_state.utility_est_pca,
            utility_est_ws=prob_state.utility_est_ws,
            utility_est_cg=prob_state.utility_est_cg,
            swap_est_pca=prob_state.swap_est_pca,
            swap_est_ws=prob_state.swap_est_ws,
            swap_est_cg=prob_state.swap_est_cg,
            prob_last_obs_pca=prob_state.last_obs_pca,
            prob_last_obs_ws=prob_state.last_obs_ws,
            prob_last_obs_cg=prob_state.last_obs_cg,
            prob_last_swap_obs_pca=prob_state.last_swap_obs_pca,
            prob_last_swap_obs_ws=prob_state.last_swap_obs_ws,
            prob_last_swap_obs_cg=prob_state.last_swap_obs_cg,
            prob_trials_pca=prob_state.trials_pca,
            prob_trials_ws=prob_state.trials_ws,
            prob_trials_cg=prob_state.trials_cg,
            p_pca=self.p_pca,
            p_ws=self.p_ws,
            p_cg=self.p_cg,
            prob_history_steps=prob_state.history_steps,
            prob_history_pca=prob_state.history_pca,
            prob_history_ws=prob_state.history_ws,
            prob_history_cg=prob_state.history_cg,
            prob_history_utility_obs_pca=prob_state.history_utility_obs_pca,
            prob_history_utility_obs_ws=prob_state.history_utility_obs_ws,
            prob_history_utility_obs_cg=prob_state.history_utility_obs_cg,
            prob_history_utility_est_pca=prob_state.history_utility_est_pca,
            prob_history_utility_est_ws=prob_state.history_utility_est_ws,
            prob_history_utility_est_cg=prob_state.history_utility_est_cg,
            move_diag_summary=move_diag_summary,
            tau_ints=tau_ints,
            ess_per_chain=ess_per_chain,
            total_ess=total_ess,
            elapsed_sec=elapsed,
            ess_per_sec=ess_per_sec,
            rhat_split_segments=rhat_split_segments,
            rhat_exclude_edge_bins_requested=rhat_exclude_edge_bins_requested,
            rhat_exclude_edge_bins_applied=rhat_exclude_edge_bins_applied,
            rhat_eval_bins=rhat_eval_bins,
            rhat_max=rhat_max,
            rhat_mean=rhat_mean,
            rhat_by_replica=rhat_eval_by_replica,
            mcse_d=mcse_d,
            mcse_threshold=mcse_threshold,
            mcse_achieved_step=achieved_step,
            mcse_achieved_time=achieved_time,
            H_samples=self.H_samples,
        )
        diagnostics["sampler_mode"] = self.sampler_mode
        diagnostics["use_replica_exchange"] = self.use_replica_exchange

        # Alpha posterior weighting:
        # - computes competing weight models internally
        # - picks strategy via config.alpha_weight_method
        # - returns normalized weights for final reconstruction
        weights, logP_alpha, _ = select_alpha_weights(
            config=self.config,
            alphas=self.alphas,
            H_means=H_means,
            S_means=S_means,
            n_means=n_means,
            n_samples=self.n_samples,
            H_samples=self.H_samples,
            S_samples=self.S_samples,
            adapt_end=adapt_end,
            device=self.device,
            compute_lambdas=lambda A_hat_i: self.compute_lambdas_jerrell(A_hat=A_hat_i),
            diagnostics=diagnostics,
        )

        D_vals = torch.tensor(self.D_w, device=self.device, dtype=torch.float64)  # shape (K,)
        diagnostics["alphas"] = self.alphas
        diagnostics["H_means"] = H_means
        diagnostics["S_means"] = S_means
        diagnostics["w_alpha"] = weights
        diagnostics["logP_alpha"] = logP_alpha
        diagnostics["n_means"] = n_means
        diagnostics["D_w"] = D_vals
        return n_means, H_means, mcse_d, diagnostics

    def _reconstruct(self, n_means: torch.Tensor, H_means: torch.Tensor):
        """Reconstruct posterior mean spectrum A(w) from replica summaries."""
        A_est, w, area = reconstruct_posterior_mean(
            device=self.device,
            D_w=self.D_w,
            omega=self.omega,
            n_means=n_means,
            H_means=H_means,
            w_alpha_cached=getattr(self, "w_alpha_cached", None),
        )
        print("[DEBUG] area(A_est_un)=", float(area.detach().cpu()))
        return A_est, w, area

    def _credible_interval_from_samples(
        self,
        w: torch.Tensor,
        q_low: float = 0.025,
        q_high: float = 0.975,
        max_samples: int = 20000,
        seed: int = 0,
    ):
        """Compute posterior credible interval for A(w) from raw chain samples."""
        adapt_end = int(getattr(self, "adapt_end", int(0.8 * self.n_samples.shape[1])))
        return credible_interval_from_samples(
            n_samples=self.n_samples,
            D_w=self.D_w,
            omega=self.omega,
            device=self.device,
            w=w,
            adapt_end=adapt_end,
            q_low=q_low,
            q_high=q_high,
            max_samples=max_samples,
            seed=seed,
        )

    def run_max_ent(self):
        """Run MaxEnt sampling and reconstruction.

        Notes
        -----
        - This method is intentionally side-effect free: no plotting, no file I/O,
          no torch.compile. Those belong to runner scripts (e.g. scripts/run_maxent.py).
        """
        # 1) Run MCMC and gather diagnostics/statistics.
        n_means, H_means, mcse_n, diagnostics = self.run_replica_exchange()

        # 2) Cache alpha weights for downstream reconstruction helpers.
        if "w_alpha" in diagnostics:
            self.w_alpha_cached = diagnostics["w_alpha"].to(self.device)
        else:
            self.w_alpha_cached = None

        # 3) Reconstruct posterior mean estimate A(w).
        A_est, w, area_est = self._reconstruct(n_means, H_means)
        diagnostics["area_est_un"] = float(area_est.detach().cpu())

        # 4) Compute default 68% credible band (q16, q84).
        max_ci = int(getattr(self.config, "max_ci_samples", 20000))
        A_low, A_high, ci_diag = self._credible_interval_from_samples(
            w=w, q_low=0.16, q_high=0.84, max_samples=max_ci, seed=0
        )
        diagnostics.update(ci_diag)

        # 5) Return final outputs consumed by runner scripts.
        return A_est, A_low, A_high, mcse_n, diagnostics

