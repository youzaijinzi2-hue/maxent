# maxent/mcmc/moves/mala.py
import math
import torch
from maxent.mcmc.moves.base import MoveResult


class CGMALAMove:
    def __init__(self, operators, alphas, eps: float = 1e-8):
        self.operators = operators
        self.alphas = alphas
        self.eps = eps
        self.device = alphas.device
        
    def suggest(
        self,
        x,
        n,
        H,
        K_current,
        G_tau,
        sigma,
        C_cg,
        s_cg,
        cg_cov_blend: float = 0.20,
        cg_cov_jitter: float = 1e-6,
    ):
        """Conjugate-gradient preconditioned MALA move.

        IMPORTANT:
          - sigma is treated as τ-space covariance matrix Σ (T,T) via Operators.
          - We intentionally ignore the function arguments (G_tau, sigma) here and
            instead use operators.G_tau / operators.sigma to keep a single source of truth
            (same philosophy as the existing NOTE).
        """
        # --- 前準備 ---
        R = self.alphas.shape[0]  # レプリカ数
        D = C_cg.shape[0]         # 次元
        N = G_tau.shape[0]        # 虚時間メッシュ数

        # Operators を単一ソースにする（引数の sigma/G_tau とズレないように）
        tau = self.operators.tau
        xgrid = self.operators.x
        G_tau_vec = self.operators.G_tau

        # sigma is now covariance Σ (T,T)
        # Cholesky factor L such that Σ = L L^T (with jitter inside Operators)
        L_tau = self.operators.sigma_cholesky()  # (T,T)
        Lb_tau = L_tau.unsqueeze(0)              # (1,T,T) for broadcast over replicas

        # trapz 重み（∫ f(t) dt ≈ Σ w_i f_i）
        def _trapz_weights(grid_1d: torch.Tensor) -> torch.Tensor:
            """Create trapezoidal-rule weights for a 1D grid."""
            if grid_1d.ndim != 1:
                raise ValueError(f"grid must be 1D, got shape={tuple(grid_1d.shape)}")
            npt = grid_1d.numel()
            if npt < 2:
                raise ValueError("grid must have at least 2 points")
            w = torch.empty(npt, device=grid_1d.device, dtype=grid_1d.dtype)
            w[0] = 0.5 * (grid_1d[1] - grid_1d[0])
            w[-1] = 0.5 * (grid_1d[-1] - grid_1d[-2])
            if npt > 2:
                w[1:-1] = 0.5 * (grid_1d[2:] - grid_1d[:-2])
            return w

        w_x = _trapz_weights(xgrid)   # (D,)
        epsilon = torch.as_tensor(s_cg, device=n.device, dtype=n.dtype).reshape(())

        def _whiten_r(r_tau: torch.Tensor) -> torch.Tensor:
            """h = Σ^{-1/2} r via Cholesky. r_tau: (R,T) -> h: (R,T)."""
            y = torch.linalg.solve_triangular(Lb_tau, r_tau.unsqueeze(-1), upper=False)
            return y.squeeze(-1)

        def _apply_A_to_vec(v_tau: torch.Tensor) -> torch.Tensor:
            """Compute (A v) where A := Σ^{-1}."""
            y = torch.linalg.solve_triangular(Lb_tau, v_tau.unsqueeze(-1), upper=False)  # (R,T,1)
            out = torch.linalg.solve_triangular(Lb_tau.transpose(-1, -2), y, upper=True)  # (R,T,1)
            return out.squeeze(-1)

        def _apply_A_to_K(K_tau_x: torch.Tensor) -> torch.Tensor:
            """Compute A K for K: (R,T,D) with A := Σ^{-1}."""
            y = torch.linalg.solve_triangular(Lb_tau, K_tau_x, upper=False)               # (R,T,D)
            return torch.linalg.solve_triangular(Lb_tau.transpose(-1, -2), y, upper=True)  # (R,T,D)

        if not bool(torch.isfinite(epsilon).item()):
            raise RuntimeError(f"s_cg is not finite: {s_cg}")
        if float(epsilon.item()) <= 0.0:
            raise RuntimeError(f"s_cg must be positive, got: {float(epsilon.item())}")
        
        # --- x の提案 ---
        # NOTE:
        #   cg(MALA) move では n のみを MALA で更新し、x は固定する。
        #   ここで x まで (s_cg^2) スケールのランダムウォークで動かすと、
        #   少し s_cg が増えただけで K が大きく変わり受容率が崩壊しやすい。
        proposal_x = x
        proposal_K = K_current

        # --- 現状態のエネルギー H_current & 勾配 g_current ---
        nw = n * w_x.view(1, -1)                    # (R,D)
        Kn_curr = torch.bmm(K_current, nw.unsqueeze(-1)).squeeze(-1)  # (R,T)
        r_curr = Kn_curr - G_tau_vec.view(1, -1)                                # (R,T)
        h_curr = _whiten_r(r_curr)                                              # (R,T)
        H_current = 0.5 * (h_curr ** 2).sum(dim=1)                               # (R,)

        v_curr = _apply_A_to_vec(r_curr)                                        # (R,T)
        KTu_curr = (K_current.transpose(1, 2) @ v_curr.unsqueeze(-1)).squeeze(-1)  # (R,D)
        dHdn_curr = 2.0 * KTu_curr * w_x.view(1, -1)                            # (R,D)
        g_current = -2.0 * self.alphas.view(-1, 1) * dHdn_curr                  # (R,D)

        # ============================================================
        # Full (non-diagonal) preconditioner for u-space MALA
        #
        # We build a per-replica SPD precision matrix
        #   M = (2*alpha) * HessH_n + jitter*I
        # where the (Gauss–Newton) Hessian of H w.r.t. n is
        #   HessH_n = 2 * diag(w_x) (K^T A K) diag(w_x)
        # with A := Σ^{-T/2} diag(w_tau) Σ^{-1/2}.
        #
        # This removes the diagonal approximation (diag(K^T A K)).
        # ============================================================

        # KTAK = K^T A K  (R,D,D)
        AK = _apply_A_to_K(K_current)  # (R,T,D)
        KTAK = torch.bmm(K_current.transpose(1, 2), AK)  # (R,D,D)

        # HessH_n = 2 * diag(w_x) KTAK diag(w_x)
        wx = w_x.to(K_current.dtype)
        Wouter = (wx.view(1, -1, 1) * wx.view(1, 1, -1))  # (1,D,D)
        HessH = 2.0 * KTAK * Wouter                       # (R,D,D)

        # Precision in u-space
        #   M_hess = (2*alpha)*HessH + jitter*I
        jitter0 = 1e-8
        I = torch.eye(D, device=K_current.device, dtype=K_current.dtype).unsqueeze(0)  # (1,D,D)
        M = (2.0 * self.alphas.view(-1, 1, 1)) * HessH + jitter0 * I  # (R,D,D)

        # Blend in adaptive covariance precision from sampler-side C_cg.
        # This makes C_cg adaptation effective for CG move while keeping Hessian guidance.
        rho = float(cg_cov_blend)
        if not math.isfinite(rho):
            rho = 0.20
        rho = min(max(rho, 0.0), 1.0)
        cov_jitter = float(cg_cov_jitter)
        if (not math.isfinite(cov_jitter)) or (cov_jitter <= 0.0):
            cov_jitter = 1e-6
        if rho > 0.0:
            C = torch.nan_to_num(
                C_cg.to(device=K_current.device, dtype=K_current.dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            C = 0.5 * (C + C.T)
            I2 = torch.eye(D, device=C.device, dtype=C.dtype)
            jit = max(cov_jitter, float(torch.finfo(C.dtype).eps))
            for _ in range(8):
                L_c, info_c = torch.linalg.cholesky_ex(C + jit * I2)
                if int(info_c.item()) == 0:
                    break
                jit *= 10.0
            if int(info_c.item()) != 0:
                evals, evecs = torch.linalg.eigh(C)
                evals = torch.clamp(evals, min=jit)
                C_spd = evecs @ torch.diag(evals) @ evecs.T
                L_c = torch.linalg.cholesky(C_spd)
            # sampler-side precision proxy from adaptive covariance
            P0 = torch.cholesky_inverse(L_c)  # (D,D)
            # scale-match P0 to Hessian precision level to avoid dominating.
            diag_m = torch.diagonal(M, dim1=-2, dim2=-1).mean(dim=1)   # (R,)
            diag_p = torch.diagonal(P0, dim1=-2, dim2=-1).mean().clamp_min(torch.tensor(1e-30, device=P0.device, dtype=P0.dtype))
            scale = (diag_m / diag_p).clamp_min(torch.tensor(1e-30, device=M.device, dtype=M.dtype))  # (R,)
            P = P0.unsqueeze(0) * scale.view(-1, 1, 1)  # (R,D,D)
            M = (1.0 - rho) * M + rho * P

        # --- Conditioning / jitter strengthening (ABS + REL) ---
        # Cholesky success does NOT guarantee that M is well-conditioned.
        # If M is nearly singular, M^{-1} and M^{-1/2} explode -> huge drift/noise -> accept=0.
        # Add per-replica jitter floor:
        #   M <- sym(M) + delta * I
        #   delta = max(delta_abs, delta_rel * median(diag(M))).
        M = 0.5 * (M + M.transpose(-1, -2))  # enforce symmetry
        diagM = torch.diagonal(M, dim1=-2, dim2=-1)  # (R,D)
        diag_med = torch.median(diagM, dim=1).values  # (R,)
        delta_abs = torch.tensor(1e-2, device=M.device, dtype=M.dtype)  # tune: 1e-4..1e0
        delta_rel = torch.tensor(1e-2, device=M.device, dtype=M.dtype)
        delta = torch.maximum(delta_abs.expand_as(diag_med), (delta_rel * diag_med))
        M = M + delta.view(-1, 1, 1) * I

        def _batch_cholesky_spd(M_in: torch.Tensor, jitter: float = 1e-8, max_tries: int = 8):
            """Batch-stabilized Cholesky: returns L such that (M_in + jitter*I) = L L^T.
            If some replicas fail, we increase jitter globally and retry.
            """
            I_local = torch.eye(M_in.shape[-1], device=M_in.device, dtype=M_in.dtype).expand(M_in.shape[0], -1, -1)
            M_work = 0.5 * (M_in + M_in.transpose(-1, -2))  # symmetrize
            jit = jitter
            for _ in range(max_tries):
                L, info = torch.linalg.cholesky_ex(M_work + jit * I_local)
                if (info == 0).all():
                    return L, jit
                jit *= 10.0
            # Last resort: add big jitter (still symmetrized)
            L, info = torch.linalg.cholesky_ex(M_work + jit * I_local)
            if (info != 0).any():
                raise RuntimeError(f"Cholesky failed even after jitter escalation. max info={int(info.max())}, jitter={jit}")
            return L, jit

        L_M, used_jitter = _batch_cholesky_spd(M, jitter=jitter0, max_tries=8)  # (R,D,D), lower-tri

        # logdet(M) from Cholesky
        logdetM = 2.0 * torch.sum(torch.log(torch.diagonal(L_M, dim1=-2, dim2=-1)), dim=1)  # (R,)

        # Helper: invM @ v via Cholesky solves
        def _invM_times(v: torch.Tensor) -> torch.Tensor:
            # v: (R,D) -> (R,D)
            y = torch.linalg.solve_triangular(L_M, v.unsqueeze(-1), upper=False)                 # (R,D,1)
            x = torch.linalg.solve_triangular(L_M.transpose(-1, -2), y, upper=True).squeeze(-1) # (R,D)
            return x

        # Helper: sample eta ~ N(0, M^{-1}) using Cholesky of M
        def _sample_Minv(z: torch.Tensor) -> torch.Tensor:
            # z ~ N(0,I): (R,D) -> eta = L^{-1} z has cov M^{-1}
            eta = torch.linalg.solve_triangular(L_M, z.unsqueeze(-1), upper=False).squeeze(-1)  # (R,D)
            return eta
        
        def _stat(name, x):
            x = x.detach()
            finite = torch.isfinite(x)
            if finite.all():
                q = torch.quantile(x, torch.tensor([0.0, 0.5, 1.0], device=x.device, dtype=x.dtype))
                print(f"[DEBUG] {name} min/med/max = {q[0].item():.3e} {q[1].item():.3e} {q[2].item():.3e}")
            else:
                print(f"[DEBUG] {name} has non-finite: {finite.float().mean().item():.3f} finite ratio")


        # --- u-space MALA with trapz-normalization built-in ---
        # We parametrize n so that sum_i w_x[i] * n[i] = 1 always:
        #   p = softmax(u)  (sum p = 1, p_i > 0)
        #   n_i = p_i / w_i  -> sum w_i n_i = sum p_i = 1
        # This removes the need for operators.normalize() inside the proposal,
        # which was a common source of NaN/Inf and crashes.
        eps_safe = 1e-300
        w_safe = torch.clamp(w_x, min=eps_safe).view(1, -1)  # (1,D)

        def _u_to_n_trapz(u_in: torch.Tensor) -> torch.Tensor:
            p = torch.softmax(u_in, dim=1)                    # (R,D), sum=1
            return p / w_safe                                 # (R,D), sum(w*n)=1

        def _log_jac_softmax(p: torch.Tensor) -> torch.Tensor:
            # Jacobian term for softmax (up to an additive constant)
            return torch.sum(torch.log(torch.clamp(p, min=eps_safe)), dim=1)  # (R,)

        # initialize u from current n by p = w*n (should already sum to 1; renorm just in case)
        p0 = torch.clamp(w_safe * n, min=eps_safe)
        p0 = p0 / p0.sum(dim=1, keepdim=True)
        u = torch.log(p0).detach().requires_grad_(True)        # (R,D)  (logits; shift-invariant)

        n_u = _u_to_n_trapz(u)                                 # (R,D)
        nw_u = n_u * w_x.view(1, -1)
        Kn_u = torch.bmm(K_current, nw_u.unsqueeze(-1)).squeeze(-1)

        r_u = Kn_u - G_tau_vec.view(1, -1)
        h_u = _whiten_r(r_u)
        H_u = 0.5 * (h_u ** 2).sum(dim=1)                                     # (R,)
        p_u = torch.softmax(u, dim=1)
        S_u = self.operators.entropy(n_u)   # ← 実装必要
        logpi_u = - H_u + self.alphas * S_u + _log_jac_softmax(p_u)          # (R,)

        logpi_u = -H_u + self.alphas * S_u + _log_jac_softmax(p_u)
        grad_u = torch.autograd.grad(logpi_u.sum(), u)[0]                      # (R,D)
        drift_vec = _invM_times(grad_u)                                        # (R,D)
        z_u = torch.randn_like(u)                                              # (R,D)
        # Full-preconditioned MALA:
        #   q(u'|u) = N(u + 0.5*eps^2*M^{-1}grad logpi, eps^2*M^{-1})
        mu_u = u + 0.5 * (epsilon ** 2) * drift_vec
        eta = _sample_Minv(z_u)
        u_prop = mu_u + epsilon * eta
        # _stat("||du||", torch.linalg.norm(u_prop - u, dim=1))
        
        # Evaluate at proposal for reverse drift
        u_prop_req = u_prop.detach().requires_grad_(True)
        n_prop_u = _u_to_n_trapz(u_prop_req)
        nw_prop = n_prop_u * w_x.view(1, -1)
        Kn_prop_u = torch.bmm(K_current, nw_prop.unsqueeze(-1)).squeeze(-1) 
        r_prop_u = Kn_prop_u - G_tau_vec.view(1, -1)
        h_prop_u = _whiten_r(r_prop_u)
        H_prop_u = 0.5 * (h_prop_u ** 2).sum(dim=1)
        p_prop = torch.softmax(u_prop_req, dim=1)
        S_prop_u = self.operators.entropy(n_prop_u)
        logpi_u_prop = -H_prop_u + self.alphas * S_prop_u + _log_jac_softmax(p_prop)
        grad_u_prop = torch.autograd.grad(logpi_u_prop.sum(), u_prop_req)[0]
        mu_u_prop = u_prop + 0.5 * (epsilon ** 2) * _invM_times(grad_u_prop)

        # Gaussian q in u-space with full precision M:
        #   log q(u'|u) = 0.5 logdet(M) - (1/(2 eps^2)) (u'-mu)^T M (u'-mu) + const
        du_fwd = (u_prop - mu_u)        # (R,D)
        du_bwd = (u - mu_u_prop)        # (R,D)

        # quad = du^T M du (batch)
        quad_fwd = torch.bmm(du_fwd.unsqueeze(1), torch.bmm(M, du_fwd.unsqueeze(2))).squeeze(-1).squeeze(-1)  # (R,)
        quad_bwd = torch.bmm(du_bwd.unsqueeze(1), torch.bmm(M, du_bwd.unsqueeze(2))).squeeze(-1).squeeze(-1)  # (R,)

        log_q_curr2prop = 0.5 * logdetM - 0.5 * quad_fwd / (epsilon ** 2)
        log_q_prop2curr = 0.5 * logdetM - 0.5 * quad_bwd / (epsilon ** 2)
        
        log_lr_cg = (logpi_u_prop - logpi_u) + (log_q_prop2curr - log_q_curr2prop)

        # Outputs in n-space (trapz-normalized)
        # CRITICAL: detach to avoid autograd graph accumulation across MCMC steps.
        proposal_n = n_prop_u.detach()
        H_prop = H_prop_u.detach()
        log_lr_cg = log_lr_cg.detach()

        # Keep representation constraints consistent across moves.
        # NOTE: If normalize() is nonlinear (clamp/renorm), it is not exactly reflected in q.
        # In practice this still tends to stabilize the move dramatically.
        # (u-space MALA already computed H_prop and log_lr_cg)
        
        # import random
        # if random.random() < 0.1:
        #     print("[DEBUG] log_pi_diff", log_pi_diff)
        #     print("[DEBUG] log_q_prop2curr - log_q_curr2prop", log_q_prop2curr - log_q_curr2prop)
        #     print("[DEBUG] log_lr_cg", log_lr_cg)
        #     print("[DEBUG] ||proposal_n - mu_curr||^2", (proposal_n - mu_curr)**2)
        #     print("[DEBUG] ||n - mu_prop||^2", (n - mu_prop)**2)
            
        return MoveResult(
            log_lr=log_lr_cg,
            proposal_n=proposal_n,
            proposal_x=proposal_x,
            proposal_H=H_prop,
        )

    def conjugate_gradient(self, B: int, A: torch.Tensor, b: torch.Tensor,
                           x0: torch.Tensor = None, tol: float = 1e-8, maxiter: int = None) -> torch.Tensor:
        """共役勾配法 (Conjugate Gradient法) を用いて最小化問題を解く.
        A x = b の x を求めるアルゴリズム.
        """
        n = b.shape[1]
        if x0 is None:
            x = torch.zeros(B, n, device=self.device, dtype=torch.float64)
        else:
            x = x0.clone()
        if maxiter is None:
            maxiter = n
        r = b - torch.bmm(A, x.unsqueeze(2)).squeeze(2)
        p = r.clone()
        rs_old = r ** 2
        for _ in range(1, maxiter + 1):
            Ap = torch.bmm(A, p.unsqueeze(2)).squeeze(2)
            alpha = rs_old / (p * Ap)
            x += alpha * p
            r -= alpha * Ap
            rs_new = r ** 2
            if torch.max(torch.sqrt(rs_new)) < tol:
                break
            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new
        return x

    def make_spd(self, A: torch.Tensor, jitter0: float = 1e-8, max_tries: int = 8):
        A = 0.5 * (A + A.T)
        I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        jitter = jitter0
        for _ in range(max_tries):
            try:
                _ = torch.linalg.cholesky(A + jitter * I)
                return A + jitter * I
            except RuntimeError:
                jitter *= 10.0
        w, Q = torch.linalg.eigh(A)
        w = torch.clamp(w, min=jitter)
        return (Q * w) @ Q.T
