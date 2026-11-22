import torch
from maxent.mcmc.moves.base import MoveResult


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


class WeightSharingMove:
    """
    Theoretically consistent WS-like move:
      - symmetric 1D shared-direction RW in u-space for n proposal
      - x is kept fixed (no x proposal)

    Target in u-space uses:
      log pi_u = -H + alpha*S + log|J_softmax|
    so MH acceptance remains explicit and state-consistent.
    """

    def __init__(self, operators, alphas, ws_k: int = 4, eps=1e-8):
        self.operators = operators
        self.alphas = alphas
        self.ws_k = max(2, int(ws_k))
        self.eps = eps

    @staticmethod
    def _log_jac_softmax(p: torch.Tensor) -> torch.Tensor:
        # Jacobian term up to additive constant (cancels in ratios).
        return torch.sum(torch.log(torch.clamp(p, min=1e-300)), dim=1)

    def _build_shared_direction(self, D: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Build a deterministic WS direction from barycentric-style coefficients.
        Depends only on the fixed x-grid (not on state), so the proposal stays symmetric.
        """
        k = min(self.ws_k, D)
        idx = torch.randperm(D, device=device)[:k]
        lam0 = int(idx[0].item())

        xg = self.operators.x.to(device=device, dtype=dtype)
        q = torch.zeros(D, device=device, dtype=dtype)
        q[lam0] = -1.0

        idx_list = [int(i.item()) for i in idx]
        others0 = [j for j in idx_list if j != lam0]

        for lam in others0:
            num = torch.tensor(1.0, device=device, dtype=dtype)
            den = torch.tensor(1.0, device=device, dtype=dtype)
            for j in others0:
                if j == lam:
                    continue
                num = num * (xg[j] - xg[lam0])
                den = den * (xg[j] - xg[lam])
            # keep finite even when nodes are close
            den = torch.where(torch.abs(den) < 1e-30, torch.sign(den) * 1e-30 + (den == 0).to(dtype) * 1e-30, den)
            q[lam] = num / den

        # Center to remove softmax shift-invariant mode.
        q = q - torch.mean(q)
        nrm = torch.linalg.norm(q).clamp_min(torch.tensor(1e-12, device=device, dtype=dtype))
        q = q / nrm
        return q

    def suggest(self, x, n, H, C_ws, s_ws):
        R = self.alphas.shape[0]
        D = C_ws.shape[0]
        dtype = n.dtype
        device = n.device

        C = torch.nan_to_num(C_ws.to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
        step = torch.as_tensor(s_ws, device=device, dtype=dtype)
        if not torch.isfinite(step):
            step = torch.as_tensor(1e-3, device=device, dtype=dtype)
        proposal_x = x

        # Build current simplex representation p from n (trapz-consistent).
        wx = _trapz_weights(self.operators.x.to(device=device, dtype=dtype)).to(dtype=dtype)
        wx = torch.clamp(wx, min=1e-30)
        p_cur = torch.clamp(n * wx.unsqueeze(0), min=1e-300)
        p_cur = p_cur / p_cur.sum(dim=1, keepdim=True).clamp_min(1e-300)
        u_cur = torch.log(p_cur)

        # Shared one-dimensional direction in u-space.
        q = self._build_shared_direction(D=D, device=device, dtype=dtype)
        cq = C @ q
        q_scale = torch.sqrt(torch.dot(q, cq).clamp_min(torch.tensor(1e-30, device=device, dtype=dtype)))
        dir_u = cq / q_scale

        # Symmetric 1D RW along the shared direction.
        s = torch.randn(R, device=device, dtype=dtype) * step
        u_prop = u_cur + s.unsqueeze(1) * dir_u.unsqueeze(0)

        p_prop = torch.softmax(u_prop, dim=1)
        proposal_n = p_prop / wx.unsqueeze(0)

        proposal_H = self.operators._H(
            self.operators._h(proposal_n, self.operators._K_tau_x(proposal_x))
        )

        S_current = self.operators.entropy(n)
        S_proposal = self.operators.entropy(proposal_n)

        logpi_cur = -H + self.alphas * S_current + self._log_jac_softmax(p_cur)
        logpi_prop = -proposal_H + self.alphas * S_proposal + self._log_jac_softmax(p_prop)
        log_lr = logpi_prop - logpi_cur

        return MoveResult(
            log_lr=log_lr,
            proposal_n=proposal_n,
            proposal_x=proposal_x,
            proposal_H=proposal_H,
        )
