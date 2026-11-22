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


class PCAMove:
    def __init__(self, operators, alphas, eps=1e-8):
        self.operators = operators
        self.alphas = alphas
        self.eps = eps

    @staticmethod
    def _log_jac_softmax(p: torch.Tensor) -> torch.Tensor:
        # Jacobian term up to additive constant (cancels in ratios).
        return torch.sum(torch.log(torch.clamp(p, min=1e-300)), dim=1)

    def suggest(self, x, n, H, C_pca, s_pca):
        """
        Symmetric random-walk proposal in u-space only:
          u' = u + N(0, Sigma_u),
        where p=softmax(u), n = p / w_x.
        x is kept fixed and only carried through for API consistency.

        Acceptance is computed on the induced u-space target:
          log pi_u = -H + alpha*S + log|J_softmax|.
        """
        R = self.alphas.shape[0]
        D = C_pca.shape[0]
        dtype = n.dtype
        device = n.device

        C = torch.nan_to_num(C_pca.to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
        step = torch.as_tensor(s_pca, device=device, dtype=dtype)
        if not torch.isfinite(step):
            step = torch.as_tensor(1e-3, device=device, dtype=dtype)

        # PCA proposal in u-space from eigendecomposition of Cov[u].
        C = 0.5 * (C + C.T)
        evals_u, evecs_u = torch.linalg.eigh(C)
        eval_floor = torch.as_tensor(self.eps, device=device, dtype=dtype)
        evals_u = torch.clamp(evals_u, min=eval_floor)
        scales_u = step * torch.sqrt(evals_u)
        z_u = torch.randn((R, D), device=device, dtype=dtype)
        delta_u = (z_u * scales_u.unsqueeze(0)) @ evecs_u.T

        # Current n -> simplex p using trapz-consistent weights.
        wx = _trapz_weights(self.operators.x.to(device=device, dtype=dtype)).to(dtype=dtype)
        wx = torch.clamp(wx, min=1e-30)
        p_cur = torch.clamp(n * wx.unsqueeze(0), min=1e-300)
        p_cur = p_cur / p_cur.sum(dim=1, keepdim=True).clamp_min(1e-300)
        u_cur = torch.log(p_cur)

        # Propose in u-space, then map back to positive normalized n.
        u_prop = u_cur + delta_u
        p_prop = torch.softmax(u_prop, dim=1)
        proposal_n = p_prop / wx.unsqueeze(0)

        # x is fixed in this design.
        proposal_x = x

        K_prop = self.operators._K_tau_x(proposal_x)
        proposal_H = self.operators._H(self.operators._h(proposal_n, K_prop))

        S_current = self.operators.entropy(n)
        S_proposal = self.operators.entropy(proposal_n)

        logpi_cur = -H + self.alphas * S_current + self._log_jac_softmax(p_cur)
        logpi_prop = -proposal_H + self.alphas * S_proposal + self._log_jac_softmax(p_prop)
        log_lr = logpi_prop - logpi_cur
        return MoveResult(log_lr, proposal_n, proposal_x, proposal_H)
