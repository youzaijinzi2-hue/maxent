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


class RandomWalkMHMove:
    """Simple isotropic random-walk MH move in unconstrained u-space."""

    def __init__(self, operators, alphas):
        self.operators = operators
        self.alphas = alphas

    @staticmethod
    def _log_jac_softmax(p: torch.Tensor) -> torch.Tensor:
        # Jacobian term up to an additive constant (cancels in MH ratio).
        return torch.sum(torch.log(torch.clamp(p, min=1e-300)), dim=1)

    def suggest(self, x, n, H, step_size):
        dtype = n.dtype
        device = n.device

        step = torch.as_tensor(step_size, device=device, dtype=dtype)
        if not torch.isfinite(step):
            step = torch.as_tensor(1e-3, device=device, dtype=dtype)

        wx = _trapz_weights(self.operators.x.to(device=device, dtype=dtype)).to(dtype=dtype)
        wx = torch.clamp(wx, min=1e-30)

        p_cur = torch.clamp(n * wx.unsqueeze(0), min=1e-300)
        p_cur = p_cur / p_cur.sum(dim=1, keepdim=True).clamp_min(1e-300)
        u_cur = torch.log(p_cur)

        # Truly simple isotropic Gaussian RW: u' = u + step * eps, eps~N(0, I)
        u_prop = u_cur + step * torch.randn_like(u_cur)
        p_prop = torch.softmax(u_prop, dim=1)
        proposal_n = p_prop / wx.unsqueeze(0)
        proposal_x = x

        K_prop = self.operators._K_tau_x(proposal_x)
        proposal_H = self.operators._H(self.operators._h(proposal_n, K_prop))

        S_current = self.operators.entropy(n)
        S_proposal = self.operators.entropy(proposal_n)

        logpi_cur = -H + self.alphas * S_current + self._log_jac_softmax(p_cur)
        logpi_prop = -proposal_H + self.alphas * S_proposal + self._log_jac_softmax(p_prop)
        log_lr = logpi_prop - logpi_cur
        return MoveResult(log_lr, proposal_n, proposal_x, proposal_H)
