import torch
from torch.func import vmap
from typing import Callable, Optional

class Operators:
    """Forward operators for MaxEnt.

    NOTE (重要):
      - sigma は τ 空間の **共分散行列** Σ として扱う。
        期待 shape: (T, T)
      - 互換のため、sigma が (T,) のときは std(τ) とみなし、diag(std^2) に変換する。
    """

    def __init__(self, phi_inv, beta, tau, x, D_w, sigma, G_tau, jacobian=None,
                 x_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.phi_inv = phi_inv
        self.beta = beta
        self.tau = tau
        self.x = x
        self.D_w = D_w
        self.sigma = self._as_covariance(sigma)
        self.G_tau = G_tau
        self.jacobian = jacobian  # optional: function x -> dω/dx (same shape as x)

        # cache: L such that Σ = L L^T
        self._sigma_chol = None

        self._x_transform = x_transform if x_transform is not None else (lambda t: t)
        
    def _as_covariance(self, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 2:
            if sigma.shape[0] != sigma.shape[1]:
                raise ValueError(f"sigma must be square (T,T), got {tuple(sigma.shape)}")
            return sigma
        if sigma.ndim == 1:
            # interpret as std vector
            return torch.diag(sigma ** 2)
        raise ValueError(f"sigma must be (T,) or (T,T), got {tuple(sigma.shape)}")

    def sigma_cholesky(self) -> torch.Tensor:
        """Return lower-triangular L such that sigma = L L^T (with jitter if needed)."""
        if self._sigma_chol is not None:
            return self._sigma_chol

        Sigma = 0.5 * (self.sigma + self.sigma.T)
        T = Sigma.shape[0]
        I = torch.eye(T, device=Sigma.device, dtype=Sigma.dtype)

        jitter = 1e-12
        last_err = None
        for _ in range(12):
            try:
                self._sigma_chol = torch.linalg.cholesky(Sigma + jitter * I)
                return self._sigma_chol
            except RuntimeError as e:
                last_err = e
                jitter *= 10.0

        raise RuntimeError(f"sigma is not SPD even after jitter; last error: {last_err}")

    def whiten_tau(self, r_tau: torch.Tensor) -> torch.Tensor:
        """Compute Σ^{-1/2} r in τ-space using Cholesky.

        r_tau: (..., T) -> (..., T)
        """
        L = self.sigma_cholesky()  # (T,T)
        # solve L y = r^T  => y = L^{-1} r
        yT = torch.linalg.solve_triangular(L, r_tau.transpose(-1, -2), upper=False)
        return yT.transpose(-1, -2)

    def apply_precision(self, v_tau: torch.Tensor) -> torch.Tensor:
        """Compute Σ^{-1} v in τ-space.

        v_tau: (..., T) -> (..., T)
        """
        L = self.sigma_cholesky()
        yT = torch.linalg.solve_triangular(L, v_tau.transpose(-1, -2), upper=False)
        zT = torch.linalg.solve_triangular(L.transpose(-1, -2), yT, upper=True)
        return zT.transpose(-1, -2)

    def _domega_dx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian dω/dx for ω = φ^{-1}(x).
        If jacobian callable is provided, use it.
        Otherwise, use finite difference (central).
        """
        if self.jacobian is not None:
            return self.jacobian(x)
        eps = 1e-6
        xp = (x + eps).clamp(0.0, 1.0)
        xm = (x - eps).clamp(0.0, 1.0)
        wp = vmap(self.phi_inv)(xp)
        wm = vmap(self.phi_inv)(xm)
        return (wp - wm) / (xp - xm + 1e-30)

    def _K_tau_x(self, x: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: interpret incoming "x" as parameter y; transform to monotone x if needed
        x = self._x_transform(x)
        # φ⁻¹(x) → ω  shape (R, D)
        omega = vmap(self.phi_inv)(x)

        tau_b = self.tau.view(1, -1, 1)  # (1, T, 1)
        omega_b = omega.view(-1, 1, omega.size(-1))  # (R, 1, D)
        beta = float(self.beta)

        # Numerically stable fermionic kernel with fermionic sign:
        #   G(tau) = -∫ dω [exp(-tau*omega)/(1 + exp(-beta*omega))] A(omega)
        # so K carries a leading minus sign.
        log_kernel = -tau_b * omega_b + torch.nn.functional.logsigmoid(beta * omega_b)
        K_tau_omega = -torch.exp(log_kernel)  # (R, T, D)

        D_vals = torch.nan_to_num(self.D_w, nan=1e-8, posinf=1e8, neginf=1e-8)  # (R, D) or (D,)
        J = torch.nan_to_num(self._domega_dx(x), nan=1e-12, posinf=1e12, neginf=1e-12)  # (R, D) or (D,)

        R, T, D = K_tau_omega.shape
        if D_vals.ndim == 1 and D_vals.shape[0] == D:
            D_vals = D_vals.unsqueeze(0).expand(R, D).contiguous()
        if J.ndim == 1 and J.shape[0] == D:
            J = J.unsqueeze(0).expand(R, D).contiguous()

        K_tau_x = K_tau_omega * D_vals.unsqueeze(1) * J.unsqueeze(1)
        return K_tau_x

    # 各レプリカ毎に whitened residual h(τ) を計算
    def _h(self, n: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        integrand = K * n.unsqueeze(1)
        Kn = torch.trapz(integrand, self.x, dim=2)  # (R, T)
        r = Kn - self.G_tau  # (R, T)
        return self.whiten_tau(r)

    # ハミルトニアン H = ∫ dτ || Σ^{-1/2} r(τ) ||^2
    def _H(self, h: torch.Tensor) -> torch.Tensor:
        # Gaussian likelihood with whitening uses discrete sum over tau bins.
        return 0.5 * torch.sum(h ** 2, dim=1)
    
    def entropy(self, n: torch.Tensor) -> torch.Tensor:
        # n: (R, X) か (R, X, ?) などあなたの形に合わせる
        eps = 1e-30
        n_safe = torch.clamp(n, min=eps)
        # m_safe = torch.clamp(self.D_w, min=eps)
        integrand = - n_safe * torch.log(n_safe)
        return torch.trapz(integrand, self.x, dim=-1)  # (R,)

    def _Q(self, H: torch.Tensor, S: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return H - alpha * S

    def normalize(self, n: torch.Tensor) -> torch.Tensor:
        Z = torch.trapz(n, self.x, dim=1)
        return n / Z.unsqueeze(1)
