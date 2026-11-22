import numpy as np
import torch
import math
from typing import List, Tuple, Optional, Dict, Any
from maxent.io.xas import load_xas

# 1 データの読み込み
# 2 スペクトル計算
# 3 スペクトルからグリーン関数を計算

class Prepare:
    def prepare(self, file_name, cfg, beta, Nn, device):
        raw = load_xas(file_name)
        E, i0, itrans = raw.E, raw.I0, raw.Itrans
        omega, D = self.calc_spectrum(E, i0, itrans, cfg=cfg)
        # 確認
        # print("抽出後の要素数:", omega.size, D.size)
        omega = torch.tensor(omega, device=device)
        D = torch.tensor(D, device=device)
        # Noise policy is handled by the runner (run_maxent.py).
        # Prepare returns *noise-free* G_thm as observation placeholder.
        G_obs, G_thm, wn, omega_t, D_t, noise_info = self.calc_green(beta, Nn, omega, D, device=device)
        # print(G_obs[:5])
        return G_obs, G_thm, wn, omega_t, D_t 

    # # データ読み込み
    # def load_data(self, path: str):
    #     raw = load_xas(path)
    #     return raw.E, raw.I0, raw.Itrans

    def calc_spectrum(self, E: List, i0: List, itrans: List, cfg):
        mu = self._compute_mu(i0, itrans)
        E_edge = self._detect_edge_energy(E, mu)

        pre_bg, _pre_mask = self._fit_pre_edge(E, mu, E_edge)
        post_mean, _post_mask = self._post_edge_scale(E, mu, pre_bg, E_edge)

        mu_norm = self._normalize_mu(mu, pre_bg, post_mean)
        omega = self._make_omega(E, E_edge)
        D_raw = self._make_D_raw(omega, E_edge, mu_norm)
        w_lo = cfg["prepare"]["w_lo"]
        w_hi = cfg["prepare"]["w_hi"]
        n_grid = int(cfg.get("grid", {}).get("N", 0))
        floor = float(cfg.get("default_model", {}).get("floor", 1e-12))
        omega_sub, D_norm_sub = self._select_window(
            omega,
            D_raw,
            w_lo=w_lo,
            w_hi=w_hi,
            n_grid=n_grid,
            floor=floor,
        )
        return omega_sub, D_norm_sub

    def _compute_mu(self, i0, itrans):
        # 吸収係数 μ(E)=ln(I0/Itrans)
        return np.log(i0 / itrans)

    def _detect_edge_energy(self, E, mu):
        # ΔE を考慮した微分 → 二階微分最大で edge 推定（現状ロジック維持）
        dmu = np.gradient(mu, E, edge_order=2)
        d2mu = np.gradient(dmu, E)
        E_edge = E[np.argmax(d2mu)]
        return E_edge

    def _fit_pre_edge(self, E, mu, E_edge, pre_lo=-40, pre_hi=-20):
        pre_mask = (E > E_edge + pre_lo) & (E < E_edge + pre_hi)
        pre_coeff = np.polyfit(E[pre_mask], mu[pre_mask], 1)
        pre_bg = np.polyval(pre_coeff, E)
        return pre_bg, pre_mask

    def _post_edge_scale(self, E, mu, pre_bg, E_edge, post_lo=5, post_hi=50):
        post_mask = (E > E_edge + post_lo) & (E < E_edge + post_hi)
        post_mean = np.mean(mu[post_mask] - pre_bg[post_mask])
        return post_mean, post_mask

    def _normalize_mu(self, mu, pre_bg, post_mean):
        return (mu - pre_bg) / post_mean

    def _make_omega(self, E, E_edge):
        return E - E_edge

    def _make_D_raw(self, omega, E_edge, mu_norm):
        # 現状の D_raw = E_actual * mu_norm を維持
        E_actual = omega + E_edge
        return E_actual * mu_norm

    def _select_window(self, omega, D_raw, *, w_lo=-10, w_hi=30, n_grid=0, floor=1e-12):
        # Build a uniform omega grid to avoid irregular-point artifacts in A(omega).
        omega = np.asarray(omega, dtype=float)
        D_raw = np.asarray(D_raw, dtype=float)

        valid = np.isfinite(omega) & np.isfinite(D_raw)
        omega = omega[valid]
        D_raw = D_raw[valid]
        if omega.size < 2:
            raise ValueError("Not enough valid omega/D points after sanitization.")

        order = np.argsort(omega)
        omega = omega[order]
        D_raw = D_raw[order]

        mask = (omega >= w_lo) & (omega <= w_hi)
        omega_win = omega[mask]
        D_win = D_raw[mask]
        if omega_win.size < 2:
            raise ValueError(f"Window [{w_lo}, {w_hi}] contains fewer than 2 points.")

        n_target = int(n_grid) if int(n_grid) > 1 else int(omega_win.size)
        omega_sub = np.linspace(float(w_lo), float(w_hi), n_target)
        D_sub = np.interp(omega_sub, omega_win, D_win, left=D_win[0], right=D_win[-1])

        # Enforce non-negativity without dropping grid points.
        D_sub = np.clip(D_sub, a_min=float(floor), a_max=None)

        W = np.trapz(D_sub, omega_sub)
        if not np.isfinite(W) or W <= 0.0:
            raise ValueError(f"Invalid normalization integral in window: W={W}")
        D_norm = D_sub / W
        return omega_sub, D_norm

    def calc_green(self, beta, Nn, omega, D, device):
        omega_t = omega.to(device=device, dtype=torch.float64)
        D_t = D.to(device=device, dtype=torch.float64)

        wn = self._make_matsubara_wn(beta, Nn, device=omega_t.device)
        K = self._kernel_iwn(wn, omega_t)

        G_thm = self._integrate_trapz(K, D_t, omega_t)
        # No noise injection here; runner controls noise model & sigma_iw.
        noise_info: Dict[str, Any] = {"note": "no iω noise injected in Prepare"}
        G_obs = G_thm
        return G_obs, G_thm, wn, omega_t, D_t, noise_info

    def _make_matsubara_wn(self, beta: float, Nn: int, device):
        # Keep wn as *real* frequencies. Use (1j*wn) in kernels.
        return torch.tensor([(2*n+1)*math.pi/beta for n in range(-Nn, Nn)],
                            dtype=torch.float64, device=device)

    def _kernel_iwn(self, wn: torch.Tensor, omega_t: torch.Tensor):
        return 1.0 / (1j * wn.reshape(-1, 1) - omega_t.reshape(1, -1))

    def _integrate_trapz(self, K: torch.Tensor, D: torch.Tensor, omega_t: torch.Tensor) -> torch.Tensor:
        # K: (Nn, Nw), D: (Nw,), omega_t: (Nw,)
        integrand = K * D.to(torch.complex128)  # (Nn, Nw)
        return torch.trapz(integrand, omega_t, dim=1)  # (Nn,)

    def _add_complex_gaussian_noise(self, G: torch.Tensor, snr: float):
        amp_max = torch.max(torch.abs(G))
        # print("[DEBUG] amp_max:", amp_max)
        sigma = amp_max / snr
        noise = (torch.randn_like(G.real) + 1j*torch.randn_like(G.real)) * (
            sigma / torch.sqrt(torch.tensor(2.0, device=G.device))
        )
        noise_info = {"snr": float(snr), "amp_max": float(amp_max), "sigma_iw": float(sigma)}
        return G + noise, noise_info

if __name__ == "__main__":
    prepare = Prepare()
    beta  = 10.0     # 逆温度
    Nn    = 50      # Matsubara 点の数
    prepare.prepare(file_name='Data/k2cr2o7_rt_003.xdi', beta=beta, Nn=Nn)

# # プロット
# plt.figure(figsize=(6,4))
# plt.plot(E, mu_norm, color='C1', lw=1.2)
# plt.axvline(E_edge, ls='--', color='k', label=f'Edge={E_edge:.1f} eV')
# plt.xlim(E_edge-10, E_edge+70)
# plt.ylim(0, np.max(mu_norm[post_mask])*1.2)
# plt.xlabel('Energy (eV)')
# plt.ylabel('Normalized μ')
# plt.title('K$_2$Cr$_2$O$_7$ XANES')
# plt.legend()
# plt.tight_layout()
# plt.show()

# Nw    = 150      # 実軸スペクトル解像度
