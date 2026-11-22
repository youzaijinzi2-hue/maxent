from __future__ import annotations
import torch
import math
from maxent.mcmc.sampler import MaxEnt
import numpy as np

def _refine_peak_positions_weighted3(
    omega_roi: torch.Tensor,
    A_roi: torch.Tensor,
    idx_local: torch.Tensor,
) -> torch.Tensor:
    """
    Sub-grid refinement of argmax positions by weighted 3-point local average.
    Supports A_roi shape (..., K) with idx_local shape (...,).
    """
    K = int(A_roi.shape[-1])
    peak = omega_roi[idx_local]
    if K < 3:
        return peak

    idx_flat = idx_local.reshape(-1)
    peak_flat = peak.reshape(-1)
    interior = (idx_flat > 0) & (idx_flat < (K - 1))
    if not torch.any(interior):
        return peak

    offsets = torch.tensor([-1, 0, 1], device=idx_local.device, dtype=torch.long)
    idx_i = idx_flat[interior]
    idx3 = idx_i.unsqueeze(1) + offsets.unsqueeze(0)  # (M_int, 3)

    A_flat = A_roi.reshape(-1, K)
    A3 = A_flat[interior].gather(1, idx3).clamp_min(0.0)  # (M_int, 3)
    w3 = omega_roi[idx3]  # (M_int, 3)
    denom = A3.sum(dim=1).clamp_min(1e-30)
    peak_int = (w3 * A3).sum(dim=1) / denom
    peak_flat[interior] = peak_int

    return peak_flat.reshape(idx_local.shape)

def _compute_peak_pos_roi_from_sampler(
    sampler: MaxEnt,
    diagnostics: dict,
    roi: tuple[float, float],
    max_samples: int = 5000,
    seed: int = 0,
) -> torch.Tensor:
    """
    sampler が保持している self.n_samples から、CI と同じ「混合リサンプリング」で
    A = n * D_w を作り、ROI 内 argmax のピーク位置 ω_peak を多数サンプルして返す。
    返り値: peak_pos_roi shape (M,)
    """
    wmin, wmax = float(roi[0]), float(roi[1])
    if wmax <= wmin:
        raise ValueError(f"roi must satisfy roi[0] < roi[1], got {roi}")

    # 必須テンソル
    omega = sampler.omega.to(dtype=torch.float64)
    D_w = sampler.D_w.to(dtype=torch.float64)
    n_samples = sampler.n_samples  # (R, Lp1, K)
    if n_samples is None:
        raise RuntimeError("sampler.n_samples is None. run_replica_exchange did not store samples?")

    R, Lp1, K = n_samples.shape

    # adapt_end は diagnostics に入っている（run_replica_exchange で保存）
    adapt_end = int(diagnostics.get("adapt_end", int(0.8 * (Lp1 - 1))))
    adapt_end = max(0, min(adapt_end, Lp1 - 1))

    post_len = Lp1 - adapt_end
    total_available = int(R * post_len)
    if total_available <= 0:
        raise RuntimeError("No post-burnin samples available for peak-pos estimation.")

    M = min(int(max_samples), total_available)

    # 混合重み w は diagnostics["w"] に入っている（H_means から softmax 済）
    w = diagnostics.get("w", None)
    if w is None:
        # フォールバック：H_means から作る
        H_means = diagnostics.get("H_means", None)
        if H_means is None:
            raise RuntimeError("diagnostics lacks both 'w' and 'H_means'.")
        w = torch.softmax(-0.5 * H_means.to(torch.float64), dim=0)
    w = w.to(device=sampler.device, dtype=torch.float64)
    w = w / (w.sum() + 1e-300)

    # RNG
    gen = torch.Generator(device=sampler.device)
    gen.manual_seed(int(seed))

    # レプリカ選択（重み付き）
    ridx = torch.multinomial(w, num_samples=M, replacement=True, generator=gen)  # (M,)
    # 各レプリカ内の時刻（post-burnin 範囲で一様）
    tidx = torch.randint(low=adapt_end, high=Lp1, size=(M,), device=sampler.device, generator=gen)  # (M,)

    # n 抽出 → A = n*D
    n_samp = n_samples[ridx, tidx, :]  # (M, K)
    A = n_samp.to(torch.float64) * D_w[None, :]  # (M, K)

    # ROI mask（omega は K 長）
    m_roi = (omega >= wmin) & (omega <= wmax)
    if not torch.any(m_roi):
        raise ValueError(f"ROI {roi} has no omega grid points. Check omega/xlim.")

    omega_roi = omega[m_roi]                 # (Kroi,)
    A_roi = A[:, m_roi]                      # (M, Kroi)
    idx_local = torch.argmax(A_roi, dim=1)   # (M,)
    peak_pos = _refine_peak_positions_weighted3(omega_roi, A_roi, idx_local)  # (M,)
    return peak_pos

def _compute_peak_pos_thin_from_sampler(
    sampler: MaxEnt,
    diagnostics: dict,
    roi: tuple[float, float],
    max_keep_per_replica: int = 2000,
) -> torch.Tensor:
    """
    sampler.n_samples から、各レプリカ(=各alpha)ごとに post-burnin 区間を thinning して
    A = n * D_w を作り、ROI 内 argmax のピーク位置 ω_peak を返す（α-condition）。

    返り値: peak_pos_thin shape (R, M)  ※M は thinning 後のサンプル数
    また、diagnostics に thinning 情報を保存する:
      - peak_thin_start, peak_thin_stride, peak_thin_indices
    """
    wmin, wmax = float(roi[0]), float(roi[1])
    if wmax <= wmin:
        raise ValueError(f"roi must satisfy roi[0] < roi[1], got {roi}")

    omega = sampler.omega.to(dtype=torch.float64)
    D_w = sampler.D_w.to(dtype=torch.float64)
    n_samples = sampler.n_samples  # (R, Lp1, K)
    if n_samples is None:
        raise RuntimeError("sampler.n_samples is None. run_replica_exchange did not store samples?")

    R, Lp1, K = n_samples.shape

    adapt_end = int(diagnostics.get("adapt_end", int(0.8 * (Lp1 - 1))))
    adapt_end = max(0, min(adapt_end, Lp1 - 1))

    post_len = int(Lp1 - adapt_end)
    if post_len <= 0:
        raise RuntimeError("No post-burnin samples available for peak-pos thinning.")

    # thinning stride so that we keep at most max_keep_per_replica points per replica
    thin_stride = int(math.ceil(post_len / max_keep_per_replica))
    thin_stride = max(1, thin_stride)

    thin_idx = torch.arange(adapt_end, Lp1, thin_stride, device=sampler.device)  # (M,)
    M = int(thin_idx.numel())
    if M <= 0:
        raise RuntimeError("Thinning produced zero samples. Check adapt_end/stride.")

    # ROI mask
    m_roi = (omega >= wmin) & (omega <= wmax)
    if not torch.any(m_roi):
        raise ValueError(f"ROI {roi} has no omega grid points. Check omega/xlim.")

    omega_roi = omega[m_roi]  # (Kroi,)

    # gather samples: (R, M, K)
    n_thin = n_samples.index_select(1, thin_idx)  # (R, M, K)
    A_thin = n_thin.to(torch.float64) * D_w[None, None, :]  # (R, M, K)
    A_roi = A_thin[:, :, m_roi]  # (R, M, Kroi)

    idx_local = torch.argmax(A_roi, dim=2)  # (R, M)
    peak_pos_thin = _refine_peak_positions_weighted3(omega_roi, A_roi, idx_local)  # (R, M)

    # save thinning metadata into diagnostics (so result.npz alone is enough)
    diagnostics["peak_thin_start"] = int(adapt_end)
    diagnostics["peak_thin_stride"] = int(thin_stride)
    diagnostics["peak_thin_indices"] = thin_idx.detach().cpu().numpy().astype(np.int64)

    return peak_pos_thin
