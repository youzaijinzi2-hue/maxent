import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import matplotlib as mpl
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Any
from typing import Any, Optional

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)

def spectral_similarity_tests(x: np.ndarray, y: np.ndarray):
    if x.shape != y.shape:
        raise ValueError(f"x,y shape mismatch: {x.shape} vs {y.shape}")

    mse = np.mean((x - y) ** 2)

    # F-test (two-sided)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    if var_x == 0 or var_y == 0:
        f_stat = np.inf
        f_pvalue = 0.0
    else:
        if var_x >= var_y:
            f_stat = var_x / var_y
            dfn, dfd = x.size - 1, y.size - 1
        else:
            f_stat = var_y / var_x
            dfn, dfd = y.size - 1, x.size - 1
        p_one_tail = 1 - stats.f.cdf(f_stat, dfn, dfd)
        f_pvalue = min(1.0, 2 * p_one_tail)

    t_stat, t_pvalue = stats.ttest_rel(x, y)
    ks_stat, ks_pvalue = stats.ks_2samp(x, y)

    return dict(
        mse=float(mse),
        f_stat=float(f_stat), f_pvalue=float(f_pvalue),
        t_stat=float(t_stat), t_pvalue=float(t_pvalue),
        ks_stat=float(ks_stat), ks_pvalue=float(ks_pvalue),
    )


def _as_float_scalar(x: Any, *, reduce: str = "auto") -> float:
    """
    Convert x to a Python float for display.
    - If x is scalar or size-1 array -> float(x)
    - If x is array with size>1:
        - reduce="max" -> float(np.max(x))
        - reduce="min" -> float(np.min(x))
        - reduce="mean" -> float(np.mean(x))
        - reduce="auto" -> if name implies max (caller chooses) else error-safe max
    """
    arr = np.asarray(x)
    if arr.shape == () or arr.size == 1:
        return float(arr.reshape(()))
    if reduce in ("auto", "max"):
        return float(np.max(arr))
    if reduce == "min":
        return float(np.min(arr))
    if reduce == "mean":
        return float(np.mean(arr))
    # fallback
    return float(np.max(arr))

def _rhat_summary(x: Any) -> dict[str, float]:
    """
    rhat がスカラー or 配列（レプリカ毎）どちらでも安全に要約する。
    - scalar: max=mean=median=q90=その値
    - array : max/mean/median/q90 を返す
    """
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        return {"max": float("nan"), "mean": float("nan"), "median": float("nan"), "q90": float("nan")}
    if arr.size == 1:
        v = float(arr.reshape(()))
        return {"max": v, "mean": v, "median": v, "q90": v}
    return {
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q90": float(np.quantile(arr, 0.90)),
    }

def load_npz_estimates(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)

    required = ["omega", "A_est", "A_low", "A_high", "mcse"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"{npz_path} に必須キー {missing} がありません。存在キー: {z.files}")

    omega  = z["omega"].astype(float)
    A_est  = z["A_est"].astype(float)
    A_low  = z["A_low"].astype(float)
    A_high = z["A_high"].astype(float)
    mcse   = z["mcse"].astype(float)

    extras = {}
    # --- まず「レプリカ毎 rhat」を優先して読む（キー名は複数候補に対応） ---
    for rk in ["rhat", "r_hat", "rhat_per_replica", "r_hat_per_replica"]:
        if rk in z.files:
            extras["rhat"] = np.asarray(z[rk], dtype=float)
            break

    # --- 既存のキーも読む（スカラーなら float、配列なら ndarray のまま） ---
    for k in ["rhat_max", "rhat_mean", "elapsed_sec"]:
        if k not in z.files:
            continue
        arr = np.asarray(z[k])
        if arr.shape == () or arr.size == 1:
            extras[k] = float(arr.reshape(()))
        else:
            extras[k] = arr

    # optional: peak positions (precomputed) or raw samples
    peak_pos_roi = z["peak_pos_roi"].astype(float) if "peak_pos_roi" in z.files else None
    A_samples = z["A_samples"].astype(float) if "A_samples" in z.files else None

    return omega, A_est, A_low, A_high, mcse, extras, peak_pos_roi, A_samples

def _peak_positions_from_samples(omega: np.ndarray,
                                 A_samples: np.ndarray,
                                 roi: tuple[float, float]) -> np.ndarray:
    """
    ROI内 argmax により、各サンプルのピーク位置を返す。
    A_samples shape: (Nsamples, Nomega)
    """
    w0, w1 = float(roi[0]), float(roi[1])
    if w1 <= w0:
        raise ValueError(f"roi must satisfy roi[0] < roi[1], got {roi}")
    m = (omega >= w0) & (omega <= w1)
    if not np.any(m):
        raise ValueError(f"ROI {roi} に omega グリッド点がありません。xlim/omega を確認してください。")
    A_roi = A_samples[:, m]  # (Nsamples, Nroi)
    idx_local = np.argmax(A_roi, axis=1)  # (Nsamples,)
    omega_roi = omega[m]
    return omega_roi[idx_local]

def plotting(npz_path="result.npz",
         D_true=None,              # 真値を外部から渡す
         D_default=None,            # ★default model を外部から渡す（omegaと同じグリッド）
         xlim=(-10, 30),
         out_pdf="spectrum.pdf",
         normalize_mode="max",      # "max" or None
         trust_factor=1.0,   # backward compatibility only (no longer used)
         roi: Optional[tuple[float, float]] = None,   # ★ peak ROI [wmin, wmax]
         peak_hist: bool = True,                      # ★ ROI peak histogram
         peak_bins: int = 20,                         # ★ histogram bins
         ):
    
    mpl.rcParams['font.family']   = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams["font.size"] = 14                 # 文字の大きさ

    # カラーパレット例
    colors = {
        'mean':   '#56B4E9',  # blue
        'ci':     '#E69F00',  # orange
        'data':   '#000000',
        'default':"#C60000",  # red
        'hist': "#D55E00",   # 濃いオレンジ
        'hist_edge': "#B97700",
        }
    omega, A_est, A_low, A_high, mcse, extras, peak_pos_roi, A_samples = load_npz_estimates(npz_path)
    # 範囲カット
    m = (omega >= xlim[0]) & (omega <= xlim[1])
    omega_sub  = omega[m]
    A_est_sub  = A_est[m]
    A_low_sub  = A_low[m]
    A_high_sub = A_high[m]
    mcse_sub   = mcse[m]

    # ===== スケール基準を決める =====
    # normalize_mode == "max" のとき:
    #   - D_true があれば D_true の max を基準にする（推奨）
    #   - D_true が無ければ従来通り A_est の max を基準にする（フォールバック）
    if normalize_mode == "max":
        if D_true is not None:
            D_true = to_numpy(D_true)
            if D_true.shape != omega.shape:
                raise ValueError(
                    f"D_true shape {D_true.shape} が omega shape {omega.shape} と一致しません。\n"
                    "→ D_true は result.npz の omega と同じグリッド/長さで用意してください。"
                )
            D_true_sub = D_true[m]
            scale = np.max(D_true_sub) if np.max(D_true_sub) != 0 else 1.0
        else:
            scale = np.max(A_est_sub) if np.max(A_est_sub) != 0 else 1.0
    else:
        scale = 1.0

    # 推定側の正規化（基準=scale）
    if normalize_mode == "max":
        A_est_n  = A_est_sub  / scale
        A_low_n  = A_low_sub  / scale
        A_high_n = A_high_sub / scale
    else:
        A_est_n, A_low_n, A_high_n = A_est_sub, A_low_sub, A_high_sub

    # 真値が与えられた場合のみ比較
    stats_result = None
    residuals = None
    D_true_n = None
    D_default_n = None

    if D_true is not None:
        # ここに来た時点で D_true_sub は既に作ってある（normalize_mode=="max" のとき）
        # normalize_mode!= "max" の場合はここで作る
        if normalize_mode != "max":
            D_true = to_numpy(D_true)

            if D_true.shape != omega.shape:
                raise ValueError(
                    f"D_true shape {D_true.shape} が omega shape {omega.shape} と一致しません。\n"
                    "→ D_true は result.npz の omega と同じグリッド/長さで用意してください。"
                )
            D_true_sub = D_true[m]

        D_true_n = (D_true_sub / scale) if normalize_mode == "max" else D_true_sub
        stats_result = spectral_similarity_tests(D_true_n, A_est_n)
        residuals = (D_true_n - A_est_n)

    # default model（デフォルト）を描画したい場合
    if D_default is not None:
        D_default = to_numpy(D_default)
        if D_default.shape != omega.shape:
            raise ValueError(
                f"D_default shape {D_default.shape} が omega shape {omega.shape} と一致しません。\n"
                "→ D_default は result.npz の omega と同じグリッド/長さで用意してください。"
            )
        D_default_sub = D_default[m]
        D_default_n = (D_default_sub / scale) if normalize_mode == "max" else D_default_sub

    # ===== plot =====
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(6.0, 4.8),
        gridspec_kw={"height_ratios": [3, 1]}, dpi=300
    )

    # CI 幅
    ci_width = (A_high_n - A_low_n)

    # MCSE も同じ正規化に合わせる

    # Default line is drawn above True markers via zorder.
    if D_default_n is not None:
        ax1.plot(
            omega_sub, D_default_n,
            lw=1.2,
            label="Default", color=colors['default'], alpha=0.3, zorder=5
        )

    ax1.plot(omega_sub, A_est_n, lw=1.5, label="Estimate",  color=colors['mean'])
    ax1.fill_between(omega_sub, A_low_n, A_high_n, alpha=0.3, label="68% CI",  color=colors['ci'])

    if D_true_n is not None:
        ax1.plot(omega_sub, D_true_n, "^", ms=2, label="True", color=colors['data'], zorder=4)

    ax1.grid(which="major", linestyle="--", alpha=0.3)
    ax1.minorticks_on()
    ax1.grid(which="minor", linestyle=":", alpha=0.15)
    ax1.set_ylabel(r"Intensity $\, [\mathrm{arb.\ units}]$")
    ymin, ymax = A_est_n.min(), A_est_n.max()
    pad_max = 0.8 * max(abs(ymax), 1e-12)
    pad_min = 0.1
    ax1.set_ylim(ymin - pad_min, ymax + pad_max)
    # print("pad_min", pad_min)
    ax1.legend(fontsize=10)

    # ===== ROI peak histogram (baseline overlay) =====
    if peak_hist and (roi is not None):
        peak_pos = None
        if A_samples is not None:
            if A_samples.ndim == 2 and A_samples.shape[1] == omega.shape[0]:
                try:
                    peak_pos = _peak_positions_from_samples(omega, A_samples, roi)
                except Exception as e:
                    print(f"[warn] peak_pos from A_samples failed: {e}. skip peak histogram.")
            else:
                print("[warn] A_samples shape mismatch. skip peak histogram.")
        elif peak_pos_roi is not None:
            peak_pos = np.asarray(peak_pos_roi, dtype=float)
        else:
            print("[info] peak histogram requested but neither A_samples nor peak_pos_roi found in npz. skip.")

        if peak_pos is not None and peak_pos.size > 0:
            # --- histogram in ROI (x along omega) ---
            w0, w1 = float(roi[0]), float(roi[1])
            counts, edges = np.histogram(peak_pos, bins=int(peak_bins), range=(w0, w1))
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths  = np.diff(edges)

            # --- map counts -> small height near the bottom of ax1 ---
            y0, y1 = ax1.get_ylim()
            yr = y1 - y0

            # margin from bottom and maximum histogram height as a fraction of plot height
            base_margin = 0.02 * yr
            hist_height = 0.14 * yr  # 14% of y-range; tune this

            cmax = counts.max() if counts.max() > 0 else 1.0
            heights = (counts / cmax) * hist_height

            # draw bars behind the spectrum
            ax1.bar(
                centers,
                heights,
                width=widths,
                bottom=y0 + base_margin,
                align="center",
                color=colors["hist"],
                alpha=0.55,
                linewidth=0.3,
                edgecolor=colors["hist_edge"],
                zorder=0.5,   # behind lines
            )

            # optional: show ROI bounds (thin dotted lines)
            ax1.axvline(w0, color="gray", lw=0.8, ls=":")
            ax1.axvline(w1, color="gray", lw=0.8, ls=":")
            
            show_peak_quantiles = False  # ← ここだけ True/False で制御
            if show_peak_quantiles:
                # optional: annotate peak quantiles (small markers on baseline)
                q16, q50, q84 = np.percentile(peak_pos, [16, 50, 84])
                y_base = y0 + base_margin + 0.02 * yr
                ax1.plot([q16, q84], [y_base, y_base], color="k", lw=1.0, zorder=1.0)
                ax1.plot([q50], [y_base], marker="o", ms=3, color="k", zorder=1.0)

            # (optional) ROI label
            ax1.text(
                0.98, 0.02, f"ROI=[{w0:g},{w1:g}]",
                transform=ax1.transAxes, fontsize=8,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="gray"),
            )
    # 上段注釈
    lines = []
    if stats_result is not None:
        lines.append(rf"$\sqrt{{\mathrm{{MSE}}}} = {math.sqrt(stats_result['mse']):.3f}\,[arb.]$")
        lines.append(rf"$t={stats_result['t_stat']:.3g},\;p={stats_result['t_pvalue']:.3g}$")
    lines.append(f"Mean MCSE = {mcse_sub.mean():.4f} [arb.]")
    # --- R-hat 表示：レプリカ毎配列なら median / q90 を主に表示し、max は参考値として併記 ---
    # if "rhat" in extras:
    #     s = _rhat_summary(extras["rhat"])
    #     lines.append(rf"$\hat R_{{\mathrm{{med}}}} = {s['median']:.4f}$")
    #     lines.append(rf"$\hat R_{{0.90}} = {s['q90']:.4f}$")
    #     lines.append(rf"$\hat R_{{\max}} = {s['max']:.4f}$")
    # elif "rhat_max" in extras:
    #     # 互換：rhat_max がスカラーで保存されている古い形式
    #     rhat_max = _as_float_scalar(extras["rhat_max"], reduce="max")
    #     lines.append(rf"$\hat R_{{\max}} = {rhat_max:.4f}$")
    # NOTE:
    # R-hat (Gelman-Rubin) is a convergence "gate" rather than a plot annotation metric.
    # In replica/tempered settings, different replicas may target different distributions,
    # so showing R-hat in the main figure can be misleading. Keep it out of the figure.
    #
    # If you still want visibility, consider logging it to stdout or saving to run_meta.json
    # instead of drawing it on the spectrum plot.

    ax1.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax1.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    # 下段
    ax2.axhline(0, color="gray", lw=1, linestyle="--")
    if residuals is not None:
        ax2.plot(omega_sub, residuals, ".", ms=4)
        ymin, ymax = residuals.min(), residuals.max()
        pad = 0.1 * max(abs(ymin), abs(ymax), 1e-12)
        ax2.set_ylim(ymin - pad, ymax + pad)

        ax2.set_ylabel("Residual\n[arb. units]")
    else:
        # 真値が無いなら mcse を表示（代替）
        ax2.plot(omega_sub, mcse_sub, ".", ms=4)
        ax2.set_ylabel("MCSE\n[arb. units]")

    ax2.grid(which="major", linestyle="--", alpha=0.3)
    ax2.minorticks_on()
    ax2.grid(which="minor", linestyle=":", alpha=0.15)
    ax2.set_xlabel(r"$\omega \,[\mathrm{eV}]$")

    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax2.set_title("Spectrum Fit and Residuals", fontsize=14, pad=10, y=-1)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[saved] {out_pdf}")

    if stats_result is None:
        print("[info] D_true が渡されていないため、t-test/KS/residual は未計算です。")
    else:
        for k, v in stats_result.items():
            print(f"{k}: {v:.6g}")


if __name__ == "__main__":
    # 例：
    # main("result.npz", D_true=D_norm_sub)
    plotting(r"C:\Users\knkyy\MaxEntropyMethod\runs_for_paper\result.npz", D_true=None, roi=(-5, 5), peak_hist=True)
