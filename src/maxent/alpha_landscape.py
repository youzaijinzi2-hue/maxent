# Retry drawing the alpha-landscape

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, LogFormatterMathtext

def _gaussian_kernel1d(sigma_bins: float) -> np.ndarray:
    s = float(sigma_bins)
    if s <= 0.0:
        return np.array([1.0], dtype=float)
    r = max(1, int(np.ceil(3.0 * s)))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (x / s) ** 2)
    k /= (k.sum() + 1e-300)
    return k

def _smooth_axis_edge(a: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    if kernel.size <= 1:
        return a
    r = kernel.size // 2
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    ap = np.pad(a, pad, mode="edge")
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis, ap)

def alpha_landscape(path, run_dir):
    res = np.load(path, allow_pickle=True)
    alphas = res["alphas"].astype(float)
    w_alpha = res["w_alpha"].astype(float)
    wpeak = res["peak_pos_thin"].astype(float)

    w_alpha = w_alpha / (w_alpha.sum() + 1e-300)

    w_min, w_max = -5.0, 5.0
    nbins_w = 240
    w_edges = np.linspace(w_min, w_max, nbins_w + 1)

    alpha_edges = np.empty(alphas.size + 1)
    alpha_edges[1:-1] = np.sqrt(alphas[:-1] * alphas[1:])
    alpha_edges[0] = alphas[0] * np.sqrt(alphas[0] / alphas[1])
    alpha_edges[-1] = alphas[-1] * np.sqrt(alphas[-1] / alphas[-2])

    H = np.zeros((alphas.size, nbins_w))
    for i in range(alphas.size):
        vals = wpeak[i]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            H[i], _ = np.histogram(vals, bins=w_edges)

    # Conditional density P(w_peak | alpha)
    row_sum = H.sum(axis=1, keepdims=True)
    Hc = np.divide(H, row_sum, out=np.zeros_like(H), where=row_sum > 0)

    # Reweight by P(alpha|G) for cleaner visual importance across rows.
    Hj = Hc * w_alpha[:, None]

    # Light smoothing to suppress binning artifacts.
    Hj = _smooth_axis_edge(Hj, _gaussian_kernel1d(1.2), axis=1)  # omega axis
    Hj = _smooth_axis_edge(Hj, _gaussian_kernel1d(0.7), axis=0)  # alpha axis

    # Normalize for display (0..1) with robust vmax.
    pos = Hj[Hj > 0]
    if pos.size > 0:
        vmax = np.quantile(pos, 0.995)
        vmax = vmax if vmax > 0 else pos.max()
    else:
        vmax = 1.0
    Hn = np.clip(Hj / (vmax + 1e-300), 0.0, 1.0)

    fig = plt.figure(figsize=(11, 4.2), dpi=150)
    gs = GridSpec(1, 2, width_ratios=[1.2, 7.0], wspace=0.15)

    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1], sharey=axL)

    axL.plot(w_alpha, alphas, lw=2)
    axL.set_xlabel(r"$P(\alpha\mid G)$")
    axL.set_ylabel(r"Regularization parameter $\alpha$")
    axL.set_yscale("log")
    axL.yaxis.set_major_locator(LogLocator(base=10))
    axL.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    axL.grid(True, which="both", alpha=0.25)

    mesh = axR.pcolormesh(w_edges, alpha_edges, Hn, shading="auto", cmap="viridis")
    axR.set_xlabel(r"Peak position $\omega_{\mathrm{peak}}$")
    axR.set_xlim(w_min, w_max)
    axR.set_yscale("log")
    axR.yaxis.set_major_locator(LogLocator(base=10))
    axR.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    axR.grid(True, which="both", alpha=0.25)

    cbar = fig.colorbar(mesh, ax=axR, fraction=0.045, pad=0.03)
    cbar.set_label(r"normalized density")

    fig.text(
        0.5, -0.02,
        r"Alpha-landscape: $P(\alpha\mid G)$ and $P(\omega_{\mathrm{peak}}\mid \alpha)$",
        ha="center", va="top", fontsize=18, fontweight="bold"
    )

    plt.subplots_adjust(bottom=0.18)
    fig.savefig(run_dir / "alpha_landscape.png")
    plt.close(fig)
    # out = "/mnt/data/alpha_landscape_redrawn.png"
    # fig.savefig(out, bbox_inches="tight")
