from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ConvergenceData:
    run_dir: Path
    result_npz: Path
    alphas: np.ndarray | None
    h_samples_thin: np.ndarray | None
    h_thin_indices: np.ndarray | None
    h_thin_start: int
    rhat_max: float
    rhat_mean: float
    rhat_per_replica_max: np.ndarray | None
    rhat_per_replica_mean: np.ndarray | None
    ess_per_chain: np.ndarray | None
    total_ess: float
    ess_per_sec: float
    tau_ints: np.ndarray | None
    mcse_mean: float
    mcse_max: float
    mcse_threshold: float
    n_steps: int
    burnin: int
    adapt_end: int
    hot_start: int
    prob_adapt_metric: str
    prob_history_steps: np.ndarray | None
    move_prob_pca_history: np.ndarray | None
    move_prob_ws_history: np.ndarray | None
    move_prob_cg_history: np.ndarray | None
    prob_utility_obs_pca_history: np.ndarray | None
    prob_utility_obs_ws_history: np.ndarray | None
    prob_utility_obs_cg_history: np.ndarray | None
    prob_utility_est_pca_history: np.ndarray | None
    prob_utility_est_ws_history: np.ndarray | None
    prob_utility_est_cg_history: np.ndarray | None


def _to_scalar(x: Any, default: float = float("nan")) -> float:
    if x is None:
        return default
    try:
        arr = np.asarray(x)
        if arr.shape == () or arr.size == 1:
            return float(arr.reshape(()))
    except Exception:
        return default
    return default


def _as_1d_float(z: Any, key: str) -> np.ndarray | None:
    if key not in z.files:
        return None
    try:
        arr = np.asarray(z[key], dtype=float).reshape(-1)
        return arr
    except Exception:
        return None


def _as_str(z: Any, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    try:
        arr = np.asarray(z[key])
        if arr.shape == ():
            return str(arr.item())
    except Exception:
        pass
    try:
        return str(z[key])
    except Exception:
        return default


def resolve_result_npz(path: str | Path) -> tuple[Path, Path]:
    """
    Resolve a user path into (run_dir, result_npz_path).
    Accepts:
      - run directory containing artifacts/result.npz
      - direct path to result.npz
    """
    p = Path(path).expanduser().resolve()
    if p.is_file():
        if p.name != "result.npz":
            raise FileNotFoundError(f"Expected result.npz file, got: {p}")
        run_dir = p.parent.parent if p.parent.name == "artifacts" else p.parent
        return run_dir, p

    if p.is_dir():
        cand = p / "artifacts" / "result.npz"
        if cand.exists():
            return p, cand
        raise FileNotFoundError(f"result.npz not found under: {p}")

    raise FileNotFoundError(f"Path not found: {p}")


def load_convergence_data(path: str | Path) -> ConvergenceData:
    run_dir, npz_path = resolve_result_npz(path)
    z = np.load(npz_path, allow_pickle=True)

    n_steps_key = "n_steps" if "n_steps" in z.files else ("n_step" if "n_step" in z.files else None)

    h_samples_thin = None
    if "H_samples_thin" in z.files:
        try:
            h_arr = np.asarray(z["H_samples_thin"], dtype=float)
            if h_arr.ndim == 2:
                h_samples_thin = h_arr
        except Exception:
            h_samples_thin = None

    data = ConvergenceData(
        run_dir=run_dir,
        result_npz=npz_path,
        alphas=_as_1d_float(z, "alphas"),
        h_samples_thin=h_samples_thin,
        h_thin_indices=_as_1d_float(z, "H_thin_indices"),
        h_thin_start=int(_to_scalar(z["H_thin_start"], default=0)) if "H_thin_start" in z.files else 0,
        rhat_max=_to_scalar(z["rhat_max"], default=np.nan) if "rhat_max" in z.files else np.nan,
        rhat_mean=_to_scalar(z["rhat_mean"], default=np.nan) if "rhat_mean" in z.files else np.nan,
        rhat_per_replica_max=_as_1d_float(z, "rhat_per_replica_max"),
        rhat_per_replica_mean=_as_1d_float(z, "rhat_per_replica_mean"),
        ess_per_chain=_as_1d_float(z, "ess_per_chain"),
        total_ess=_to_scalar(z["total_ess"], default=np.nan) if "total_ess" in z.files else np.nan,
        ess_per_sec=_to_scalar(z["ess_per_sec"], default=np.nan) if "ess_per_sec" in z.files else np.nan,
        tau_ints=_as_1d_float(z, "tau_ints"),
        mcse_mean=_to_scalar(z["mcse_mean"], default=np.nan) if "mcse_mean" in z.files else np.nan,
        mcse_max=_to_scalar(z["mcse_max"], default=np.nan) if "mcse_max" in z.files else np.nan,
        mcse_threshold=_to_scalar(z["mcse_threshold"], default=np.nan) if "mcse_threshold" in z.files else np.nan,
        n_steps=int(_to_scalar(z[n_steps_key], default=-1)) if n_steps_key is not None else -1,
        burnin=int(_to_scalar(z["burnin"], default=-1)) if "burnin" in z.files else -1,
        adapt_end=int(_to_scalar(z["adapt_end"], default=-1)) if "adapt_end" in z.files else -1,
        hot_start=int(_to_scalar(z["hot_start"], default=-1)) if "hot_start" in z.files else -1,
        prob_adapt_metric=_as_str(z, "prob_adapt_metric", default=""),
        prob_history_steps=_as_1d_float(z, "prob_history_steps"),
        move_prob_pca_history=_as_1d_float(z, "move_prob_pca_history"),
        move_prob_ws_history=_as_1d_float(z, "move_prob_ws_history"),
        move_prob_cg_history=_as_1d_float(z, "move_prob_cg_history"),
        prob_utility_obs_pca_history=_as_1d_float(z, "prob_utility_obs_pca_history"),
        prob_utility_obs_ws_history=_as_1d_float(z, "prob_utility_obs_ws_history"),
        prob_utility_obs_cg_history=_as_1d_float(z, "prob_utility_obs_cg_history"),
        prob_utility_est_pca_history=_as_1d_float(z, "prob_utility_est_pca_history"),
        prob_utility_est_ws_history=_as_1d_float(z, "prob_utility_est_ws_history"),
        prob_utility_est_cg_history=_as_1d_float(z, "prob_utility_est_cg_history"),
    )

    has_any_diag = any(
        [
            np.isfinite(data.rhat_max),
            np.isfinite(data.rhat_mean),
            data.rhat_per_replica_max is not None,
            data.ess_per_chain is not None,
            data.tau_ints is not None,
            data.h_samples_thin is not None,
            data.prob_history_steps is not None,
        ]
    )
    if not has_any_diag:
        raise ValueError(
            "No convergence diagnostics found in result.npz. "
            "Expected at least one of: rhat/ess/tau_ints/H_samples_thin/prob history."
        )
    return data


def _fmt(v: float, nd: int = 4) -> str:
    if not np.isfinite(v):
        return "N/A"
    return f"{v:.{nd}g}"


def _fmt_i(v: int) -> str:
    return str(int(v)) if int(v) >= 0 else "N/A"


def _select_indices(n: int, k: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)
    return np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))


def plot_convergence_data(
    data: ConvergenceData,
    *,
    out_path: str | Path,
    rhat_threshold: float = 1.05,
    ess_min_per_chain: float = 50.0,
    max_replicas: int = 12,
    title: str | None = None,
    dpi: int = 180,
) -> Path:
    out_path = Path(out_path).expanduser().resolve()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 8.5),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.92, wspace=0.30, hspace=0.28)
    ax_trace, ax_rhat, ax_ess, ax_txt = axes.flat

    # Panel 1: move-prob adaptation (preferred) or energy trace fallback.
    has_prob_hist = (
        data.prob_history_steps is not None
        and data.move_prob_pca_history is not None
        and data.move_prob_ws_history is not None
        and data.move_prob_cg_history is not None
        and data.prob_history_steps.size > 0
    )
    if has_prob_hist:
        x_step = np.asarray(data.prob_history_steps, dtype=float).reshape(-1)
        p_pca = np.asarray(data.move_prob_pca_history, dtype=float).reshape(-1)
        p_ws = np.asarray(data.move_prob_ws_history, dtype=float).reshape(-1)
        p_cg = np.asarray(data.move_prob_cg_history, dtype=float).reshape(-1)
        n_prob = int(min(x_step.size, p_pca.size, p_ws.size, p_cg.size))
        if n_prob > 0:
            x_step = x_step[:n_prob]
            p_pca = p_pca[:n_prob]
            p_ws = p_ws[:n_prob]
            p_cg = p_cg[:n_prob]
            ax_trace.plot(x_step, p_pca, lw=1.8, color="#4C78A8", label="p_pca")
            ax_trace.plot(x_step, p_ws, lw=1.8, color="#F58518", label="p_ws")
            ax_trace.plot(x_step, p_cg, lw=1.8, color="#54A24B", label="p_cg")
            ax_trace.set_xlabel("MCMC step")
            ax_trace.set_ylabel("Move probability p_m(t)")
            ax_trace.set_ylim(0.0, 1.0)
            ax_trace.grid(True, linestyle=":", alpha=0.4)

            util_label = data.prob_adapt_metric or "utility"
            if util_label == "esjd_per_sec":
                util_label = "ESJD/sec"
            util_obs_available = (
                data.prob_utility_obs_pca_history is not None
                and data.prob_utility_obs_ws_history is not None
                and data.prob_utility_obs_cg_history is not None
            )
            util_est_available = (
                data.prob_utility_est_pca_history is not None
                and data.prob_utility_est_ws_history is not None
                and data.prob_utility_est_cg_history is not None
            )
            ax_util = ax_trace.twinx()
            util_plotted = False
            if util_obs_available:
                uo_pca = np.asarray(data.prob_utility_obs_pca_history, dtype=float).reshape(-1)[:n_prob]
                uo_ws = np.asarray(data.prob_utility_obs_ws_history, dtype=float).reshape(-1)[:n_prob]
                uo_cg = np.asarray(data.prob_utility_obs_cg_history, dtype=float).reshape(-1)[:n_prob]
                ax_util.plot(x_step, uo_pca, ls="--", lw=1.2, color="#4C78A8", alpha=0.65, label=f"{util_label}_pca(obs)")
                ax_util.plot(x_step, uo_ws, ls="--", lw=1.2, color="#F58518", alpha=0.65, label=f"{util_label}_ws(obs)")
                ax_util.plot(x_step, uo_cg, ls="--", lw=1.2, color="#54A24B", alpha=0.65, label=f"{util_label}_cg(obs)")
                util_plotted = True
            if util_est_available:
                ue_pca = np.asarray(data.prob_utility_est_pca_history, dtype=float).reshape(-1)[:n_prob]
                ue_ws = np.asarray(data.prob_utility_est_ws_history, dtype=float).reshape(-1)[:n_prob]
                ue_cg = np.asarray(data.prob_utility_est_cg_history, dtype=float).reshape(-1)[:n_prob]
                ax_util.plot(x_step, ue_pca, ls="-.", lw=1.4, color="#4C78A8", alpha=0.95, label=f"{util_label}_pca(est)")
                ax_util.plot(x_step, ue_ws, ls="-.", lw=1.4, color="#F58518", alpha=0.95, label=f"{util_label}_ws(est)")
                ax_util.plot(x_step, ue_cg, ls="-.", lw=1.4, color="#54A24B", alpha=0.95, label=f"{util_label}_cg(est)")
                util_plotted = True
            if util_plotted:
                ax_util.set_ylabel(f"{util_label}_m(t)")
                h1, l1 = ax_trace.get_legend_handles_labels()
                h2, l2 = ax_util.get_legend_handles_labels()
                ax_trace.legend(h1 + h2, l1 + l2, fontsize=7, loc="best", ncol=2)
            else:
                ax_trace.legend(fontsize=8, loc="best")

            if x_step.size > 0:
                x_min = float(np.nanmin(x_step))
                x_max = float(np.nanmax(x_step))
                phase_defs: list[tuple[str, int, str]] = [
                    ("burnin", int(data.burnin), "tab:gray"),
                    ("hot_start", int(data.hot_start), "tab:orange"),
                    ("adapt_end", int(data.adapt_end), "tab:red"),
                ]
                for nm, step, color in phase_defs:
                    if step >= 0 and x_min <= float(step) <= x_max:
                        ax_trace.axvline(float(step), color=color, ls=":", lw=1.1, alpha=0.85)
            if data.h_samples_thin is not None:
                try:
                    h = np.asarray(data.h_samples_thin, dtype=float)
                    if h.ndim == 2 and h.shape[1] > 0:
                        n_rep, n_t = h.shape
                        x_h = np.arange(n_t, dtype=float)
                        if data.h_thin_indices is not None and data.h_thin_indices.size == n_t:
                            x_h = np.asarray(data.h_thin_indices, dtype=float)
                        ax_inset = ax_trace.inset_axes([0.57, 0.56, 0.40, 0.40])
                        sel_h = _select_indices(n_rep, min(max_replicas, 6))
                        for ridx in sel_h:
                            ax_inset.plot(x_h, h[ridx], lw=0.7, alpha=0.45, color="#666666")
                        h_mean = np.nanmean(h, axis=0)
                        ax_inset.plot(x_h, h_mean, color="black", lw=1.2)
                        if x_h.size > 0:
                            xh_min = float(np.nanmin(x_h))
                            xh_max = float(np.nanmax(x_h))
                            phase_defs_in: list[tuple[str, int, str]] = [
                                ("burnin", int(data.burnin), "tab:gray"),
                                ("hot_start", int(data.hot_start), "tab:orange"),
                                ("adapt_end", int(data.adapt_end), "tab:red"),
                            ]
                            for _, step, color in phase_defs_in:
                                if step >= 0 and xh_min <= float(step) <= xh_max:
                                    ax_inset.axvline(float(step), color=color, ls="--", lw=0.8, alpha=0.8)
                        ax_inset.set_title("H trace", fontsize=8)
                        ax_inset.tick_params(axis="both", labelsize=7)
                        ax_inset.grid(True, linestyle=":", alpha=0.25)
                except Exception:
                    pass
            ax_trace.set_title("Move Adaptation: p_m(t) and utility_m(t)")
        else:
            has_prob_hist = False

    if not has_prob_hist:
        if data.h_samples_thin is not None:
            h = np.asarray(data.h_samples_thin, dtype=float)
            n_rep, n_t = h.shape
            x = np.arange(n_t, dtype=float)
            if data.h_thin_indices is not None and data.h_thin_indices.size == n_t:
                x = np.asarray(data.h_thin_indices, dtype=float)

            sel = _select_indices(n_rep, max_replicas)
            for ridx in sel:
                lbl = None
                if data.alphas is not None and data.alphas.size == n_rep and sel.size <= 8:
                    lbl = f"alpha={data.alphas[ridx]:.2e}"
                ax_trace.plot(x, h[ridx], lw=1.0, alpha=0.70, label=lbl)

            h_mean = np.nanmean(h, axis=0)
            mean_line, = ax_trace.plot(x, h_mean, color="black", lw=2.0, label="replica mean")

            phase_lines: list[tuple[Any, str]] = []
            if x.size > 0:
                x_min = float(np.nanmin(x))
                x_max = float(np.nanmax(x))
                phase_defs: list[tuple[str, int, str]] = [
                    ("burnin", int(data.burnin), "tab:gray"),
                    ("hot_start", int(data.hot_start), "tab:orange"),
                    ("adapt_end", int(data.adapt_end), "tab:red"),
                ]
                for nm, step, color in phase_defs:
                    if step >= 0 and x_min <= float(step) <= x_max:
                        ln = ax_trace.axvline(float(step), color=color, ls="--", lw=1.2, alpha=0.85, label=nm)
                        phase_lines.append((ln, nm))

            if int(data.h_thin_start) > 0:
                trace_scope = f"from step {int(data.h_thin_start)}"
            else:
                trace_scope = "full run"
            ax_trace.set_title(f"Energy Trace H[n] (thinned, {trace_scope})")
            ax_trace.set_xlabel("MCMC step")
            ax_trace.set_ylabel("H[n]")
            ax_trace.grid(True, linestyle=":", alpha=0.4)
            if sel.size <= 8:
                ax_trace.legend(fontsize=8, loc="best")
            elif phase_lines:
                handles = [mean_line] + [hnd for hnd, _ in phase_lines]
                labels = ["replica mean"] + [nm for _, nm in phase_lines]
                ax_trace.legend(handles, labels, fontsize=8, loc="best")
        else:
            ax_trace.text(
                0.5,
                0.5,
                "H_samples_thin not found.\nTrace panel is unavailable for this run.",
                ha="center",
                va="center",
                transform=ax_trace.transAxes,
            )
            ax_trace.set_title("Energy Trace H[n]")
            ax_trace.set_axis_off()

    # Panel 2: R-hat diagnostics
    def _series_with_x(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, bool]:
        yy = np.asarray(y, dtype=float).reshape(-1)
        xx = np.arange(yy.size, dtype=float)
        xlabel = "Replica index"
        use_log = False
        if data.alphas is not None and data.alphas.size == yy.size:
            aa = np.asarray(data.alphas, dtype=float).reshape(-1)
            if np.all(np.isfinite(aa)):
                order = np.argsort(aa)
                xx = aa[order]
                yy = yy[order]
                xlabel = "alpha"
                use_log = bool(np.all(aa > 0.0))
        return xx, yy, xlabel, use_log

    ax_rhat.axhline(rhat_threshold, color="tab:red", ls="--", lw=1.5, label=f"threshold={rhat_threshold:g}")
    plotted = False
    rhat_xlabel = "Replica index"
    rhat_log_x = False
    if data.rhat_per_replica_max is not None and data.rhat_per_replica_max.size > 0:
        rmax = np.asarray(data.rhat_per_replica_max, dtype=float)
        xx, yy, xlabel, use_log = _series_with_x(rmax)
        sel = _select_indices(yy.size, max_replicas)
        ax_rhat.plot(xx[sel], yy[sel], "o-", lw=1.5, ms=4, label="rhat_per_replica_max")
        if xlabel == "alpha":
            rhat_xlabel = "alpha"
            rhat_log_x = rhat_log_x or use_log
        plotted = True
    if data.rhat_per_replica_mean is not None and data.rhat_per_replica_mean.size > 0:
        rmean = np.asarray(data.rhat_per_replica_mean, dtype=float)
        xx, yy, xlabel, use_log = _series_with_x(rmean)
        sel = _select_indices(yy.size, max_replicas)
        ax_rhat.plot(xx[sel], yy[sel], "s-", lw=1.2, ms=4, label="rhat_per_replica_mean")
        if xlabel == "alpha":
            rhat_xlabel = "alpha"
            rhat_log_x = rhat_log_x or use_log
        plotted = True
    if np.isfinite(data.rhat_max):
        ax_rhat.axhline(data.rhat_max, color="tab:purple", ls=":", lw=1.2, label=f"global max={data.rhat_max:.4g}")
        plotted = True
    if plotted:
        ax_rhat.set_title("Split R-hat")
        ax_rhat.set_xlabel(rhat_xlabel)
        ax_rhat.set_ylabel("R-hat")
        if rhat_xlabel == "alpha" and rhat_log_x:
            ax_rhat.set_xscale("log")
        ax_rhat.grid(True, linestyle=":", alpha=0.4)
        ax_rhat.legend(fontsize=8, loc="best")
    else:
        ax_rhat.text(
            0.5,
            0.5,
            "R-hat diagnostics not found.",
            ha="center",
            va="center",
            transform=ax_rhat.transAxes,
        )
        ax_rhat.set_title("Split R-hat")
        ax_rhat.set_axis_off()

    # Panel 3: ESS and integrated autocorrelation time
    has_ess = data.ess_per_chain is not None and data.ess_per_chain.size > 0
    has_tau = data.tau_ints is not None and data.tau_ints.size > 0
    if has_ess:
        ess = np.asarray(data.ess_per_chain, dtype=float)
        x = np.arange(ess.size, dtype=int)
        sel = _select_indices(ess.size, max_replicas)
        ax_ess.bar(x[sel], ess[sel], alpha=0.85, color="#4C78A8", label="ESS/chain")
        ax_ess.axhline(ess_min_per_chain, color="tab:red", ls="--", lw=1.3, label=f"ESS target={ess_min_per_chain:g}")
        ax_ess.set_ylabel("ESS")
        ax_ess.set_xlabel("Replica index")
        ax_ess.grid(True, axis="y", linestyle=":", alpha=0.4)

        if has_tau and data.tau_ints.size == ess.size:
            tau = np.asarray(data.tau_ints, dtype=float)
            ax_tau = ax_ess.twinx()
            ax_tau.plot(x[sel], tau[sel], "o-", color="tab:orange", lw=1.3, ms=3, label="tau_int")
            ax_tau.set_ylabel("tau_int")
            h1, l1 = ax_ess.get_legend_handles_labels()
            h2, l2 = ax_tau.get_legend_handles_labels()
            ax_ess.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
        else:
            ax_ess.legend(fontsize=8, loc="best")
        ax_ess.set_title("Effective Sample Size (and tau_int)")
    else:
        ax_ess.text(
            0.5,
            0.5,
            "ESS/tau_int diagnostics not found.",
            ha="center",
            va="center",
            transform=ax_ess.transAxes,
        )
        ax_ess.set_title("ESS and tau_int")
        ax_ess.set_axis_off()

    # Panel 4: numeric summary
    ax_txt.set_axis_off()
    ax_txt.set_xlim(0.0, 1.0)
    ax_txt.set_ylim(0.0, 1.0)
    ok_rhat = bool(np.isfinite(data.rhat_max) and (data.rhat_max <= rhat_threshold))
    ok_ess = False
    if has_ess:
        ess_min = float(np.nanmin(np.asarray(data.ess_per_chain, dtype=float)))
        ok_ess = bool(np.isfinite(ess_min) and ess_min >= ess_min_per_chain)
    else:
        ess_min = float("nan")
    if np.isfinite(data.mcse_threshold):
        ok_mcse = bool(np.isfinite(data.mcse_max) and data.mcse_max <= data.mcse_threshold)
        mcse_target_txt = _fmt(data.mcse_threshold)
    else:
        ok_mcse = False
        mcse_target_txt = "N/A"

    checks: list[tuple[str, bool | None]] = [
        (f"R-hat max <= {rhat_threshold:g}", ok_rhat if np.isfinite(data.rhat_max) else None),
        (f"ESS/chain min >= {ess_min_per_chain:g}", ok_ess if has_ess else None),
        (f"MCSE max <= {mcse_target_txt}", ok_mcse if np.isfinite(data.mcse_threshold) else None),
    ]
    valid_checks = [v for _, v in checks if v is not None]
    overall_ok = all(valid_checks) if valid_checks else False

    try:
        result_display = str(data.result_npz.relative_to(data.run_dir))
    except Exception:
        result_display = str(data.result_npz.name)
    prob_updates = int(data.prob_history_steps.size) if data.prob_history_steps is not None else 0
    prob_metric_txt = data.prob_adapt_metric if data.prob_adapt_metric else "N/A"

    lines = [
        f"run: {data.run_dir.name}",
        f"result: {result_display}",
        "",
        f"rhat_max: {_fmt(data.rhat_max)}",
        f"rhat_mean: {_fmt(data.rhat_mean)}",
        f"ess_min_per_chain: {_fmt(ess_min)}",
        f"total_ess: {_fmt(data.total_ess)}",
        f"ess_per_sec: {_fmt(data.ess_per_sec)}",
        f"mcse_mean: {_fmt(data.mcse_mean)}",
        f"mcse_max: {_fmt(data.mcse_max)}",
        f"prob_metric: {prob_metric_txt}",
        f"prob_updates: {prob_updates}",
        "",
        (
            f"n_step={_fmt_i(data.n_steps)}, "
            f"burnin={_fmt_i(data.burnin)}, "
            f"hot_start={_fmt_i(data.hot_start)}, "
            f"adapt_end={_fmt_i(data.adapt_end)}"
        ),
        "",
        f"overall: {'PASS' if overall_ok else 'CHECK'}",
    ]

    def _wrap_line(line: str, width: int = 56) -> list[str]:
        if not line:
            return [""]
        return textwrap.wrap(
            line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [line]

    y = 0.98
    for ln in lines:
        for sub in _wrap_line(ln):
            ax_txt.text(
                0.02,
                y,
                sub,
                ha="left",
                va="top",
                transform=ax_txt.transAxes,
                fontsize=9,
                clip_on=True,
            )
            y -= 0.055 if sub else 0.03

    y -= 0.02
    for name, state in checks:
        if state is True:
            tag = "[OK]"
            color = "tab:green"
        elif state is False:
            tag = "[NG]"
            color = "tab:red"
        else:
            tag = "[N/A]"
            color = "tab:gray"
        for sub in _wrap_line(f"{tag} {name}", width=52):
            ax_txt.text(
                0.02,
                y,
                sub,
                ha="left",
                va="top",
                transform=ax_txt.transAxes,
                fontsize=9,
                color=color,
                clip_on=True,
            )
            y -= 0.065

    if title:
        fig.suptitle(title, fontsize=13)
    else:
        fig.suptitle("MCMC Convergence Diagnostics", fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=max(72, int(dpi)))
    plt.close(fig)
    return out_path


def summarize_convergence(data: ConvergenceData) -> dict[str, float]:
    ess_min = (
        float(np.nanmin(np.asarray(data.ess_per_chain, dtype=float)))
        if data.ess_per_chain is not None and data.ess_per_chain.size > 0
        else float("nan")
    )
    return {
        "rhat_max": float(data.rhat_max),
        "rhat_mean": float(data.rhat_mean),
        "ess_min_per_chain": float(ess_min),
        "total_ess": float(data.total_ess),
        "ess_per_sec": float(data.ess_per_sec),
        "mcse_mean": float(data.mcse_mean),
        "mcse_max": float(data.mcse_max),
        "mcse_threshold": float(data.mcse_threshold),
    }


def save_mcmc_convergence_figure(
    path: str | Path,
    *,
    out_path: str | Path | None = None,
    rhat_threshold: float = 1.05,
    ess_min_per_chain: float = 50.0,
    max_replicas: int = 12,
    title: str | None = None,
    dpi: int = 180,
) -> tuple[Path, dict[str, float]]:
    data = load_convergence_data(path)
    if out_path is None:
        out_path = data.run_dir / "mcmc_convergence.png"
    saved = plot_convergence_data(
        data,
        out_path=out_path,
        rhat_threshold=float(rhat_threshold),
        ess_min_per_chain=float(ess_min_per_chain),
        max_replicas=max(1, int(max_replicas)),
        title=title,
        dpi=int(dpi),
    )
    return saved, summarize_convergence(data)
