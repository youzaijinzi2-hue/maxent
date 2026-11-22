from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import matplotlib.pyplot as plt
import yaml


def _to_scalar(x: Any, default: Any = None) -> Any:
    if x is None:
        return default
    arr = np.asarray(x)
    if arr.shape == () or arr.size == 1:
        return arr.reshape(()).item()
    return default


def _resolve_result_npz(path: Path) -> tuple[Path, Path]:
    """
    Resolve a user path into (run_dir, result_npz_path).
    Accepts:
      - run directory containing artifacts/result.npz
      - direct path to result.npz
    """
    p = path.expanduser().resolve()
    if p.is_file():
        if p.name != "result.npz":
            raise FileNotFoundError(f"Expected result.npz file, got: {p}")
        run_dir = p.parent.parent
        if p.parent.name != "artifacts":
            run_dir = p.parent
        return run_dir, p

    if p.is_dir():
        cand = p / "artifacts" / "result.npz"
        if cand.exists():
            return p, cand
        raise FileNotFoundError(f"result.npz not found under: {p}")

    raise FileNotFoundError(f"Path not found: {p}")


def _default_label_from_config(run_dir: Path) -> str:
    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        return run_dir.name
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return run_dir.name

    dcfg = (cfg.get("default_model", {}) or {})
    base = str(dcfg.get("base", "unknown"))
    b = base.lower()
    if b in ("gaussian", "gauss", "normal"):
        g = dcfg.get("gaussian", {}) or {}
        mu = g.get("mu", g.get("mean", "?"))
        sig = g.get("sigma", g.get("std", "?"))
        return f"{run_dir.name} | {base} mu={mu},sig={sig}"
    if b in ("gaussian_mixture", "gaussian-mixture", "gmm", "mixture"):
        gm = dcfg.get("gaussian_mixture", {}) or {}
        comps = gm.get("components", [])
        n = len(comps) if isinstance(comps, list) else "?"
        return f"{run_dir.name} | {base} ncomp={n}"

    sim = dcfg.get("similar_to_Dw", dcfg.get("similar_to_dw", {})) or {}
    if bool(sim.get("enabled", False)):
        sig = sim.get("smooth_sigma", "?")
        mix = sim.get("mix_smoothed", "?")
        return f"{run_dir.name} | {base} sim(s={sig},mix={mix})"
    return f"{run_dir.name} | {base}"


@dataclass
class EvidenceEntry:
    label: str
    run_dir: Path
    result_npz: Path
    method: str
    logz: float
    alphas: np.ndarray | None
    w_alpha: np.ndarray | None
    omega: np.ndarray
    A_est: np.ndarray
    A_low: np.ndarray | None
    A_high: np.ndarray | None


def _load_entry(path: Path, label_override: str | None = None) -> EvidenceEntry:
    run_dir, npz_path = _resolve_result_npz(path)
    z = np.load(npz_path, allow_pickle=True)

    omega = np.asarray(z["omega"], dtype=float)
    A_est = np.asarray(z["A_est"], dtype=float)
    A_low = np.asarray(z["A_low"], dtype=float) if "A_low" in z.files else None
    A_high = np.asarray(z["A_high"], dtype=float) if "A_high" in z.files else None

    alphas = np.asarray(z["alphas"], dtype=float) if "alphas" in z.files else None
    w_alpha = np.asarray(z["w_alpha"], dtype=float) if "w_alpha" in z.files else None
    if w_alpha is not None:
        s = float(np.sum(w_alpha))
        if np.isfinite(s) and s > 0:
            w_alpha = w_alpha / s

    method = str(_to_scalar(z["alpha_weight_method"], default="unknown")) if "alpha_weight_method" in z.files else "unknown"
    # Prefer explicit key; then prefixed safety-net key.
    if "logZ_alpha" in z.files:
        logz = float(_to_scalar(z["logZ_alpha"], default=np.nan))
    elif "diag_logZ_alpha" in z.files:
        logz = float(_to_scalar(z["diag_logZ_alpha"], default=np.nan))
    else:
        logz = float("nan")

    label = label_override if label_override else _default_label_from_config(run_dir)
    return EvidenceEntry(
        label=label,
        run_dir=run_dir,
        result_npz=npz_path,
        method=method,
        logz=logz,
        alphas=alphas,
        w_alpha=w_alpha,
        omega=omega,
        A_est=A_est,
        A_low=A_low,
        A_high=A_high,
    )


def _stable_sort_entries(entries: Iterable[EvidenceEntry]) -> list[EvidenceEntry]:
    es = list(entries)
    # finite logZ first (descending), NaN at the end.
    return sorted(
        es,
        key=lambda e: (0 if np.isfinite(e.logz) else 1, -(e.logz if np.isfinite(e.logz) else -np.inf), e.label),
    )


def _draw_three_panel(
    entries: list[EvidenceEntry],
    *,
    out_path: Path,
    top_k: int,
    xlim: tuple[float, float] | None,
    title: str | None,
    show_ci: bool,
) -> None:
    if len(entries) == 0:
        raise ValueError("No entries to plot.")

    ordered = _stable_sort_entries(entries)
    finite = [e for e in ordered if np.isfinite(e.logz)]
    if len(finite) == 0:
        raise ValueError(
            "No finite logZ_alpha found. Re-run with updated code to save logZ_alpha."
        )

    z_best = max(e.logz for e in finite)
    dlogz = [e.logz - z_best if np.isfinite(e.logz) else np.nan for e in ordered]

    top = finite[: max(1, min(top_k, len(finite)))]

    fig, axes = plt.subplots(3, 1, figsize=(12, 13), constrained_layout=True)
    ax1, ax2, ax3 = axes

    # Panel 1: delta log evidence bars
    y = np.arange(len(ordered))
    bar_vals = np.array([v if np.isfinite(v) else 0.0 for v in dlogz], dtype=float)
    colors = ["#1f77b4" if i == 0 else "#4C78A8" for i in range(len(ordered))]
    ax1.barh(y, bar_vals, color=colors, alpha=0.85)
    ax1.axvline(0.0, color="black", linewidth=1.0)
    ax1.set_yticks(y)
    ax1.set_yticklabels([e.label for e in ordered], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel(r"$\Delta \log Z_\alpha = \log Z_\alpha - \max_m \log Z_\alpha$")
    ax1.set_title("Default-Model Ranking by Relative Evidence")
    ax1.grid(axis="x", linestyle=":", alpha=0.4)
    for i, v in enumerate(dlogz):
        txt = "N/A" if not np.isfinite(v) else f"{v:.3f} ({ordered[i].method})"
        x_text = 0.02 if not np.isfinite(v) else (v + 0.03 if v <= 0 else v)
        ax1.text(x_text, i, txt, va="center", ha="left", fontsize=8)

    # Panel 2: alpha posterior for top models
    has_any_alpha = False
    for e in top:
        if e.alphas is None or e.w_alpha is None:
            continue
        if len(e.alphas) != len(e.w_alpha):
            continue
        ax2.plot(e.alphas, e.w_alpha, linewidth=2.0, label=f"{e.label} ({e.method})")
        has_any_alpha = True
    if has_any_alpha:
        ax2.set_xscale("log")
        ax2.set_ylabel(r"$P(\alpha\mid G,m)$ (discrete weights)")
        ax2.set_title("Alpha Posterior (Top Models)")
        ax2.grid(True, which="both", linestyle=":", alpha=0.4)
        ax2.legend(fontsize=8, loc="best")
    else:
        ax2.text(
            0.5, 0.5,
            "No alpha posterior weights found in selected runs.",
            ha="center", va="center", transform=ax2.transAxes
        )
        ax2.set_axis_off()

    # Panel 3: spectra overlay for top models
    for e in top:
        ax3.plot(e.omega, e.A_est, linewidth=2.0, label=e.label)
        if show_ci and e.A_low is not None and e.A_high is not None and len(e.A_low) == len(e.A_est):
            ax3.fill_between(e.omega, e.A_low, e.A_high, alpha=0.15)
    ax3.set_title("Recovered Spectrum Overlay (Top Models)")
    ax3.set_xlabel(r"$\omega$")
    ax3.set_ylabel(r"$A(\omega)$")
    if xlim is not None:
        ax3.set_xlim(*xlim)
    ax3.grid(True, linestyle=":", alpha=0.4)
    ax3.legend(fontsize=8, loc="best")

    if title:
        fig.suptitle(title, fontsize=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Compare default-model evidence across runs and generate a 3-panel figure: "
            "Delta logZ bar chart, P(alpha|G,m), and A(omega) overlay."
        )
    )
    ap.add_argument(
        "paths",
        nargs="+",
        help="Run directories or direct paths to artifacts/result.npz (at least 2 recommended).",
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels corresponding to paths (same length as paths).",
    )
    ap.add_argument(
        "--out",
        default="default_evidence_comparison.png",
        help="Output figure path.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top models (by logZ_alpha) to show in panels 2 and 3.",
    )
    ap.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Optional x-range for spectrum panel.",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    ap.add_argument(
        "--show-ci",
        action="store_true",
        help="Show confidence-interval bands in the spectrum overlay panel (default: off).",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if args.labels is not None and len(args.labels) > 0 and len(args.labels) != len(args.paths):
        raise ValueError("--labels length must match number of paths.")

    entries: list[EvidenceEntry] = []
    for i, p in enumerate(args.paths):
        label = args.labels[i] if args.labels is not None and len(args.labels) > 0 else None
        try:
            entries.append(_load_entry(Path(p), label_override=label))
        except FileNotFoundError as e:
            print(f"[warn] skip path: {e}")

    if len(entries) == 0:
        raise ValueError("No valid runs/result.npz were found.")

    out_path = Path(args.out).expanduser().resolve()
    xlim = tuple(args.xlim) if args.xlim is not None else None
    _draw_three_panel(
        entries,
        out_path=out_path,
        top_k=max(1, int(args.top_k)),
        xlim=xlim,
        title=args.title,
        show_ci=bool(args.show_ci),
    )

    print(f"[saved] {out_path}")
    print("[ranking] logZ_alpha (higher is better; additive constant cancels in differences)")
    for e in _stable_sort_entries(entries):
        ztxt = "N/A" if not np.isfinite(e.logz) else f"{e.logz:.6g}"
        print(f"  {ztxt:>12}  |  {e.method:<16}  |  {e.label}")


if __name__ == "__main__":
    main()
