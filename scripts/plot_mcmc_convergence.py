from __future__ import annotations

import argparse
from pathlib import Path

from maxent.diagnostics.mcmc_convergence import save_mcmc_convergence_figure


def _fmt(v: float, nd: int = 4) -> str:
    try:
        vv = float(v)
    except Exception:
        return "N/A"
    if vv != vv:  # NaN
        return "N/A"
    return f"{vv:.{nd}g}"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Plot MCMC convergence diagnostics from a MaxEnt run directory "
            "(or artifacts/result.npz): move-adaptation (p_m + utility), split-Rhat, ESS/tau_int, and summary."
        )
    )
    ap.add_argument("path", help="Run directory or direct path to artifacts/result.npz.")
    ap.add_argument("--out", default=None, help="Output PNG path. Default: <run_dir>/mcmc_convergence.png")
    ap.add_argument(
        "--rhat-threshold",
        type=float,
        default=1.05,
        help="Threshold line for R-hat panel and pass/fail summary.",
    )
    ap.add_argument(
        "--ess-min-per-chain",
        type=float,
        default=50.0,
        help="Target minimum ESS per chain used in summary.",
    )
    ap.add_argument(
        "--max-replicas",
        type=int,
        default=12,
        help="Maximum number of replicas to display in per-replica panels.",
    )
    ap.add_argument("--title", default=None, help="Optional figure title.")
    ap.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = Path(args.out).expanduser().resolve() if args.out else None
    saved, summary = save_mcmc_convergence_figure(
        args.path,
        out_path=out_path,
        rhat_threshold=float(args.rhat_threshold),
        ess_min_per_chain=float(args.ess_min_per_chain),
        max_replicas=max(1, int(args.max_replicas)),
        title=args.title,
        dpi=int(args.dpi),
    )
    print(f"[saved] {saved}")
    print(
        "[summary] "
        f"rhat_max={_fmt(summary.get('rhat_max', float('nan')))} "
        f"ess_min_per_chain={_fmt(summary.get('ess_min_per_chain', float('nan')))} "
        f"total_ess={_fmt(summary.get('total_ess', float('nan')))} "
        f"mcse_max={_fmt(summary.get('mcse_max', float('nan')))}"
    )


if __name__ == "__main__":
    main()
