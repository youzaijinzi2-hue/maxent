from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from maxent.plotting import plotting
from maxent.alpha_landscape import alpha_landscape

def run_postprocess(
    *,
    run_dir: str | Path,
    D_true: torch.Tensor,
    D_default: torch.Tensor,
    roi: Tuple[float, float],
    plot_cfg: Dict[str, Any] | None = None,
) -> Path:
    """
    Runner-level postprocess entrypoint.

    - Reads result.npz written by the runner
    - Generates standard spectrum plot + optional peak histogram

    This keeps all plotting policy in one place.
    """
    run_dir = Path(run_dir)
    plot_cfg = plot_cfg or {}

    # Inputs (with defaults matching old run_maxent.py behavior)
    xlim = tuple(plot_cfg.get("xlim", None))
    normalize_mode = str(plot_cfg.get("normalize_mode", "max"))
    peak_hist = bool(plot_cfg.get("peak_hist", True))
    peak_bins = int(plot_cfg.get("peak_bins", 25))

    out_pdf = Path(plot_cfg.get("out_pdf", run_dir / "spectrum.pdf"))
    result_npz = run_dir / "artifacts" / "result.npz"

    plotting(
        npz_path=result_npz,
        D_true=D_true,
        xlim=xlim,
        D_default=D_default,
        out_pdf=out_pdf,
        normalize_mode=normalize_mode,
        roi=roi,
        peak_hist=peak_hist,
        peak_bins=peak_bins,
    )
    alpha_landscape(result_npz, run_dir)
    
    return out_pdf