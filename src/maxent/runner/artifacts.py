from __future__ import annotations

"""
Runner-level artifact IO facade.

目的:
  - run_maxent.py が「どの順で何を保存するか」をここに委譲できるようにする
  - 実体の保存処理は既存の maxent.io.artifacts を呼ぶ（互換性維持）

このファイルを窓口にしておくと、将来
  - 保存キーの増減
  - ファイル名のルール変更
  - 追加のメタデータ保存（png, csv, parquet など）
を runner 側だけ触って対応できる。
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np

# Keep using the canonical implementation
from maxent.io.artifacts import (
    save_inputs_npz as _save_inputs_npz,
    build_result_dict as _build_result_dict,
    save_result_npz as _save_result_npz,
)


# --- Re-export the existing API (drop-in replacement) ---
def save_inputs_npz(**kwargs: Any) -> None:
    """Drop-in wrapper for maxent.io.artifacts.save_inputs_npz."""
    # NOTE: underlying implementation is keyword-only (no positional args)
    _save_inputs_npz(**kwargs)


def build_result_dict(**kwargs: Any) -> Dict[str, Any]:
    """Drop-in wrapper for maxent.io.artifacts.build_result_dict."""
    return _build_result_dict(**kwargs)

def save_result_npz(**kwargs: Any) -> None:
    """Drop-in wrapper for maxent.io.artifacts.save_result_npz."""
    # NOTE: underlying implementation is keyword-only (no positional args)
    _save_result_npz(**kwargs)

# --- Optional higher-level helper (future use) ---
def save_run_artifacts(
    *,
    run_dir: str | Path,
    inputs: Dict[str, Any],
    results: Dict[str, Any],
    inputs_name: str = "inputs.npz",
    result_name: str = "result.npz",
) -> Dict[str, str]:
    """
    Convenience function (not required by runner yet):
      - saves inputs.npz and result.npz into run_dir/artifacts

    inputs:  save_inputs_npz(**inputs)
    results: save_result_npz(build_result_dict(**results))
    """
    run_dir = Path(run_dir)
    art_dir = run_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    inputs_path = art_dir / inputs_name
    result_path = art_dir / result_name

    # Underlying save_inputs_npz expects keyword-only; our wrapper handles it
    save_inputs_npz(inputs_path, **inputs)
    rd = build_result_dict(**results)
    save_result_npz(result_path, rd)

    return {"inputs_npz": str(inputs_path), "result_npz": str(result_path)}

# NOTE:
# ここには既に save_inputs_npz / build_result_dict / save_result_npz が存在している前提。
# 本パッチでは「runner が保存手順を一括で握る」ための save_run_artifacts を追加する。

def save_run_artifacts(
    *,
    run_dir: str | Path,
    save_dtype,
    # ---- inputs ----
    wn,
    G_obs,
    omega,
    D_w,
    tau,
    sigma_tau,
    alphas,
    # ---- results ----
    A_est,
    A_low,
    A_high,
    mcse,
    diagnostics: Dict[str, Any],
    c_tau: float,
    tau_metrics_raw: Dict[str, Any],
    tau_metrics: Dict[str, Any],
    G_tau_noisy,
    G_tau_rec,
    peak_pos_roi=None,
) -> Dict[str, Any]:
    """
    One-shot artifact writer for the runner.

    This preserves the previous behavior of run_maxent.py:
      - writes inputs.npz
      - builds result_dict
      - force-add peak_pos_thin + thinning metadata (if present)
      - writes result.npz
    """
    run_dir = Path(run_dir)
    art_dir = run_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    inputs_npz = art_dir / "inputs.npz"
    # --- inputs ---
    save_inputs_npz(
        run_dir=run_dir,
        save_dtype=save_dtype,
        wn=wn,
        G_obs=G_obs,
        omega=omega,
        D_w=D_w,
        tau=tau,
        sigma_tau=sigma_tau,
        alphas=alphas,
    )

    # --- results ---
    result_dict = build_result_dict(
        save_dtype=save_dtype,
        omega=omega,
        A_est=A_est,
        A_low=A_low,
        A_high=A_high,
        mcse=mcse,
        diagnostics=diagnostics,
        c_tau=c_tau,
        tau_metrics_raw=tau_metrics_raw,
        tau_metrics=tau_metrics,
        G_tau_noisy=G_tau_noisy,
        G_tau_rec=G_tau_rec,
        peak_pos_roi=peak_pos_roi,
    )

    # ---- force-save alpha-conditioned peak positions for alpha-landscape ----
    if isinstance(diagnostics, dict) and ("peak_pos_thin" in diagnostics):
        result_dict["peak_pos_thin"] = diagnostics["peak_pos_thin"]
    for k in ["peak_thin_start", "peak_thin_stride", "peak_thin_indices"]:
        if isinstance(diagnostics, dict) and (k in diagnostics):
            result_dict[k] = diagnostics[k]

    save_result_npz(run_dir=run_dir, save_dtype=save_dtype, result_dict=result_dict)
    return result_dict