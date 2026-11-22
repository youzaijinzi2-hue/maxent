from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from maxent.io.xas import load_xas
from maxent.utils.provenance import sha256_file, get_git_info, get_env_info

def _build_input_manifest(*, cfg: Dict[str, Any], raw_path: str) -> Dict[str, Any]:
    raw_for_manifest = load_xas(str(raw_path))

    manifest = {
        "raw_files": [
           {
                "path": str(raw_path),
                "sha256": sha256_file(raw_path),
               "format": raw_for_manifest.meta.get("format", "unknown"),
                "xdi_header_lines": raw_for_manifest.meta.get("header_lines", [])[:300],
                "columns": {"0": "E", "1": "I0", "2": "Itrans"},
                "n_rows": int(raw_for_manifest.E.shape[0]),
                "stats": {
                    "E_min": float(np.min(raw_for_manifest.E)),
                    "E_max": float(np.max(raw_for_manifest.E)),
                    "I0_min": float(np.min(raw_for_manifest.I0)),
                    "I0_max": float(np.max(raw_for_manifest.I0)),
                    "Itrans_min": float(np.min(raw_for_manifest.Itrans)),
                    "Itrans_max": float(np.max(raw_for_manifest.Itrans)),
                },
            }
        ],
        "prepare_snapshot": {
            "beta": float(cfg["prepare"]["beta"]),
            "Nn": int(cfg["prepare"]["Nn"]),
           "grid": cfg.get("grid", {}),
            "alphas": cfg.get("alphas", {}),
            "sampler": cfg.get("sampler", {}),
        },
    }
    return manifest


def write_input_manifest_and_meta(
    *,
    run_dir: Path,
    run_id: str,
    cfg: Dict[str, Any],
   cfg_path: Path,
    raw_path: str,
    repro: Dict[str, str],
) -> None:
    # input_manifest.json
    manifest = _build_input_manifest(cfg=cfg, raw_path=raw_path)
    (run_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # run_meta.json
    meta = {
        "run_id": run_id,
        "git": get_git_info(),
        "env": get_env_info(),
        "seed": int(cfg.get("seed", 0)),
        "config_path": str(cfg_path),
        "input_manifest": manifest,
        "repro_files": repro,
        "code_fingerprint": {
            "prepare_module": "maxent.preprocess.xas_preprocess.Prepare",
            "sampler_module": "maxent.mcmc.sampler.MaxEnt",
        },
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )