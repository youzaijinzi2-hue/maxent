from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def prepare_run_dir(*, out_root: Path, run_id: str) -> Path:
    run_dir = out_root / run_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_resolved_config(*, run_dir: Path, cfg: Dict[str, Any]) -> Path:
    p = run_dir / "config_resolved.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p

def save_reproducibility_files(run_dir: Path) -> dict[str, str]:
    """
    Save minimal reproducibility artifacts into run_dir/meta:
      - git_sha.txt: `git rev-parse HEAD`
      - pip_freeze.txt: `pip freeze`
      - python.txt: python version string

    Returns keys: git_sha, pip_freeze_path, python_path
    """
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # --- git sha ---
    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    (meta_dir / "git_sha.txt").write_text(git_sha + "\n", encoding="utf-8")

    # --- pip freeze ---
    pip_freeze_txt = "pip freeze failed\n"
    try:
        pip_freeze_txt = subprocess.check_output(["pip", "freeze"], text=True)
    except Exception:
        pass
    (meta_dir / "pip_freeze.txt").write_text(pip_freeze_txt, encoding="utf-8")

    # --- python version ---
    (meta_dir / "python.txt").write_text(sys.version + "\n", encoding="utf-8")

    return {
        "git_sha": git_sha,
        "pip_freeze_path": str(meta_dir / "pip_freeze.txt"),
        "python_path": str(meta_dir / "python.txt"),
    }