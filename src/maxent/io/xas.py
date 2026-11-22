# src/maxent/io/xas.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import os

@dataclass(frozen=True)
class XASRaw:
    E: np.ndarray
    I0: np.ndarray
    Itrans: np.ndarray
    source_path: str
    meta: Dict[str, Any]


def _read_numeric_3col(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read numeric table where columns are E, I0, Itrans in the first 3 columns.
    Works for .xdi-like text files with header lines starting with '#'.
    """
    arr = np.genfromtxt(path, comments="#")
    if arr.ndim == 1:
        # single row edge case
        arr = arr[None, :]
    if arr.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns (E, I0, Itrans) in {path}, got shape {arr.shape}")
    E = arr[:, 0].astype(float)
    I0 = arr[:, 1].astype(float)
    Itrans = arr[:, 2].astype(float)
    return E, I0, Itrans


def _read_xdi_meta(path: Path) -> Dict[str, Any]:
    """
    Very lightweight XDI header reader:
    - Collect lines beginning with '#'
    - Keep them as raw lines + try to parse 'key: value' style if present.
    This is optional but useful for provenance.
    """
    meta: Dict[str, Any] = {"format": "xdi_or_text", "header_lines": []}
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                s = line.strip()
                meta["header_lines"].append(s)
                # Try parse: "# key: value"
                s2 = s.lstrip("#").strip()
                if ":" in s2:
                    k, v = s2.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if k:
                        meta[k] = v
    except Exception:
        # If meta parse fails, we still can read numeric part
        pass
    return meta

def find_repo_root(start: Path | None = None) -> Path:
    # 目印ファイルでルートを決める（pyproject.toml / .git など）
    start = start or Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    # 最後の手段：カレントをルート扱い
    return Path.cwd().resolve()

def resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    root = Path(os.environ.get("PROJECT_ROOT", find_repo_root()))
    return (root / path).resolve()

def load_xas(path: str) -> XASRaw:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    meta = _read_xdi_meta(p)
    E, I0, Itrans = _read_numeric_3col(p)
    return XASRaw(E=E, I0=I0, Itrans=Itrans, source_path=str(p), meta=meta)
