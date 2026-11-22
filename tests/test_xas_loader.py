# tests/test_xas_loader.py
# .xdi っぽいヘッダ（# 行）付き
# 3列数値データをちゃんと読める
# 返る配列の長さ一致・finite・metaにheader_linesが入る
from __future__ import annotations

from pathlib import Path
import numpy as np

from maxent.io.xas import load_xas


def test_load_xas_reads_header_and_3cols(tmp_path: Path) -> None:
    # Create a minimal XDI-like file:
    # - header lines start with '#'
    # - numeric table has at least 3 columns: E, I0, Itrans
    content = "\n".join(
        [
            "# XDI/1.0 example",
            "# Element: Cr",
            "# Beamline: BL-TEST",
            "# Column.1: energy eV",
            "# Column.2: I0",
            "# Column.3: Itrans",
            "100.0  10.0  1.0",
            "101.0  11.0  1.1",
            "102.0  12.0  1.2",
        ]
    )
    p = tmp_path / "sample.xdi"
    p.write_text(content, encoding="utf-8")

    raw = load_xas(str(p))

    # shapes
    assert raw.E.shape == raw.I0.shape == raw.Itrans.shape
    assert raw.E.ndim == 1
    assert raw.E.size == 3

    # finite values
    assert np.isfinite(raw.E).all()
    assert np.isfinite(raw.I0).all()
    assert np.isfinite(raw.Itrans).all()

    # meta should contain header_lines
    assert "header_lines" in raw.meta
    assert len(raw.meta["header_lines"]) >= 1
    assert raw.meta["header_lines"][0].startswith("#")
