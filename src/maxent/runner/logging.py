from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    *,
    run_dir: str | Path,
    name: str = "maxent",
    level: str = "INFO",
    log_filename: str = "run.log",
) -> logging.Logger:
    """
    Create a runner logger that logs to:
      - stdout (console)
      - run_dir/logs/run.log (file)

    Notes:
      - idempotent: calling multiple times won't duplicate handlers
      - level: "DEBUG"/"INFO"/"WARNING"/"ERROR"
    """
    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / log_filename

    logger = logging.getLogger(name)
    logger.propagate = False  # avoid double logging via root

    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(lvl)

    # avoid duplicate handlers if re-imported / re-run in notebook
    if getattr(logger, "_maxent_configured", False):
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console handler
    sh = logging.StreamHandler()
    sh.setLevel(lvl)
    sh.setFormatter(fmt)

    # file handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(lvl)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    logger._maxent_configured = True  # type: ignore[attr-defined]
    logger.info(f"[log] logger initialized: level={logging.getLevelName(lvl)} file={log_path}")
    return logger


def set_level(logger: logging.Logger, level: str) -> None:
    """Update logger + handlers level."""
    lvl = getattr(logging, str(level).upper(), None)
    if lvl is None:
        raise ValueError(f"Unknown logging level: {level!r}")
    logger.setLevel(lvl)
    for h in list(logger.handlers):
        h.setLevel(lvl)