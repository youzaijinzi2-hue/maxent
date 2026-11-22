from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

# # 4並列で実行
# python scripts/run_pipeline.py --pipeline configs/pipeline_default_gaussian_sweep.yaml --workers 4

# # 途中再開（成功済みをスキップ）
# python scripts/run_pipeline.py --pipeline configs/pipeline_default_gaussian_sweep.yaml --resume

# # 失敗分だけ再試行
# python scripts/run_pipeline.py --pipeline configs/pipeline_default_gaussian_sweep.yaml --resume-failed

@dataclass
class TrialSpec:
    index: int
    name: str
    source: str
    set_values: dict[str, Any]
    overrides: dict[str, Any]


@dataclass
class TrialJob:
    spec: TrialSpec
    cfg_path: Path
    log_path: Path


def _slug(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("._-") or "trial"


def _resolve_path(raw: str | Path, *, base_dir: Path) -> Path:
    """
    Resolve path with a practical fallback strategy:
      1) absolute path as-is
      2) relative to pipeline directory
      3) relative to current working directory
    """
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()

    c1 = (base_dir / p).resolve()
    c2 = (Path.cwd() / p).resolve()
    if c1.exists():
        return c1
    if c2.exists():
        return c2
    # default to pipeline-relative for outputs/new files
    return c1


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = deepcopy(v)
    return dst


def _set_by_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [p for p in str(dotted_key).split(".") if p]
    if not parts:
        raise ValueError(f"Invalid dotted key: {dotted_key!r}")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = deepcopy(value)


def _render_template(template: str, values: dict[str, Any]) -> str:
    out = str(template)
    for k, v in values.items():
        out = out.replace("{" + str(k) + "}", str(v))
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _expand_matrix(
    matrix: dict[str, Any],
    *,
    start_index: int,
) -> list[TrialSpec]:
    axes = matrix.get("axes", {}) or {}
    if not isinstance(axes, dict):
        raise ValueError("matrix.axes must be a mapping.")
    if len(axes) == 0:
        return []

    keys = list(axes.keys())
    vals_list: list[list[Any]] = []
    for k in keys:
        vals = axes[k]
        if not isinstance(vals, list) or len(vals) == 0:
            raise ValueError(f"matrix.axes[{k!r}] must be a non-empty list.")
        vals_list.append(vals)

    name_template = matrix.get("name_template", None)
    common_set = matrix.get("set", {}) or {}
    common_overrides = matrix.get("overrides", {}) or {}
    if not isinstance(common_set, dict):
        raise ValueError("matrix.set must be a mapping of dotted_key -> value.")
    if not isinstance(common_overrides, dict):
        raise ValueError("matrix.overrides must be a mapping.")

    specs: list[TrialSpec] = []
    idx = int(start_index)
    for combo in itertools.product(*vals_list):
        axis_values = dict(zip(keys, combo))

        if name_template is None:
            # Keep auto-name short and deterministic.
            tokens = [f"{k.split('.')[-1]}={axis_values[k]}" for k in keys]
            name = "_".join(tokens)
        else:
            name = _render_template(str(name_template), axis_values)

        merged_set = dict(common_set)
        for k, v in axis_values.items():
            merged_set[k] = v

        specs.append(
            TrialSpec(
                index=idx,
                name=name,
                source="matrix",
                set_values=merged_set,
                overrides=deepcopy(common_overrides),
            )
        )
        idx += 1
    return specs


def _build_trial_specs(pipeline: dict[str, Any]) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    idx = 1

    # Explicit experiments
    experiments = pipeline.get("experiments", []) or []
    if not isinstance(experiments, list):
        raise ValueError("experiments must be a list.")

    for e in experiments:
        if not isinstance(e, dict):
            raise ValueError("Each experiments item must be a mapping.")
        name = str(e.get("name", f"exp_{idx:03d}"))
        set_values = e.get("set", {}) or {}
        overrides = e.get("overrides", {}) or {}
        if not isinstance(set_values, dict):
            raise ValueError(f"experiments[{name}].set must be a mapping.")
        if not isinstance(overrides, dict):
            raise ValueError(f"experiments[{name}].overrides must be a mapping.")
        specs.append(
            TrialSpec(
                index=idx,
                name=name,
                source="experiments",
                set_values=set_values,
                overrides=overrides,
            )
        )
        idx += 1

    # Matrix sweep (single mapping or list of mappings)
    matrix_cfg = pipeline.get("matrix", None)
    if matrix_cfg is not None:
        if isinstance(matrix_cfg, list):
            for m in matrix_cfg:
                if not isinstance(m, dict):
                    raise ValueError("matrix list entries must be mappings.")
                ms = _expand_matrix(m, start_index=idx)
                specs.extend(ms)
                idx += len(ms)
        elif isinstance(matrix_cfg, dict):
            ms = _expand_matrix(matrix_cfg, start_index=idx)
            specs.extend(ms)
            idx += len(ms)
        else:
            raise ValueError("matrix must be a mapping or list of mappings.")

    if len(specs) == 0:
        raise ValueError("No trials found. Add experiments and/or matrix in pipeline YAML.")
    return specs


def _extract_run_dir(stdout_text: str, stderr_text: str) -> str | None:
    txt = (stdout_text or "") + "\n" + (stderr_text or "")
    m = re.findall(r"\[OK\]\s*run saved to:\s*(.+)", txt)
    if m:
        return m[-1].strip()
    return None


def _extract_result_metrics(run_dir: str | None) -> tuple[float, str]:
    if not run_dir:
        return float("nan"), "unknown"
    p = Path(run_dir) / "artifacts" / "result.npz"
    if not p.exists():
        return float("nan"), "unknown"
    try:
        z = np.load(p, allow_pickle=True)
        if "logZ_alpha" in z.files:
            logz = float(np.asarray(z["logZ_alpha"]).reshape(()))
        elif "diag_logZ_alpha" in z.files:
            logz = float(np.asarray(z["diag_logZ_alpha"]).reshape(()))
        else:
            logz = float("nan")
        method = "unknown"
        if "alpha_weight_method" in z.files:
            method = str(np.asarray(z["alpha_weight_method"]).reshape(()))
        return logz, method
    except Exception:
        return float("nan"), "unknown"


def _run_one(
    *,
    python_exec: str,
    runner_script: Path,
    cfg_path: Path,
    log_path: Path,
    extra_args: list[str],
) -> tuple[int, float, str, str]:
    cmd = [python_exec, str(runner_script), "--config", str(cfg_path), *extra_args]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.write(f"[returncode] {proc.returncode}\n")
        f.write(f"[elapsed_sec] {elapsed:.3f}\n\n")
        f.write("===== STDOUT =====\n")
        f.write(proc.stdout or "")
        f.write("\n===== STDERR =====\n")
        f.write(proc.stderr or "")
    return proc.returncode, elapsed, (proc.stdout or ""), (proc.stderr or "")


def _run_plot_evidence(
    *,
    python_exec: str,
    plot_script: Path,
    run_dirs: list[str],
    out_path: Path,
    top_k: int,
    xlim: list[float] | tuple[float, float] | None,
    title: str | None,
    log_path: Path,
) -> tuple[int, float, str, str, list[str]]:
    cmd = [
        python_exec,
        str(plot_script),
        *run_dirs,
        "--out",
        str(out_path),
        "--top-k",
        str(int(top_k)),
    ]
    if xlim is not None and len(xlim) == 2:
        cmd.extend(["--xlim", str(float(xlim[0])), str(float(xlim[1]))])
    if title:
        cmd.extend(["--title", str(title)])

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.write(f"[returncode] {proc.returncode}\n")
        f.write(f"[elapsed_sec] {elapsed:.3f}\n\n")
        f.write("===== STDOUT =====\n")
        f.write(proc.stdout or "")
        f.write("\n===== STDERR =====\n")
        f.write(proc.stderr or "")
    return proc.returncode, elapsed, (proc.stdout or ""), (proc.stderr or ""), cmd


def _write_summary_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    if len(rows) == 0:
        return
    keys = [
        "index",
        "name",
        "source",
        "status",
        "note",
        "returncode",
        "duration_sec",
        "config_path",
        "run_dir",
        "logZ_alpha",
        "alpha_weight_method",
        "log_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in keys}
            w.writerow(out)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _normalize_resume_mode(mode: str) -> str:
    m = str(mode or "none").strip().lower()
    aliases = {
        "none": "none",
        "off": "none",
        "no": "none",
        "resume": "resume",
        "on": "resume",
        "true": "resume",
        "failed_only": "failed_only",
        "failed-only": "failed_only",
        "failed": "failed_only",
        "retry_failed": "failed_only",
        "retry-failed": "failed_only",
    }
    if m not in aliases:
        raise ValueError(
            f"Unknown resume_mode={mode!r}. Use one of: none, resume, failed_only."
        )
    return aliases[m]


def _load_previous_summary_rows(summary_csv: Path) -> dict[int, dict[str, Any]]:
    if not summary_csv.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    try:
        with summary_csv.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                idx_txt = str(row.get("index", "")).strip()
                if not idx_txt:
                    continue
                try:
                    idx = int(idx_txt)
                except Exception:
                    continue
                rows[idx] = {
                    "index": idx,
                    "name": str(row.get("name", "")),
                    "source": str(row.get("source", "")),
                    "status": str(row.get("status", "")),
                    "note": str(row.get("note", "")),
                    "returncode": row.get("returncode", ""),
                    "duration_sec": _safe_float(row.get("duration_sec", 0.0), 0.0),
                    "config_path": str(row.get("config_path", "")),
                    "run_dir": str(row.get("run_dir", "")),
                    "logZ_alpha": _safe_float(row.get("logZ_alpha", np.nan), np.nan),
                    "alpha_weight_method": str(row.get("alpha_weight_method", "")),
                    "log_path": str(row.get("log_path", "")),
                }
    except Exception:
        return {}
    return rows


def _previous_effective_state(prev_row: dict[str, Any] | None) -> str:
    """
    Infer effective previous state for resume logic.
    Returns one of: "missing", "ok", "failed", "unknown".
    """
    if prev_row is None:
        return "missing"

    prev_status = str(prev_row.get("status", "")).strip().lower()
    if prev_status == "ok":
        return "ok"
    if prev_status == "failed":
        return "failed"

    # For rows like "skipped", infer prior outcome from saved metadata.
    rc_raw = prev_row.get("returncode", "")
    rc_txt = str(rc_raw).strip()
    rc_val: int | None = None
    if rc_txt != "":
        try:
            rc_val = int(rc_txt)
        except Exception:
            rc_val = None
    run_dir = str(prev_row.get("run_dir", "")).strip()
    logz = _safe_float(prev_row.get("logZ_alpha", np.nan), np.nan)

    if rc_val == 0 and run_dir and np.isfinite(logz):
        return "ok"
    if rc_val is not None and rc_val != 0:
        return "failed"
    return "unknown"


def _decide_trial_action(
    *,
    resume_mode: str,
    prev_row: dict[str, Any] | None,
) -> tuple[str, str]:
    """
    Returns (action, note):
      - action in {"run", "skip"}
    """
    if resume_mode == "none":
        return "run", "fresh_run"

    prev_state = _previous_effective_state(prev_row)
    if resume_mode == "resume":
        # Skip completed successes, rerun failed/missing/unknown.
        if prev_state == "ok":
            return "skip", "resume_skip_ok"
        return "run", "resume_rerun_missing_failed_or_unknown"

    # failed_only
    if prev_state == "missing":
        return "skip", "failed_only_skip_no_previous_record"
    if prev_state == "failed":
        return "run", "failed_only_rerun_failed"
    if prev_state == "ok":
        return "skip", "failed_only_skip_ok"
    return "skip", "failed_only_skip_unknown"


def _build_trial_row_from_run(
    *,
    spec: TrialSpec,
    cfg_path: Path,
    log_path: Path,
    code: int,
    elapsed: float,
    stdout_text: str,
    stderr_text: str,
) -> dict[str, Any]:
    run_dir = _extract_run_dir(stdout_text, stderr_text)
    logz, method = _extract_result_metrics(run_dir)
    status = "ok" if code == 0 else "failed"
    return {
        "index": spec.index,
        "name": spec.name,
        "source": spec.source,
        "status": status,
        "note": "executed",
        "returncode": int(code),
        "duration_sec": round(float(elapsed), 3),
        "config_path": str(cfg_path),
        "run_dir": run_dir or "",
        "logZ_alpha": logz,
        "alpha_weight_method": method,
        "log_path": str(log_path),
        "set_values": deepcopy(spec.set_values),
        "overrides": deepcopy(spec.overrides),
    }


def _build_trial_row_from_previous(
    *,
    spec: TrialSpec,
    cfg_path: Path,
    log_path: Path,
    prev_row: dict[str, Any] | None,
    note: str,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        status = "planned_skip"
    else:
        status = "skipped"
    return {
        "index": spec.index,
        "name": spec.name,
        "source": spec.source,
        "status": status,
        "note": note,
        "returncode": (prev_row or {}).get("returncode", ""),
        "duration_sec": float((prev_row or {}).get("duration_sec", 0.0)),
        "config_path": str(cfg_path),
        "run_dir": str((prev_row or {}).get("run_dir", "")),
        "logZ_alpha": float((prev_row or {}).get("logZ_alpha", np.nan)),
        "alpha_weight_method": str((prev_row or {}).get("alpha_weight_method", "")),
        "log_path": str(log_path),
        "set_values": deepcopy(spec.set_values),
        "overrides": deepcopy(spec.overrides),
    }


def _execute_jobs(
    *,
    jobs: list[TrialJob],
    python_exec: str,
    runner_script: Path,
    runner_args: list[str],
    parallel_workers: int,
    fail_fast: bool,
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    if len(jobs) == 0:
        return rows, False

    total = len(jobs)
    saw_failure = False
    workers = max(1, int(parallel_workers))

    def run_job(job: TrialJob) -> dict[str, Any]:
        code, elapsed, out_txt, err_txt = _run_one(
            python_exec=python_exec,
            runner_script=runner_script,
            cfg_path=job.cfg_path,
            log_path=job.log_path,
            extra_args=runner_args,
        )
        return _build_trial_row_from_run(
            spec=job.spec,
            cfg_path=job.cfg_path,
            log_path=job.log_path,
            code=code,
            elapsed=elapsed,
            stdout_text=out_txt,
            stderr_text=err_txt,
        )

    if workers <= 1:
        for k, job in enumerate(jobs, start=1):
            print(f"[run] {job.spec.index:03d}/{total:03d} {job.spec.name}")
            row = run_job(job)
            rows.append(row)
            logz = _safe_float(row.get("logZ_alpha", np.nan), np.nan)
            print(
                f"  -> status={row.get('status')} elapsed={_safe_float(row.get('duration_sec', 0.0), 0.0):.1f}s "
                f"run_dir={row.get('run_dir') or 'N/A'} "
                f"logZ_alpha={logz if np.isfinite(logz) else 'N/A'} "
                f"method={row.get('alpha_weight_method') or 'unknown'}"
            )
            if row.get("status") != "ok":
                saw_failure = True
                if fail_fast:
                    print("[stop] fail_fast=true and a trial failed.")
                    break
        return rows, saw_failure

    print(f"[parallel] workers={workers}")
    pending: dict[Future, TrialJob] = {}
    job_iter = iter(jobs)
    stop_submitting = False
    launched = 0
    completed = 0

    def submit_next(ex: ThreadPoolExecutor) -> bool:
        nonlocal launched
        try:
            job = next(job_iter)
        except StopIteration:
            return False
        launched += 1
        print(f"[run] {job.spec.index:03d}/{total:03d} {job.spec.name} (queued)")
        fut = ex.submit(run_job, job)
        pending[fut] = job
        return True

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in range(workers):
            if not submit_next(ex):
                break

        while pending:
            done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                job = pending.pop(fut)
                try:
                    row = fut.result()
                except Exception as e:
                    row = {
                        "index": job.spec.index,
                        "name": job.spec.name,
                        "source": job.spec.source,
                        "status": "failed",
                        "note": f"internal_runner_error: {type(e).__name__}: {e}",
                        "returncode": -999,
                        "duration_sec": 0.0,
                        "config_path": str(job.cfg_path),
                        "run_dir": "",
                        "logZ_alpha": float("nan"),
                        "alpha_weight_method": "",
                        "log_path": str(job.log_path),
                        "set_values": deepcopy(job.spec.set_values),
                        "overrides": deepcopy(job.spec.overrides),
                    }
                rows.append(row)
                completed += 1
                logz = _safe_float(row.get("logZ_alpha", np.nan), np.nan)
                print(
                    f"  -> done {completed}/{total} [{job.spec.index:03d}] "
                    f"status={row.get('status')} "
                    f"elapsed={_safe_float(row.get('duration_sec', 0.0), 0.0):.1f}s "
                    f"logZ_alpha={logz if np.isfinite(logz) else 'N/A'}"
                )
                if row.get("status") != "ok":
                    saw_failure = True
                    if fail_fast:
                        stop_submitting = True

            if not stop_submitting:
                while len(pending) < workers:
                    if not submit_next(ex):
                        break

    if fail_fast and stop_submitting and launched < total:
        print(f"[stop] fail_fast=true; skipped submitting {total - launched} remaining trials.")

    return rows, saw_failure


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run multiple MaxEnt experiments from one pipeline YAML."
    )
    ap.add_argument("--pipeline", required=True, help="Path to pipeline YAML.")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and plan only (do not execute run_maxent.py).",
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on first failed run (overrides YAML setting).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for run_maxent.py (default: run.parallel_workers or 1).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume mode: skip previously successful trials; run failed or missing ones.",
    )
    ap.add_argument(
        "--resume-failed",
        action="store_true",
        help="Resume mode: rerun only trials that were previously failed.",
    )
    ap.add_argument(
        "--auto-plot",
        dest="auto_plot",
        action="store_true",
        help="Run plot_default_evidence.py automatically after pipeline finishes.",
    )
    ap.add_argument(
        "--no-auto-plot",
        dest="auto_plot",
        action="store_false",
        help="Skip automatic plot_default_evidence.py execution.",
    )
    ap.set_defaults(auto_plot=None)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    pipe_path = Path(args.pipeline).expanduser().resolve()
    pipe_dir = pipe_path.parent
    pipeline = _load_yaml(pipe_path)

    base_cfg_path_raw = pipeline.get("base_config", None)
    if base_cfg_path_raw is None:
        raise ValueError("pipeline YAML must define 'base_config'.")
    base_cfg_path = _resolve_path(base_cfg_path_raw, base_dir=pipe_dir)
    base_cfg = _load_yaml(base_cfg_path)

    run_cfg = pipeline.get("run", {}) or {}
    if not isinstance(run_cfg, dict):
        raise ValueError("run must be a mapping.")

    pipeline_name = str(run_cfg.get("name", pipe_path.stem))
    out_root_raw = run_cfg.get("output_root", f"runs/pipelines/{_slug(pipeline_name)}")
    out_root = Path(out_root_raw)
    if not out_root.is_absolute():
        out_root = (Path.cwd() / out_root).resolve()
    else:
        out_root = out_root.resolve()
    generated_cfg_dir = out_root / "generated_configs"
    logs_dir = out_root / "logs"
    canonical_summary_csv = out_root / "summary.csv"
    canonical_summary_json = out_root / "summary.json"
    if args.dry_run:
        summary_csv = out_root / "summary.dry_run.csv"
        summary_json = out_root / "summary.dry_run.json"
    else:
        summary_csv = canonical_summary_csv
        summary_json = canonical_summary_json
    runs_root = out_root / "runs"

    python_exec = str(run_cfg.get("python", sys.executable))
    runner_script = _resolve_path(run_cfg.get("runner_script", "scripts/run_maxent.py"), base_dir=pipe_dir)
    runner_args = run_cfg.get("runner_args", []) or []
    if not isinstance(runner_args, list) or not all(isinstance(x, str) for x in runner_args):
        raise ValueError("run.runner_args must be a list of strings.")
    plot_script = _resolve_path(run_cfg.get("plot_script", "scripts/plot_default_evidence.py"), base_dir=pipe_dir)
    plot_top_k = int(run_cfg.get("plot_top_k", 3))
    plot_title = run_cfg.get("plot_title", pipeline_name)
    plot_xlim = run_cfg.get("plot_xlim", None)
    plot_out_raw = run_cfg.get("plot_output", str(out_root / "default_evidence_comparison.png"))
    plot_out = Path(plot_out_raw)
    if not plot_out.is_absolute():
        plot_out = (Path.cwd() / plot_out).resolve()
    else:
        plot_out = plot_out.resolve()
    workers_cfg = int(run_cfg.get("parallel_workers", 1))
    parallel_workers = int(args.workers) if args.workers is not None else workers_cfg
    if parallel_workers <= 0:
        raise ValueError(f"workers/parallel_workers must be >=1, got {parallel_workers}")
    resume_mode_cfg = _normalize_resume_mode(str(run_cfg.get("resume_mode", "none")))

    global_set = pipeline.get("set", {}) or {}
    global_overrides = pipeline.get("overrides", {}) or {}
    if not isinstance(global_set, dict):
        raise ValueError("set must be a mapping (dotted_key -> value).")
    if not isinstance(global_overrides, dict):
        raise ValueError("overrides must be a mapping.")

    fail_fast = bool(args.fail_fast or pipeline.get("fail_fast", False))
    if args.resume and args.resume_failed:
        raise ValueError("Use either --resume or --resume-failed, not both.")
    if args.resume_failed:
        resume_mode = "failed_only"
    elif args.resume:
        resume_mode = "resume"
    else:
        resume_mode = resume_mode_cfg
    auto_plot_cfg = bool(run_cfg.get("auto_plot_evidence", True))
    auto_plot = auto_plot_cfg if args.auto_plot is None else bool(args.auto_plot)

    specs = _build_trial_specs(pipeline)

    generated_cfg_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    previous_rows = _load_previous_summary_rows(canonical_summary_csv)

    rows: list[dict[str, Any]] = []
    jobs: list[TrialJob] = []
    tag_prefix = _slug(str(run_cfg.get("tag_prefix", pipeline_name)))

    print(f"[pipeline] {pipe_path}")
    print(f"[base_config] {base_cfg_path}")
    print(f"[output_root] {out_root}")
    print(f"[trials] {len(specs)}")
    print(f"[runner] {runner_script}")
    print(f"[python] {python_exec}")
    print(f"[workers] {parallel_workers}")
    print(f"[resume_mode] {resume_mode}")
    print(f"[auto_plot] {auto_plot}")
    if len(previous_rows) > 0:
        print(f"[resume] loaded previous summary rows: {len(previous_rows)}")
    if args.dry_run:
        print("[mode] dry-run")

    for s in specs:
        name_slug = _slug(s.name)
        cfg = deepcopy(base_cfg)

        # Apply global + trial overrides.
        _deep_merge(cfg, deepcopy(global_overrides))
        for k, v in global_set.items():
            _set_by_dotted(cfg, k, v)
        _deep_merge(cfg, deepcopy(s.overrides))
        for k, v in s.set_values.items():
            _set_by_dotted(cfg, k, v)

        # Force output location/tag so trials are grouped by pipeline.
        if not isinstance(cfg.get("output"), dict):
            cfg["output"] = {}
        cfg["output"]["root"] = str(runs_root)
        cfg["output"]["tag"] = f"{tag_prefix}_{s.index:03d}_{name_slug}"

        cfg_path = generated_cfg_dir / f"{s.index:03d}_{name_slug}.yaml"
        cfg_path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        log_path = logs_dir / f"{s.index:03d}_{name_slug}.log"

        prev_row = previous_rows.get(int(s.index))
        action, action_note = _decide_trial_action(
            resume_mode=resume_mode,
            prev_row=prev_row,
        )

        if args.dry_run:
            if action == "run":
                row = {
                    "index": s.index,
                    "name": s.name,
                    "source": s.source,
                    "status": "planned_run",
                    "note": action_note,
                    "returncode": "",
                    "duration_sec": 0.0,
                    "config_path": str(cfg_path),
                    "run_dir": "",
                    "logZ_alpha": "",
                    "alpha_weight_method": "",
                    "log_path": str(log_path),
                    "set_values": deepcopy(s.set_values),
                    "overrides": deepcopy(s.overrides),
                }
                rows.append(row)
                print(f"[plan] run  {s.index:03d} {s.name} -> {cfg_path}")
            else:
                row = _build_trial_row_from_previous(
                    spec=s,
                    cfg_path=cfg_path,
                    log_path=log_path,
                    prev_row=prev_row,
                    note=action_note,
                    dry_run=True,
                )
                rows.append(row)
                print(f"[plan] skip {s.index:03d} {s.name} ({action_note})")
            continue

        if action == "run":
            jobs.append(TrialJob(spec=s, cfg_path=cfg_path, log_path=log_path))
        else:
            row = _build_trial_row_from_previous(
                spec=s,
                cfg_path=cfg_path,
                log_path=log_path,
                prev_row=prev_row,
                note=action_note,
                dry_run=False,
            )
            rows.append(row)
            print(f"[skip] {s.index:03d}/{len(specs):03d} {s.name} ({action_note})")

    saw_failure = False
    if (not args.dry_run) and len(jobs) > 0:
        run_rows, saw_failure = _execute_jobs(
            jobs=jobs,
            python_exec=python_exec,
            runner_script=runner_script,
            runner_args=runner_args,
            parallel_workers=parallel_workers,
            fail_fast=fail_fast,
        )
        rows.extend(run_rows)

    rows = sorted(rows, key=lambda r: _safe_int(r.get("index", 0), 0))

    ok_count = int(sum(1 for r in rows if r.get("status") == "ok"))
    failed_count = int(sum(1 for r in rows if r.get("status") == "failed"))
    skipped_count = int(sum(1 for r in rows if str(r.get("status", "")).startswith("skipped")))
    planned_count = int(sum(1 for r in rows if str(r.get("status", "")).startswith("planned")))

    # Plot candidates include both newly successful and skipped(previously successful) runs.
    plot_dirs: list[str] = []
    seen_plot_dirs: set[str] = set()
    for r in rows:
        rd = str(r.get("run_dir", "")).strip()
        if not rd:
            continue
        lz = _safe_float(r.get("logZ_alpha", np.nan), np.nan)
        if not np.isfinite(lz):
            continue
        if rd not in seen_plot_dirs:
            seen_plot_dirs.add(rd)
            plot_dirs.append(rd)

    _write_summary_csv(rows, summary_csv)
    summary_payload = {
        "pipeline": str(pipe_path),
        "base_config": str(base_cfg_path),
        "output_root": str(out_root),
        "dry_run": bool(args.dry_run),
        "fail_fast": bool(fail_fast),
        "resume_mode": str(resume_mode),
        "parallel_workers": int(parallel_workers),
        "runner_script": str(runner_script),
        "plot_script": str(plot_script),
        "auto_plot_evidence": bool(auto_plot),
        "plot_output": str(plot_out),
        "python": python_exec,
        "total_trials_defined": len(specs),
        "total_trials_executed": len(rows),
        "success_count": ok_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "planned_count": planned_count,
        "plot_candidate_count": int(len(plot_dirs)),
        "rows": rows,
    }
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[saved] {summary_csv}")
    print(f"[saved] {summary_json}")
    print(f"[summary] ok={ok_count} failed={failed_count} skipped={skipped_count} planned={planned_count}")
    if saw_failure and fail_fast and (not args.dry_run):
        print("[summary] fail_fast triggered before completing all runnable jobs.")

    # Suggest manual command (always useful for reproducibility).
    if len(plot_dirs) >= 1:
        joined = " ".join(f'"{p}"' for p in plot_dirs)
        print("[next] Compare evidence with:")
        print(
            f'  {python_exec} {plot_script} {joined} '
            f'--out "{plot_out}" --top-k {plot_top_k}'
        )

    # Optional auto-plot after all runs.
    if (not args.dry_run) and auto_plot:
        if len(plot_dirs) == 0:
            print("[plot] skip: no successful run with finite logZ_alpha.")
        elif not plot_script.exists():
            print(f"[plot] skip: plot script not found: {plot_script}")
        else:
            plot_log = logs_dir / "plot_default_evidence.log"
            print(f"[plot] running: {plot_script}")
            code, elapsed, out_txt, err_txt, cmd = _run_plot_evidence(
                python_exec=python_exec,
                plot_script=plot_script,
                run_dirs=plot_dirs,
                out_path=plot_out,
                top_k=plot_top_k,
                xlim=plot_xlim,
                title=(str(plot_title) if plot_title is not None else None),
                log_path=plot_log,
            )
            if code == 0:
                print(f"[plot] done in {elapsed:.1f}s -> {plot_out}")
            else:
                print(f"[plot] failed (code={code}). see log: {plot_log}")
                print(f"[plot] command: {' '.join(cmd)}")
                if out_txt.strip():
                    print("[plot][stdout-tail]")
                    print("\n".join(out_txt.strip().splitlines()[-10:]))
                if err_txt.strip():
                    print("[plot][stderr-tail]")
                    print("\n".join(err_txt.strip().splitlines()[-10:]))


if __name__ == "__main__":
    main()
