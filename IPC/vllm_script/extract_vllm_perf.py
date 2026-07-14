#!/usr/bin/env python3
"""Collect vLLM overhead + e2e performance numbers into one CSV.

No arguments. Run it from anywhere:

    python extract_vllm_perf.py

It scans the report directories next to this script:

    overhead/report/<method>/<model>.json            (single-GPU)
    overhead/report_multigpu/<method>/<model>.json   (multi-GPU)
    e2e/report1/<variant>/<model>.json               (replay / kv_memory / ipc, single & multi)

and writes two CSVs in this same directory — ``vllm_overhead_summary.csv`` and
``vllm_e2e_summary.csv`` (kept separate so neither has blank columns). Auxiliary files
(``*_state.json``, ``*_meta.json``, ``*_runtime.json``, recovery/inference dumps, ...)
are skipped automatically: a file is treated as a main report only if it carries the
suite's headline metric (throughput for overhead, ms/request for e2e).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OVERHEAD_CSV = SCRIPT_DIR / "vllm_overhead_summary.csv"
E2E_CSV = SCRIPT_DIR / "vllm_e2e_summary.csv"

OVERHEAD_COLUMNS = [
    "method", "config", "model",
    "throughput_tps", "gpu_used_gib",
]
E2E_COLUMNS = [
    "method", "config", "model",
    "ms_per_request", "completed_requests",
    "fault_injection_count",
    "fault_to_recovery_ready_seconds",
    "connector_python_startup_seconds", "connector_engine_startup_seconds",
    "other_seconds",
]

E2E_METHOD_BY_BACKEND = {
    "vllm_replay_incremental_token_log": "replay",
    "vllm_kv_cpu_buffer": "kv_memory",
    "vllm_ipc_live": "ipc",
}

AUX_TOKENS = ("_state", "_meta", "_runtime", "_recovery", "_inference", "_first")


def _is_aux(path: Path) -> bool:
    stem = path.stem.lower()
    return any(tok in stem for tok in AUX_TOKENS)


def _load(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _config_of(tp) -> str:
    try:
        tp = int(tp)
    except (TypeError, ValueError):
        return ""
    return "single" if tp <= 1 else f"TP{tp}"


def _model_size(label) -> float:
    """Extract the parameter count from a label like 'qwen3 14B' for sorting."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", str(label))
    return float(m.group(1)) if m else float("inf")


def _e2e_method(data: dict, dirname: str) -> str:
    backend = data.get("checkpoint_backend", "")
    if backend in E2E_METHOD_BY_BACKEND:
        return E2E_METHOD_BY_BACKEND[backend]
    for key in ("kv_memory", "replay", "ipc"):
        if key in dirname:
            return key
    return dirname


def collect_overhead(rows: list[dict]) -> None:
    for base in ("overhead/report", "overhead/report_multigpu"):
        root = SCRIPT_DIR / base
        if not root.is_dir():
            continue
        for jf in sorted(root.glob("*/*.json")):
            if _is_aux(jf):
                continue
            data = _load(jf)
            if not data or "average_token_throughput_tps" not in data:
                continue
            rows.append({
                "suite": "overhead",
                "method": data.get("method", jf.parent.name),
                "config": _config_of(data.get("tensor_parallel_size")),
                "model": data.get("model_label", jf.stem),
                "source": str(jf.parent.relative_to(SCRIPT_DIR)),
                "tensor_parallel_size": data.get("tensor_parallel_size", ""),
                "max_model_len": data.get("max_model_len", ""),
                "gpu_memory_utilization": data.get("gpu_memory_utilization", ""),
                "dtype": data.get("dtype", ""),
                "throughput_tps": data.get("average_token_throughput_tps", ""),
                "gpu_used_gib": data.get("mid_benchmark_gpu_used_gib_avg", ""),
                "num_requests": data.get("num_requests", ""),
            })


def collect_e2e(rows: list[dict]) -> None:
    root = SCRIPT_DIR / "e2e" / "report1"
    if not root.is_dir():
        return
    for jf in sorted(root.glob("*/*.json")):
        if _is_aux(jf):
            continue
        data = _load(jf)
        if not data or "average_ms_per_request" not in data:
            continue
        rows.append({
            "suite": "e2e",
            "method": _e2e_method(data, jf.parent.name),
            "config": _config_of(data.get("tensor_parallel_size")),
            "model": data.get("model_label", jf.stem),
            "source": str(jf.parent.relative_to(SCRIPT_DIR)),
            "tensor_parallel_size": data.get("tensor_parallel_size", ""),
            "max_model_len": data.get("max_model_len", ""),
            "gpu_memory_utilization": data.get("gpu_memory_utilization", ""),
            "dtype": data.get("dtype", ""),
            "ms_per_request": data.get("average_ms_per_request", ""),
            "completed_requests": data.get("completed_requests", ""),
            "fault_injection_count": data.get("fault_injection_count", ""),
            "worker_restart_count": data.get("worker_restart_count", ""),
            "fault_to_recovery_ready_seconds": data.get("fault_to_recovery_ready_seconds", ""),
            "recovery_sample_count": data.get("recovery_sample_count", ""),
            "connector_python_startup_seconds": data.get("connector_python_startup_seconds", ""),
            "connector_engine_startup_seconds": data.get("connector_vllm_engine_startup_seconds", ""),
            "other_seconds": data.get("other_seconds", ""),
        })


def _fmt(value):
    return round(value, 4) if isinstance(value, float) else value


def _write(path: Path, columns: list[str], rows: list[dict]) -> None:
    method_order = {"baseline": 0, "replay": 1, "kv_memory": 2, "ipc": 3}
    rows.sort(key=lambda r: (
        _model_size(r["model"]),
        r.get("config", ""),
        method_order.get(r["method"], 9),
    ))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: _fmt(r.get(c, "")) for c in columns})
    print(f"Wrote {len(rows):2d} rows -> {path.name}")


def main() -> int:
    overhead_rows: list[dict] = []
    e2e_rows: list[dict] = []
    collect_overhead(overhead_rows)
    collect_e2e(e2e_rows)

    _write(OVERHEAD_CSV, OVERHEAD_COLUMNS, overhead_rows)
    _write(E2E_CSV, E2E_COLUMNS, e2e_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
