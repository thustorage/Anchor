#!/usr/bin/env python3
"""Collect DeepSpeed (training) fault / e2e / overhead numbers into three CSVs.

No arguments. Run it from anywhere:

    python extract_ds_perf.py

It scans the report directories next to this script:

    fault/report1/<method>[_multigpu]/<model>.json          single crash -> recover breakdown
    e2e/report1/e2e_<method>[_multigpu]/<model>.json         throughput under repeated crashes
    overhead/report1[_multigpu]/<method>/<model>.json        steady-state cost (baseline vs ipc)

and writes three CSVs in this same directory:

    ds_fault_summary.csv  ds_e2e_summary.csv  ds_overhead_summary.csv

Each row is identified by only (method, config, model); everything else is a performance
metric. Auxiliary dumps (``*_state.json``, ``*_profile.json``, ...) are skipped: a file is
a main report only if it carries the suite's headline metric.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

FAULT_CSV = SCRIPT_DIR / "ds_fault_summary.csv"
E2E_CSV = SCRIPT_DIR / "ds_e2e_summary.csv"
OVERHEAD_CSV = SCRIPT_DIR / "ds_overhead_summary.csv"

FAULT_COLUMNS = [
    "method", "config", "model",
    "recovery_total_seconds", "engine_init_seconds", "load_seconds",
    "train_block_seconds", "other_seconds",
]
E2E_COLUMNS = [
    "method", "config", "model",
    "average_steps_per_second", "fault_injection_count",
]
OVERHEAD_COLUMNS = [
    "method", "config", "model",
    "train_step_milliseconds", "gpu_used_gib",
]

METHOD_ORDER = {"baseline": 0, "ds_native": 1, "native": 1, "checkfreq": 2,
                "memory": 3, "ipc": 4}

AUX_TOKENS = ("_state", "_meta", "_runtime", "_recovery", "_inference", "_first", "_profile")


def _is_aux(path: Path) -> bool:
    stem = path.stem.lower()
    return any(tok in stem for tok in AUX_TOKENS)


def _load(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _model_size(label) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", str(label))
    return float(m.group(1)) if m else float("inf")


def _config(data: dict, path: Path) -> str:
    gids = data.get("gpu_ids")
    if gids:
        return f"{len(str(gids).split(','))}gpu"
    ws = data.get("expected_world_size")
    if ws:
        return f"{int(ws)}gpu"
    return "multigpu" if "_multigpu" in str(path) else "1gpu"


def _method_from_dir(dirname: str) -> str:
    """Normalise a report subdir to a bare method name."""
    name = dirname
    if name.startswith("e2e_"):
        name = name[len("e2e_"):]
    if name.endswith("_multigpu"):
        name = name[:-len("_multigpu")]
    return name


def collect_fault(rows: list[dict]) -> None:
    root = SCRIPT_DIR / "fault" / "report1"
    if not root.is_dir():
        return
    for jf in sorted(root.glob("*/*.json")):
        if _is_aux(jf):
            continue
        data = _load(jf)
        if not data or "recovery_total_seconds" not in data:
            continue
        rows.append({
            "method": _method_from_dir(jf.parent.name),
            "config": _config(data, jf),
            "model": data.get("model_label", jf.stem),
            "recovery_total_seconds": data.get("recovery_total_seconds", ""),
            "engine_init_seconds": data.get("engine_init_seconds", ""),
            "load_seconds": data.get("load_seconds", ""),
            "train_block_seconds": data.get("train_block_seconds", ""),
            "other_seconds": data.get("other_seconds", ""),
        })


def collect_e2e(rows: list[dict]) -> None:
    root = SCRIPT_DIR / "e2e" / "report1"
    if not root.is_dir():
        return
    for jf in sorted(root.glob("*/*.json")):
        if _is_aux(jf):
            continue
        data = _load(jf)
        if not data or "average_steps_per_second" not in data:
            continue
        rows.append({
            "method": _method_from_dir(jf.parent.name),
            "config": _config(data, jf),
            "model": data.get("model_label", jf.stem),
            "average_steps_per_second": data.get("average_steps_per_second", ""),
            "fault_injection_count": data.get("fault_injection_count", ""),
        })


def collect_overhead(rows: list[dict]) -> None:
    for base in ("overhead/report1", "overhead/report1_multigpu"):
        root = SCRIPT_DIR / base
        if not root.is_dir():
            continue
        for jf in sorted(root.glob("*/*.json")):
            if _is_aux(jf):
                continue
            data = _load(jf)
            if not data or "train_step_milliseconds" not in data:
                continue
            rows.append({
                "method": data.get("method", jf.parent.name),
                "config": _config(data, jf),
                "model": data.get("model_label", jf.stem),
                "train_step_milliseconds": data.get("train_step_milliseconds", ""),
                "gpu_used_gib": data.get("mid_training_gpu_used_gib_avg", ""),
            })


def _fmt(value):
    return round(value, 4) if isinstance(value, float) else value


def _write(path: Path, columns: list[str], rows: list[dict]) -> None:
    rows.sort(key=lambda r: (
        _model_size(r["model"]),
        r.get("config", ""),
        METHOD_ORDER.get(r["method"], 9),
    ))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: _fmt(r.get(c, "")) for c in columns})
    print(f"Wrote {len(rows):2d} rows -> {path.name}")


def main() -> int:
    fault_rows: list[dict] = []
    e2e_rows: list[dict] = []
    overhead_rows: list[dict] = []
    collect_fault(fault_rows)
    collect_e2e(e2e_rows)
    collect_overhead(overhead_rows)

    _write(FAULT_CSV, FAULT_COLUMNS, fault_rows)
    _write(E2E_CSV, E2E_COLUMNS, e2e_rows)
    _write(OVERHEAD_CSV, OVERHEAD_COLUMNS, overhead_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
