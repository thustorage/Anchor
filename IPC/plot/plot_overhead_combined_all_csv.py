#!/usr/bin/env python3
"""Side-by-side dual-axis overhead charts: Training (left) + Inference (right).

Data is extracted from the two tidy summary CSVs produced by the extractors:
    ../ds_script/ds_overhead_summary.csv       (training: gpu_used_gib, train_step_milliseconds)
    ../vllm_script/vllm_overhead_summary.csv    (inference: gpu_used_gib, throughput_tps)

X-axis ordering: single-GPU models by size ascending (small on the left); multi-GPU
entries are appended at the very end (regardless of size) with a ``TP<n>`` suffix.
Only the data-extraction changed — the plotting format is unchanged.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR / "report1"
DEFAULT_OUTPUT_PREFIX = SCRIPT_DIR / "overhead_combined_all"

DEFAULT_TRAIN_CSV = SCRIPT_DIR.parent / "ds_script" / "ds_overhead_summary.csv"
DEFAULT_INFER_CSV = SCRIPT_DIR.parent / "vllm_script" / "vllm_overhead_summary.csv"

METHOD_ORDER = ["no_ckpt", "ipc"]
METHOD_LABELS = {"no_ckpt": "No Checkpoint", "ipc": "ANCHOR"}
BAR_COLORS = {"no_ckpt": "#4C72B0", "ipc": "#55A868"}
BAR_HATCHES = {"no_ckpt": "", "ipc": "//"}
MARKER_COLORS = {"no_ckpt": "#DD8452", "ipc": "#8172B3"}
MARKER_STYLES = {"no_ckpt": "o", "ipc": "D"}

_METHOD_ALIASES: dict[str, str] = {}
for _k, _v in {
    "no-ckpt": "no_ckpt", "no_ckpt": "no_ckpt", "no ckpt": "no_ckpt",
    "nockpt": "no_ckpt", "baseline": "no_ckpt",
    "ipc": "ipc", "anchor": "ipc",
}.items():
    _METHOD_ALIASES[_k.lower()] = _v


def _cap(n: str) -> str:
    return n[0].upper() + n[1:] if n else n


def _norm_method(s: str) -> str:
    key = re.sub(r"\s+", " ", s.strip().lower().replace("_", " ").replace("-", " "))
    if key in _METHOD_ALIASES:
        return _METHOD_ALIASES[key]
    raise SystemExit(f"Unknown method in CSV: {s!r}")


def _model_size(label: str) -> float:
    """Parameter count for sorting. Handles '1_7B' (== 1.7B) written with an underscore."""
    norm = re.sub(r"(\d)_(\d)", r"\1.\2", str(label))
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", norm)
    return float(m.group(1)) if m else float("inf")


def _config_info(config: str) -> tuple[bool, int]:
    """Return (is_multi_gpu, degree). Single = {'single','1gpu'} or degree 1. Multi configs
    without a number (e.g. DeepSpeed 'multigpu') default to degree 2 — every multi-GPU
    overhead run in this artifact used 2 GPUs."""
    c = config.strip().lower()
    m = re.search(r"\d+", c)
    n = int(m.group()) if m else None
    if c in ("single", "1gpu") or n == 1:
        return False, 1
    return True, (n or 2)


def load_overhead_summary(
    path: Path, secondary_col: str,
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Read a tidy overhead CSV -> (ordered_column_keys, mem_by_method, secondary_by_method).

    Column key = model name for single-GPU, ``"<model> TP<n>"`` for multi-GPU.
    """
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    mem: dict[str, dict[str, float]] = {}
    secondary: dict[str, dict[str, float]] = {}
    meta: dict[str, tuple[float, bool]] = {}  # colkey -> (size, is_multi)

    for row in rows:
        if not row.get("method"):
            continue
        method = _norm_method(row["method"])
        model = row["model"].strip().replace("_", ".")  # 'qwen3 1_7B' -> 'qwen3 1.7B'
        is_multi, degree = _config_info(row.get("config", ""))
        colkey = f"{model} TP{degree}" if is_multi else model

        def _num(col: str) -> float:
            cell = (row.get(col) or "").strip()
            return float(cell) if cell else 0.0

        mem.setdefault(method, {})[colkey] = _num("gpu_used_gib")
        secondary.setdefault(method, {})[colkey] = _num(secondary_col)
        meta[colkey] = (_model_size(model), is_multi)

    singles = sorted((k for k, v in meta.items() if not v[1]), key=lambda k: meta[k][0])
    multis = sorted((k for k, v in meta.items() if v[1]), key=lambda k: meta[k][0])
    return singles + multis, mem, secondary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combined training+inference overhead (side-by-side).")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--infer-csv", type=Path, default=DEFAULT_INFER_CSV)
    p.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def draw_panel(
    ax_bar, ax_marker,
    models, mem_data, secondary_data,
    ylabel_left, ylabel_right,
    subtitle, secondary_scale,
):
    bar_width = 0.28
    gc = list(range(len(models)))
    offsets = {m: (i - 0.5) * bar_width for i, m in enumerate(METHOD_ORDER)}

    for method in METHOD_ORDER:
        xs = [c + offsets[method] for c in gc]
        heights = [mem_data.get(method, {}).get(m, 0.0) for m in models]
        ax_bar.bar(
            xs, heights, width=bar_width,
            color=BAR_COLORS[method], edgecolor="#333333", linewidth=0.5,
            hatch=BAR_HATCHES[method], alpha=0.85, zorder=2,
        )

    marker_size = 11
    for method in METHOD_ORDER:
        xs = [c + offsets[method] for c in gc]
        raw = [secondary_data.get(method, {}).get(m, 0.0) for m in models]
        vals = [v * secondary_scale for v in raw]
        ax_marker.plot(
            xs, vals,
            marker=MARKER_STYLES[method], linestyle="none",
            color=MARKER_COLORS[method], markersize=marker_size,
            markeredgecolor="#222222", markeredgewidth=0.8, zorder=4,
        )

    BK = "black"
    ax_bar.set_ylabel(ylabel_left, fontsize=33, color=BK)
    ax_bar.set_xlabel("Model", fontsize=33, color=BK)
    ax_bar.set_xticks(gc)
    ax_bar.set_xticklabels(
        [_cap(m) for m in models], fontsize=27, rotation=25, ha="right",
        color=BK,
    )
    ax_bar.tick_params(axis="y", labelsize=27, colors=BK)
    ax_bar.grid(axis="y", linestyle="--", linewidth=0.4, color="#AAAAAA", alpha=0.4, zorder=0)
    ax_bar.set_axisbelow(True)
    for sp in ax_bar.spines.values():
        sp.set_visible(True)
        sp.set_color("#444444")
        sp.set_linewidth(0.7)

    ax_marker.set_ylabel(ylabel_right, fontsize=33, color=BK)
    ax_marker.tick_params(axis="y", labelsize=27, colors=BK)
    ax_marker.spines["right"].set_color("#444444")

    ax_bar.set_title(subtitle, fontsize=36, pad=14, color=BK)


def main() -> None:
    args = parse_args()
    for p in (args.train_csv, args.infer_csv):
        if not p.resolve().is_file():
            raise SystemExit(f"CSV not found: {p}")

    tr_models, tr_mem_d, tr_lat_d = load_overhead_summary(
        args.train_csv.resolve(), secondary_col="train_step_milliseconds")
    in_models, in_mem_d, in_tp_d = load_overhead_summary(
        args.infer_csv.resolve(), secondary_col="throughput_tps")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 30,
        "axes.linewidth": 0.8,
        "hatch.linewidth": 1.5,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(24, 9))
    ax_l2 = ax_l.twinx()
    ax_r2 = ax_r.twinx()

    draw_panel(
        ax_l, ax_l2,
        tr_models, tr_mem_d, tr_lat_d,
        "GPU Memory (GiB)", "Latency (s/step)",
        "(a) Training",
        secondary_scale=1e-3,
    )
    draw_panel(
        ax_r, ax_r2,
        in_models, in_mem_d, in_tp_d,
        "GPU Memory (GiB)", "Throughput (K token/s)",
        "(b) Inference",
        secondary_scale=1e-3,
    )

    marker_size = 14
    bar_handles = [
        Patch(facecolor=BAR_COLORS[m], edgecolor="#333333", linewidth=0.5,
              hatch=BAR_HATCHES[m], label=f"{METHOD_LABELS[m]} (Memory)")
        for m in METHOD_ORDER
    ]
    marker_handles = [
        Line2D([0], [0], marker=MARKER_STYLES[m], color="none",
               markerfacecolor=MARKER_COLORS[m], markeredgecolor="#222222",
               markeredgewidth=0.8, markersize=marker_size,
               label=f"{METHOD_LABELS[m]} (Lat./Thpt.)")
        for m in METHOD_ORDER
    ]

    leg = fig.legend(
        handles=bar_handles + marker_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.99),
        fontsize=26, ncol=4,
        frameon=True, fancybox=False, framealpha=0.95,
        edgecolor="#BBBBBB", facecolor="#F8F8F8",
        columnspacing=1.2, handletextpad=0.5, handlelength=1.8,
    )
    for t in leg.get_texts():
        t.set_color("black")

    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("=== Training ===")
    print("model,method,memory_gib,latency_ms")
    for model in tr_models:
        for method in METHOD_ORDER:
            mem = tr_mem_d.get(method, {}).get(model, 0.0)
            lat = tr_lat_d.get(method, {}).get(model, 0.0)
            print(f"{model},{method},{mem:.2f},{lat:.2f}")
    print("=== Inference ===")
    print("model,method,memory_gib,throughput_token_s")
    for model in in_models:
        for method in METHOD_ORDER:
            mem = in_mem_d.get(method, {}).get(model, 0.0)
            tp = in_tp_d.get(method, {}).get(model, 0.0)
            print(f"{model},{method},{mem:.2f},{tp:.2f}")
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
