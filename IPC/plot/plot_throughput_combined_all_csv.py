#!/usr/bin/env python3
"""Side-by-side grouped bar charts: Training throughput (left) + Inference throughput (right)."""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR / "report1"
DEFAULT_OUTPUT_PREFIX = SCRIPT_DIR / "throughput_combined_all"

TRAIN_METHOD_ORDER = ["ds_native", "checkfreq", "memory", "ipc"]
TRAIN_METHOD_LABELS = {
    "ds_native": "DeepSpeed",
    "checkfreq": "CheckFreq",
    "memory": "In-Memory",
    "ipc": "ANCHOR",
}
TRAIN_METHOD_COLORS = {
    "ds_native": "#4C72B0",
    "checkfreq": "#55A868",
    "memory": "#DD8452",
    "ipc": "#C44E52",
}
TRAIN_METHOD_HATCHES = {
    "ds_native": "",
    "checkfreq": "//",
    "memory": "\\\\",
    "ipc": "xx",
}

INFER_METHOD_ORDER = ["replay", "kv_memory", "ipc"]
INFER_METHOD_LABELS = {
    "replay": "Replay",
    "kv_memory": "KV Memory",
    "ipc": "ANCHOR",
}
INFER_METHOD_COLORS = {
    "replay": "#8172B3",
    "kv_memory": "#64B5CD",
    "ipc": "#C44E52",
}
INFER_METHOD_HATCHES = {
    "replay": "..",
    "kv_memory": "||",
    "ipc": "xx",
}

_TRAIN_ALIASES: dict[str, str] = {}
for _k, _v in {
    "deepspeed": "ds_native", "ds native": "ds_native", "ds_native": "ds_native",
    "checkfreq": "checkfreq",
    "in memory": "memory", "in-memory": "memory", "memory": "memory",
    "ipc": "ipc", "anchor": "ipc",
}.items():
    _TRAIN_ALIASES[_k.lower()] = _v

_INFER_ALIASES: dict[str, str] = {}
for _k, _v in {
    "replay": "replay",
    "kv memory": "kv_memory", "kv_memory": "kv_memory",
    "kvmemory": "kv_memory", "kv-memory": "kv_memory",
    "ipc": "ipc", "anchor": "ipc",
}.items():
    _INFER_ALIASES[_k.lower()] = _v


def _cap(n: str) -> str:
    return n[0].upper() + n[1:] if n else n


def _norm(s: str, aliases: dict[str, str]) -> str:
    key = re.sub(r"\s+", " ", s.strip().lower().replace("_", " ").replace("-", " "))
    if key in aliases:
        return aliases[key]
    raise SystemExit(f"Unknown method in CSV: {s!r}")


def _model_size(label: str) -> float:
    """Parameter count for sorting. Handles '1_7B' (== 1.7B) written with an underscore."""
    norm = re.sub(r"(\d)_(\d)", r"\1.\2", str(label))
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", norm)
    return float(m.group(1)) if m else float("inf")


def _config_info(config: str) -> tuple[bool, int]:
    """Return (is_multi_gpu, degree). Single = {'single','1gpu'} or degree 1. Multi configs
    without a number default to degree 2 (all multi-GPU runs in this artifact used 2 GPUs)."""
    c = config.strip().lower()
    m = re.search(r"\d+", c)
    n = int(m.group()) if m else None
    if c in ("single", "1gpu") or n == 1:
        return False, 1
    return True, (n or 2)


def load_metric_summary(
    path: Path, method_order: list[str], aliases: dict[str, str], value_col: str,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Read a tidy per-(method,config,model) CSV -> (ordered_model_keys, data[method][model]).

    Column key = model for single-GPU, ``"<model> TP<n>"`` for multi-GPU; single-GPU keys
    ordered by model size, multi-GPU keys appended at the end.
    """
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    data: dict[str, dict[str, float]] = {}
    meta: dict[str, tuple[float, bool]] = {}
    for row in rows:
        if not row.get("method"):
            continue
        method = _norm(row["method"], aliases)
        if method not in method_order:
            continue
        model = row["model"].strip().replace("_", ".")  # 'qwen3 1_7B' -> 'qwen3 1.7B'
        is_multi, degree = _config_info(row.get("config", ""))
        colkey = f"{model} TP{degree}" if is_multi else model
        cell = (row.get(value_col) or "").strip()
        data.setdefault(method, {})[colkey] = float(cell) if cell else 0.0
        meta[colkey] = (_model_size(model), is_multi)

    singles = sorted((k for k, v in meta.items() if not v[1]), key=lambda k: meta[k][0])
    multis = sorted((k for k, v in meta.items() if v[1]), key=lambda k: meta[k][0])
    models = singles + multis
    for method in method_order:  # fill gaps so draw_panel never KeyErrors
        dm = data.setdefault(method, {})
        for mk in models:
            dm.setdefault(mk, 0.0)
    return models, data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combined training+inference throughput (side-by-side).")
    p.add_argument("--train-csv", type=Path,
                   default=SCRIPT_DIR.parent / "ds_script" / "ds_e2e_summary.csv")
    p.add_argument("--infer-csv", type=Path,
                   default=SCRIPT_DIR.parent / "vllm_script" / "vllm_e2e_summary.csv")
    p.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def draw_panel(
    ax, models, data,
    method_order, method_labels, method_colors, method_hatches,
    ylabel, subtitle, ytick_step,
):
    from matplotlib.ticker import MultipleLocator

    n_methods = len(method_order)
    bar_width = min(0.22, 0.8 / n_methods)
    gc = list(range(len(models)))
    offsets = {
        m: (i - (n_methods - 1) / 2.0) * bar_width
        for i, m in enumerate(method_order)
    }

    BK = "black"
    for method in method_order:
        xs = [c + offsets[method] for c in gc]
        heights = [data.get(method, {}).get(m, 0.0) for m in models]
        ax.bar(
            xs, heights, width=bar_width,
            color=method_colors[method], edgecolor="#333333", linewidth=0.6,
            hatch=method_hatches[method], alpha=0.88, zorder=2,
        )

    ax.yaxis.set_major_locator(MultipleLocator(ytick_step))
    ax.set_ylabel(ylabel, fontsize=28, color=BK, labelpad=14)
    ax.set_xlabel("Model", fontsize=28, color=BK)
    ax.set_xticks(gc)
    ax.set_xticklabels(
        [_cap(m) for m in models], fontsize=24, rotation=25, ha="right",
        color=BK,
    )
    ax.tick_params(axis="y", labelsize=22, colors=BK)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, color="#AAAAAA", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#444444")
        sp.set_linewidth(0.7)

    ax.set_title(subtitle, fontsize=28, pad=14, color=BK)


def main() -> None:
    args = parse_args()
    for p in (args.train_csv, args.infer_csv):
        if not p.resolve().is_file():
            raise SystemExit(f"CSV not found: {p}")

    tr_models, tr_data = load_metric_summary(
        args.train_csv.resolve(), TRAIN_METHOD_ORDER, _TRAIN_ALIASES, "average_steps_per_second")
    in_models, in_data = load_metric_summary(
        args.infer_csv.resolve(), INFER_METHOD_ORDER, _INFER_ALIASES, "ms_per_request")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 24,
        "axes.linewidth": 0.8,
        "hatch.linewidth": 1.5,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(24, 7.5))

    draw_panel(
        ax_l, tr_models, tr_data,
        TRAIN_METHOD_ORDER, TRAIN_METHOD_LABELS,
        TRAIN_METHOD_COLORS, TRAIN_METHOD_HATCHES,
        "Training Throughput (steps/s)",
        "(a) Training", 0.5,
    )
    draw_panel(
        ax_r, in_models, in_data,
        INFER_METHOD_ORDER, INFER_METHOD_LABELS,
        INFER_METHOD_COLORS, INFER_METHOD_HATCHES,
        "Inference Latency (ms/req)",
        "(b) Inference", 100,
    )

    all_labels_seen: dict[str, Patch] = {}
    for order, colors, hatches, labels in [
        (TRAIN_METHOD_ORDER, TRAIN_METHOD_COLORS, TRAIN_METHOD_HATCHES, TRAIN_METHOD_LABELS),
        (INFER_METHOD_ORDER, INFER_METHOD_COLORS, INFER_METHOD_HATCHES, INFER_METHOD_LABELS),
    ]:
        for m in order:
            lbl = labels[m]
            if lbl not in all_labels_seen:
                all_labels_seen[lbl] = Patch(
                    facecolor=colors[m], edgecolor="#333333", linewidth=0.6,
                    hatch=hatches[m], label=lbl,
                )

    leg = fig.legend(
        handles=list(all_labels_seen.values()),
        loc="lower center", bbox_to_anchor=(0.5, 0.99),
        fontsize=28, ncol=len(all_labels_seen),
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
    print("model,method,steps_per_second")
    for model in tr_models:
        for method in TRAIN_METHOD_ORDER:
            v = tr_data.get(method, {}).get(model, 0.0)
            print(f"{model},{method},{v:.6f}")
    print("=== Inference ===")
    print("model,method,ms_per_req")
    for model in in_models:
        for method in INFER_METHOD_ORDER:
            v = in_data.get(method, {}).get(model, 0.0)
            print(f"{model},{method},{v:.2f}")
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
