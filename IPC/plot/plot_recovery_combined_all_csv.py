#!/usr/bin/env python3
"""Side-by-side stacked bars: Training fault recovery (left) + Inference recovery (right)."""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR / "report1"
DEFAULT_OUTPUT_PREFIX = SCRIPT_DIR / "recovery_combined_all"

# ── Training ──
TRAIN_METHOD_ORDER = ["ds_native", "checkfreq", "memory", "ipc"]
TRAIN_METHOD_LABELS = {
    "ds_native": "DeepSpeed",
    "checkfreq": "CheckFreq",
    "memory": "In-Memory",
    "ipc": "ANCHOR",
}
TRAIN_METHOD_HATCHES = {
    "ds_native": "",
    "checkfreq": "//",
    "memory": "\\\\",
    "ipc": "xx",
}
TRAIN_COMPONENT_ORDER = ("init", "load", "other")
TRAIN_COMPONENT_COLORS = {
    "init": "#739EE4",
    "load": "#AFD2F8",
    "other": "#E0EFFE",
}
TRAIN_COMPONENT_LABELS = {
    "init": "Engine Init",
    "load": "Ckpt Load",
    "other": "Other",
}

_TRAIN_METHOD_ALIASES: dict[str, str] = {}
for _k, _v in {
    "deepspeed": "ds_native", "ds native": "ds_native", "ds_native": "ds_native",
    "deep speed": "ds_native",
    "checkfreq": "checkfreq",
    "in memory": "memory", "in-memory": "memory", "memory": "memory", "inmem": "memory",
    "ipc": "ipc", "anchor": "ipc",
}.items():
    _TRAIN_METHOD_ALIASES[_k.lower()] = _v

_TRAIN_BREAKDOWN_ALIASES = {
    "total": "total", "engine init": "init", "engine_init": "init", "init": "init",
    "load": "load", "ckpt load": "load", "attach": "load", "other": "other",
    "remainder": "other",
}

# ── Inference ──
INFER_METHOD_ORDER = ["replay", "kv_memory", "ipc"]
INFER_METHOD_LABELS = {
    "replay": "Replay",
    "kv_memory": "KV Memory",
    "ipc": "ANCHOR",
}
INFER_METHOD_HATCHES = {
    "replay": "..",
    "kv_memory": "||",
    "ipc": "xx",
}
INFER_COMPONENT_ORDER = ("python_init", "engine_init", "other")
INFER_COMPONENT_COLORS = {
    "python_init": "#6DB57E",
    "engine_init": "#A8D8B0",
    "other": "#DEF0E0",
}
INFER_COMPONENT_LABELS = {
    "python_init": "Python Init",
    "engine_init": "Engine Init",
    "other": "Other",
}

_INFER_METHOD_ALIASES: dict[str, str] = {}
for _k, _v in {
    "replay": "replay",
    "kv memory": "kv_memory", "kv_memory": "kv_memory",
    "kvmemory": "kv_memory", "kv-memory": "kv_memory",
    "ipc": "ipc", "anchor": "ipc",
}.items():
    _INFER_METHOD_ALIASES[_k.lower()] = _v

_INFER_BREAKDOWN_ALIASES: dict[str, str] = {
    "total": "total",
    "python init": "python_init", "python_init": "python_init",
    "pythoninit": "python_init", "py init": "python_init",
    "engine init": "engine_init", "engine_init": "engine_init",
    "engineinit": "engine_init", "init": "engine_init",
    "other": "other", "remainder": "other",
}


def _cap(n: str) -> str:
    return n[0].upper() + n[1:] if n else n


def _norm(s: str, aliases: dict[str, str]) -> str:
    key = re.sub(r"\s+", " ", s.strip().lower().replace("_", " ").replace("-", " "))
    if key in aliases:
        return aliases[key]
    raise SystemExit(f"Unknown key in CSV: {s!r}")


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


def load_stacked_summary(
    path: Path,
    method_order: list[str],
    method_aliases: dict[str, str],
    component_cols: dict[str, str],
    total_col: str,
    tolerance: float,
) -> tuple[list[str], dict[str, dict[str, dict[str, float]]]]:
    """Read a tidy per-(method,config,model) CSV -> (ordered_model_keys, data[m][model][comp|total]).

    ``component_cols`` maps each stacked component name to its CSV column; ``total_col`` is the
    recovery-total column. Column key = model for single-GPU, ``"<model> TP<n>"`` for multi-GPU;
    single-GPU keys are ordered by model size, multi-GPU keys appended at the end.
    """
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    data: dict[str, dict[str, dict[str, float]]] = {}
    meta: dict[str, tuple[float, bool]] = {}

    def _num(row: dict, col: str) -> float:
        cell = (row.get(col) or "").strip()
        return float(cell) if cell else 0.0

    for row in rows:
        if not row.get("method"):
            continue
        method = _norm(row["method"], method_aliases)
        if method not in method_order:
            continue
        model = row["model"].strip().replace("_", ".")  # 'qwen3 1_7B' -> 'qwen3 1.7B'
        is_multi, degree = _config_info(row.get("config", ""))
        colkey = f"{model} TP{degree}" if is_multi else model
        d = data.setdefault(method, {}).setdefault(colkey, {})
        for comp, col in component_cols.items():
            d[comp] = _num(row, col)
        d["total"] = _num(row, total_col)
        meta[colkey] = (_model_size(model), is_multi)

    singles = sorted((k for k, v in meta.items() if not v[1]), key=lambda k: meta[k][0])
    multis = sorted((k for k, v in meta.items() if v[1]), key=lambda k: meta[k][0])
    models = singles + multis

    # Ensure every method has every model key with all components (fill gaps with 0).
    for method in method_order:
        dm = data.setdefault(method, {})
        for mk in models:
            dd = dm.setdefault(mk, {})
            for comp in component_cols:
                dd.setdefault(comp, 0.0)
            dd.setdefault("total", 0.0)
            s = sum(dd[comp] for comp in component_cols)
            if abs(s - dd["total"]) > tolerance:
                print(f"[warn] {method}/{mk}: sum={s:.4f} vs total={dd['total']:.4f}")

    return models, data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combined training+inference recovery stacked bars.")
    p.add_argument("--train-csv", type=Path,
                   default=SCRIPT_DIR.parent / "ds_script" / "ds_fault_summary.csv")
    p.add_argument("--infer-csv", type=Path,
                   default=SCRIPT_DIR.parent / "vllm_script" / "vllm_e2e_summary.csv")
    p.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--tolerance", type=float, default=0.15)
    return p.parse_args()


def draw_panel(
    ax, models, data,
    method_order, method_hatches,
    component_order, component_colors,
    ylabel, subtitle,
):
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
        bottoms = [0.0] * len(models)
        for comp in component_order:
            vals = [data[method][m][comp] for m in models]
            ax.bar(
                xs, vals, width=bar_width, bottom=bottoms,
                color=component_colors[comp], edgecolor="#333333", linewidth=0.6,
                hatch=method_hatches[method], alpha=0.88, zorder=2,
            )
            bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_ylabel(ylabel, fontsize=36, color=BK)
    ax.set_xlabel("Model", fontsize=36, color=BK)
    ax.set_xticks(gc)
    ax.set_xticklabels(
        [_cap(m) for m in models], fontsize=30, rotation=25, ha="right",
        color=BK,
    )
    ax.tick_params(axis="y", labelsize=27, colors=BK)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, color="#AAAAAA", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#444444")
        sp.set_linewidth(0.7)

    ax.set_title(subtitle, fontsize=36, pad=14, color=BK)


def main() -> None:
    args = parse_args()
    for p in (args.train_csv, args.infer_csv):
        if not p.resolve().is_file():
            raise SystemExit(f"CSV not found: {p}")

    tr_models, tr_data = load_stacked_summary(
        args.train_csv.resolve(), TRAIN_METHOD_ORDER, _TRAIN_METHOD_ALIASES,
        {"init": "engine_init_seconds", "load": "load_seconds", "other": "other_seconds"},
        "recovery_total_seconds", args.tolerance,
    )
    in_models, in_data = load_stacked_summary(
        args.infer_csv.resolve(), INFER_METHOD_ORDER, _INFER_METHOD_ALIASES,
        {"python_init": "connector_python_startup_seconds",
         "engine_init": "connector_engine_startup_seconds",
         "other": "other_seconds"},
        "fault_to_recovery_ready_seconds", args.tolerance,
    )

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerPatch
    import matplotlib.patches as mpatches

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

    draw_panel(
        ax_l, tr_models, tr_data,
        TRAIN_METHOD_ORDER, TRAIN_METHOD_HATCHES,
        TRAIN_COMPONENT_ORDER, TRAIN_COMPONENT_COLORS,
        "Fault Recovery Time (s)", "(a) Training",
    )
    draw_panel(
        ax_r, in_models, in_data,
        INFER_METHOD_ORDER, INFER_METHOD_HATCHES,
        INFER_COMPONENT_ORDER, INFER_COMPONENT_COLORS,
        "Inference Recovery Time (s)", "(b) Inference",
    )

    # ── Build legend: 3 rows (Methods / Training Breakdown / Inference Breakdown) ──
    class _InvisHandler(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            p = mpatches.FancyBboxPatch(
                (0, 0), 0, 0, boxstyle="square,pad=0",
                facecolor="none", edgecolor="none", transform=trans,
            )
            return [p]

    invis = Patch(facecolor="none", edgecolor="none")
    pad = Patch(facecolor="none", edgecolor="none")

    all_method_items: list[tuple[Any, str]] = []
    seen_labels: set[str] = set()
    for order, labels, hatches in [
        (TRAIN_METHOD_ORDER, TRAIN_METHOD_LABELS, TRAIN_METHOD_HATCHES),
        (INFER_METHOD_ORDER, INFER_METHOD_LABELS, INFER_METHOD_HATCHES),
    ]:
        for m in order:
            lbl = labels[m]
            if lbl not in seen_labels:
                seen_labels.add(lbl)
                all_method_items.append((
                    Patch(facecolor="#999999", edgecolor="#333333", linewidth=0.6,
                          hatch=hatches[m]),
                    lbl,
                ))

    tr_break_items = [
        (Patch(facecolor=TRAIN_COMPONENT_COLORS[c], edgecolor="#333333", linewidth=0.6),
         TRAIN_COMPONENT_LABELS[c])
        for c in TRAIN_COMPONENT_ORDER
    ]
    in_break_items = [
        (Patch(facecolor=INFER_COMPONENT_COLORS[c], edgecolor="#333333", linewidth=0.6),
         INFER_COMPONENT_LABELS[c])
        for c in INFER_COMPONENT_ORDER
    ]

    ncol = max(1 + len(all_method_items), 1 + len(tr_break_items), 1 + len(in_break_items))

    def _pad_row(items: list[tuple[Any, str]], title: str) -> tuple[list[Any], list[str]]:
        h = [invis] + [it[0] for it in items]
        l = [title] + [it[1] for it in items]
        while len(h) < ncol:
            h.append(pad)
            l.append("")
        return h, l

    r1_h, r1_l = _pad_row(all_method_items, "Method:")
    r2_h, r2_l = _pad_row(tr_break_items, "Train:")
    r3_h, r3_l = _pad_row(in_break_items, "Infer:")

    rows = [(r1_h, r1_l), (r2_h, r2_l), (r3_h, r3_l)]
    n_rows = len(rows)
    interleaved_h: list[Any] = []
    interleaved_l: list[str] = []
    for col_idx in range(ncol):
        for row_idx in range(n_rows):
            interleaved_h.append(rows[row_idx][0][col_idx])
            interleaved_l.append(rows[row_idx][1][col_idx])

    handler_map = {invis: _InvisHandler(), pad: _InvisHandler()}

    leg = fig.legend(
        handles=interleaved_h, labels=interleaved_l,
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=ncol, fontsize=26,
        frameon=True, fancybox=False, framealpha=0.95,
        edgecolor="#BBBBBB", facecolor="#F8F8F8",
        handlelength=1.6, handleheight=1.0,
        labelspacing=0.35, handletextpad=0.4,
        borderpad=0.5, columnspacing=1.0,
        handler_map=handler_map,
    )
    for t in leg.get_texts():
        t.set_color("black")

    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.subplots_adjust(top=0.80)  # axes unchanged; legend sits above via a taller (tight) canvas
    fig.savefig(output_prefix.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("=== Training ===")
    print("model,method," + ",".join(TRAIN_COMPONENT_ORDER) + ",total")
    for model in tr_models:
        for method in TRAIN_METHOD_ORDER:
            d = tr_data[method][model]
            vals = ",".join(f"{d[c]:.4f}" for c in TRAIN_COMPONENT_ORDER)
            print(f"{model},{method},{vals},{d['total']:.4f}")
    print("=== Inference ===")
    print("model,method," + ",".join(INFER_COMPONENT_ORDER) + ",total")
    for model in in_models:
        for method in INFER_METHOD_ORDER:
            d = in_data[method][model]
            vals = ",".join(f"{d[c]:.4f}" for c in INFER_COMPONENT_ORDER)
            print(f"{model},{method},{vals},{d['total']:.4f}")
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
