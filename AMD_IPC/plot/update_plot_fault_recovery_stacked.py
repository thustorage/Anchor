#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LAB1_REPORT_ROOT = SCRIPT_DIR / "report1"
DEFAULT_OUTPUT_PREFIX = DEFAULT_LAB1_REPORT_ROOT / "figures" / "update_lab1_fault_recovery_stacked"

MODEL_DISPLAY_NAMES = {
    "qwen3_0_6b": "Qwen3 0.6B",
    "llama3_2_1b": "Llama3.2 1.0B",
    "qwen2_5_1_5b": "Qwen2.5 1.5B",
}

MODEL_GROUP_ORDER = {
    "qwen3_0_6b": 1,
    "llama3_2_1b": 2,
    "qwen2_5_1_5b": 3,
}

SCOPE_ORDER = ["lab1", ]
SCOPE_REPORT_DIRS = {
    "lab1": {
        "ds_native": "ds_native",
        "checkfreq": "checkfreq",
        "memory": "memory",
        "ipc": "ipc",
    },
}

METHOD_ORDER = ["ds_native", "checkfreq", "memory", "ipc"]
METHOD_LABELS = {
    "ds_native": "DS Native",
    "checkfreq": "CheckFreq",
    "memory": "In-Memory",
    "ipc": "ANCHOR",
}
COMPONENT_COLORS = {
    "init":  "#739EE4",
    "load":  "#AFD2F8",
    "other": "#E9F5FE",
}
METHOD_HATCHES = {
    "ds_native":  "",
    "checkfreq":  "//",
    "memory":     "\\\\",
    "ipc":        "xx",
}
COMPONENT_LABELS = {
    "init": "Engine Init",
    "load": "Ckpt Load / Attach",
    "other": "Other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load lab1/lab2 recovery reports and draw grouped stacked bars for "
            "fault recovery time across DS Native, CheckFreq, In-Memory, and ANCHOR."
        )
    )
    parser.add_argument(
        "--lab1-report-root",
        type=Path,
        default=DEFAULT_LAB1_REPORT_ROOT,
        help="Root directory containing lab1 report1/{ds_native,checkfreq,memory,ipc}.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output path prefix without suffix; writes .png and .pdf.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model stem order.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate total recovery time above each bar.",
    )
    parser.add_argument(
        "--no-annotate",
        dest="annotate",
        action="store_false",
        help="Disable total recovery time annotations above bars.",
    )
    parser.set_defaults(annotate=False)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_scope_reports(
    report_root: Path,
    report_dirs: dict[str, str],
) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for method in METHOD_ORDER:
        method_dir = report_root / report_dirs[method]
        if not method_dir.is_dir():
            raise SystemExit(f"Report directory not found: {method_dir}")
        method_reports: dict[str, dict[str, Any]] = {}
        for path in sorted(method_dir.glob("*.json")):
            if path.name.endswith("_state.json"):
                continue
            method_reports[path.stem] = load_json(path)
        if not method_reports:
            raise SystemExit(f"No report JSON files found in {method_dir}")
        loaded[method] = method_reports
    return loaded


def load_reports(args: argparse.Namespace) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "lab1": load_scope_reports(args.lab1_report_root.resolve(), SCOPE_REPORT_DIRS["lab1"]),
    }


def parse_billions_from_label(label: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*B\b", label, flags=re.IGNORECASE)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def display_label(loaded: dict[str, dict[str, dict[str, Any]]], model: str) -> str:
    mapped = MODEL_DISPLAY_NAMES.get(model)
    if mapped:
        return mapped
    for scope in SCOPE_ORDER:
        for method in METHOD_ORDER:
            report = loaded[scope][method].get(model)
            if report:
                return str(report.get("model_label") or model)
    return model


def resolve_entries(
    loaded: dict[str, dict[str, dict[str, Any]]],
    requested: list[str] | None,
) -> list[tuple[str, str]]:
    scope_common_models: dict[str, set[str]] = {}
    for scope in SCOPE_ORDER:
        available_sets = [set(loaded[scope][method].keys()) for method in METHOD_ORDER]
        common = set.intersection(*available_sets)
        if not common:
            raise SystemExit(f"No common model reports were found across all methods in {scope}.")
        scope_common_models[scope] = common

    if requested:
        entries: list[tuple[str, str]] = []
        missing: list[str] = []
        for model in requested:
            found = False
            for scope in SCOPE_ORDER:
                if model in scope_common_models[scope]:
                    entries.append((scope, model))
                    found = True
            if not found:
                missing.append(model)
        if missing:
            raise SystemExit(
                "Requested model reports are missing in both lab1 and lab2: "
                + ", ".join(missing)
            )
        return entries

    all_models = set().union(*scope_common_models.values())

    def model_sort_key(model: str) -> tuple[int, float, str]:
        label = display_label(loaded, model)
        manual_order = MODEL_GROUP_ORDER.get(model)
        if manual_order is not None:
            return (0, float(manual_order), label.lower())
        size_b = parse_billions_from_label(label)
        if size_b is None:
            return (1, float("inf"), label.lower())
        return (1, size_b, label.lower())

    entries = []
    for model in sorted(all_models, key=model_sort_key):
        for scope in SCOPE_ORDER:
            if model in scope_common_models[scope]:
                entries.append((scope, model))
    return entries


def recovery_total_seconds(report: dict[str, Any]) -> float:
    value = report.get("recovery_total_seconds")
    if value is None:
        value = report.get("fault_to_recovery_ready_seconds",
                           report.get("restart_to_recovery_ready_seconds", 0.0))
    return float(value or 0.0)


def recovery_breakdown(method: str, report: dict[str, Any]) -> dict[str, float]:
    total = recovery_total_seconds(report)
    init = float(report.get("engine_init_seconds", 0.0) or 0.0)
    load = float(report.get("load_seconds", 0.0) or 0.0)
    other = report.get("other_seconds")
    other = float(other) if other is not None else max(0.0, total - init - load)
    return {
        "init": init,
        "load": load,
        "other": other,
        "total": total,
    }


def print_summary(
    loaded: dict[str, dict[str, dict[str, Any]]],
    entries: list[tuple[str, str]],
) -> None:
    print("scope,model,method,init_seconds,load_or_attach_seconds,other_seconds,total_seconds")
    for scope, model in entries:
        for method in METHOD_ORDER:
            breakdown = recovery_breakdown(method, loaded[scope][method][model])
            print(
                f"{scope},{model},{method},"
                f"{breakdown['init']:.6f},"
                f"{breakdown['load']:.6f},"
                f"{breakdown['other']:.6f},"
                f"{breakdown['total']:.6f}"
            )


def annotate_total(ax: Any, x_pos: float, total: float) -> None:
    ax.annotate(
        f"{total:.1f}",
        xy=(x_pos, total),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333333",
    )


def main() -> None:
    """Load recovery reports and render the grouped stacked-bar figure."""
    args = parse_args()

    loaded = load_reports(args)
    entries = resolve_entries(loaded, args.models)
    labels = [display_label(loaded, model) for _, model in entries]

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise SystemExit(f"matplotlib is required: {exc}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 14,
        "axes.linewidth": 0.8,
        "hatch.linewidth": 1.5,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })

    bar_width = 0.18
    group_centers = list(range(len(entries)))
    method_offsets = {
        method: (idx - (len(METHOD_ORDER) - 1) / 2.0) * bar_width
        for idx, method in enumerate(METHOD_ORDER)
    }

    fig_width = max(10.0, len(entries) * 1.85)
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))

    for method in METHOD_ORDER:
        xs = [center + method_offsets[method] for center in group_centers]
        bottoms = [0.0 for _ in entries]
        for component in ("init", "load", "other"):
            values = [
                recovery_breakdown(method, loaded[scope][method][model])[component]
                for scope, model in entries
            ]
            ax.bar(
                xs,
                values,
                width=bar_width,
                bottom=bottoms,
                color=COMPONENT_COLORS[component],
                edgecolor="#333333",
                linewidth=0.6,
                hatch=METHOD_HATCHES[method],
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

        if args.annotate:
            totals = [
                recovery_breakdown(method, loaded[scope][method][model])["total"]
                for scope, model in entries
            ]
            for x_pos, total in zip(xs, totals):
                annotate_total(ax, x_pos, total)

    BK = "black"
    if args.title:
        fig.suptitle(args.title, y=0.99, fontsize=13, color=BK)
    ax.set_xlabel("Model", fontsize=16, color=BK)
    ax.set_ylabel("Fault Recovery Time (s)", fontsize=16, color=BK)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels, fontsize=14, color=BK)
    ax.tick_params(axis="y", labelsize=14, colors=BK)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#AAAAAA", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#444444")
        spine.set_linewidth(0.8)

    method_handles = [
        Patch(facecolor="#999999", edgecolor="#333333", linewidth=0.6,
              hatch=METHOD_HATCHES[m])
        for m in METHOD_ORDER
    ]
    method_labels = [METHOD_LABELS[m] for m in METHOD_ORDER]

    breakdown_handles = [
        Patch(facecolor=COMPONENT_COLORS[c], edgecolor="#333333", linewidth=0.6)
        for c in ("init", "load", "other")
    ]
    breakdown_labels = [COMPONENT_LABELS[c] for c in ("init", "load", "other")]

    from matplotlib.legend_handler import HandlerPatch
    import matplotlib.patches as mpatches

    class _InvisHandler(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            p = mpatches.FancyBboxPatch((0, 0), 0, 0,
                                        boxstyle="square,pad=0",
                                        facecolor="none", edgecolor="none",
                                        transform=trans)
            return [p]

    title_patch = Patch(facecolor="none", edgecolor="none")

    all_handles = [title_patch] + method_handles + [title_patch] + breakdown_handles
    all_labels = (["Method:"] + method_labels
                  + ["Breakdown:"] + breakdown_labels)

    n_method_cols = 1 + len(METHOD_ORDER)
    n_break_cols = 1 + 3
    ncol = max(n_method_cols, n_break_cols)

    pad_handle = Patch(facecolor="none", edgecolor="none")

    row1_h = [title_patch] + method_handles
    row1_l = ["Method:"] + method_labels
    row2_h = [title_patch] + breakdown_handles
    row2_l = ["Breakdown:"] + breakdown_labels

    while len(row1_h) < ncol:
        row1_h.append(pad_handle)
        row1_l.append("")
    while len(row2_h) < ncol:
        row2_h.append(pad_handle)
        row2_l.append("")

    interleaved_h = []
    interleaved_l = []
    for h1, l1, h2, l2 in zip(row1_h, row1_l, row2_h, row2_l):
        interleaved_h.extend([h1, h2])
        interleaved_l.extend([l1, l2])

    leg = fig.legend(
        handles=interleaved_h,
        labels=interleaved_l,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=ncol,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#BBBBBB",
        facecolor="#F8F8F8",
        fontsize=13,
        handlelength=1.4,
        handleheight=1.0,
        labelspacing=0.4,
        handletextpad=0.5,
        borderpad=0.6,
        columnspacing=1.2,
        handler_map={title_patch: _InvisHandler(), pad_handle: _InvisHandler()},
    )
    for text in leg.get_texts():
        text.set_color("black")

    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print_summary(loaded, entries)
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
