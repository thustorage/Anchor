#!/usr/bin/env python3
"""Reproduce Figure 9 — AMD (MI100) training fault-recovery breakdown — in one step.

Runs the four training baselines (DS Native, CheckFreq, In-Memory, ANCHOR/IPC) over
three models, writing the per-run JSON logs, then plots the stacked-bar figure.

    python reproduce_figure9.py                 # full: run experiments -> logs -> Figure 9
    python reproduce_figure9.py --plot-only     # just (re)draw Figure 9 from existing logs
    python reproduce_figure9.py --methods ipc   # a subset of {ds_native,checkfreq,memory,ipc}

Run on the AMD node inside the `gpm` env (ROCm torch). Requires the models under
/root/mhy/model and the wikitext dataset under /root/mhy/datasets (the run scripts
reference those paths). Selects the GPU with --gpu (HIP_VISIBLE_DEVICES).
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS_SCRIPT = HERE / "ds_script"
IPC_TOOLS = HERE / "ipc_tools"
PLOT = HERE / "plot"
LOGS = HERE / "experiment_logs" / "report1"
FIG_PREFIX = PLOT / "figures" / "update_lab1_fault_recovery_stacked"

RUN_SCRIPTS = {
    "ds_native": "run_ds_hf_three_models.sh",
    "checkfreq": "run_checkfreq_hf_three_models.sh",
    "memory":    "run_memory_hf_three_models.sh",
    "ipc":       "run_ipc_hf_three_models.sh",
}
METHOD_ORDER = ["ds_native", "checkfreq", "memory", "ipc"]


def run(cmd: list[str], env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    """Run the selected fault-recovery baselines and plot Figure 9."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python-bin", default=sys.executable,
                    help="Interpreter for the benches (default: this one; use the gpm python).")
    ap.add_argument("--gpu", default="0", help="HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES (default 0).")
    ap.add_argument("--methods", nargs="*", default=METHOD_ORDER,
                    help="Subset of ds_native checkfreq memory ipc.")
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip the experiments; just draw Figure 9 from existing logs.")
    args = ap.parse_args()

    if not args.plot_only:
        base = os.environ.copy()
        base["PYTHON_BIN"] = args.python_bin
        base["GPU_ID"] = args.gpu
        base["PYTHONPATH"] = os.pathsep.join(
            [str(IPC_TOOLS), str(DS_SCRIPT), base.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        for method in args.methods:
            if method not in RUN_SCRIPTS:
                raise SystemExit(f"unknown method: {method}")
            env = dict(base, BASELINE_NAME=method, REPORT_DIR=str(LOGS / method))
            (LOGS / method).mkdir(parents=True, exist_ok=True)
            print(f"\n===== [{method}] running 3 models -> {LOGS / method} =====", flush=True)
            run(["bash", str(DS_SCRIPT / RUN_SCRIPTS[method])], env=env)

    FIG_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    print("\n===== plotting Figure 9 =====", flush=True)
    run([args.python_bin, str(PLOT / "update_plot_fault_recovery_stacked.py"),
         "--lab1-report-root", str(LOGS),
         "--output-prefix", str(FIG_PREFIX)])
    print(f"\nFigure 9 -> {FIG_PREFIX}.png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
