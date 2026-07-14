#!/usr/bin/env bash
# One-step generator for Figure 9 — AMD (MI100) training fault-recovery breakdown.
# Reads the fault-recovery JSON logs and writes the stacked-bar chart (png+pdf).
#
# Usage:
#   bash make_figure9.sh
#   PYTHON_BIN=/root/miniconda3/bin/python bash make_figure9.sh   # if matplotlib not in default python
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"                 # AMD_IPC/plot
ROOT="${LAB1_REPORT_ROOT:-${HERE}/../experiment_logs/report1}"       # {ds_native,checkfreq,memory,ipc}/*.json
OUT="${OUT_DIR:-${HERE}/figures}"
PY="${PYTHON_BIN:-python3}"
mkdir -p "$OUT"

"$PY" -c "import matplotlib" 2>/dev/null || {
  echo "ERROR: '$PY' has no matplotlib. Re-run with PYTHON_BIN=<python-with-matplotlib> (gpm on the AMD node, or /root/miniconda3/bin/python on the NVIDIA host)."; exit 3; }

"$PY" "$HERE/update_plot_fault_recovery_stacked.py" \
  --lab1-report-root "$ROOT" \
  --output-prefix "$OUT/update_lab1_fault_recovery_stacked"

echo "[done] Figure 9 written to:"
ls -1 "$OUT"/update_lab1_fault_recovery_stacked.png "$OUT"/update_lab1_fault_recovery_stacked.pdf
