#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =====================================================================
# Top-level DeepSpeed runner: overhead + fault (4 methods) + e2e suite.
#
# Usage:
#   ./run_ds.sh [single|multi|all]
#
#   single       single-GPU only
#   multi        multi-GPU only
#   all  (default) single-GPU then multi-GPU
#
# Each sub-script / suite takes the same single|multi mode arg. The e2e
# step calls only the suite (run_e2e_ds_model_suite.sh), which itself
# drives warmup + all four fault methods for the chosen mode.
# =====================================================================

MODE="${1:-all}"
case "${MODE}" in
  single|multi|all) ;;
  *)
    echo "Unknown mode '${MODE}'. Usage: $0 [single|multi|all]" >&2
    exit 1
    ;;
esac

run_mode() {
  local m="$1"

  # --- overhead ---
  bash "${SCRIPT_DIR}/overhead/run_train_overhead_three_models.sh" "${m}"

  # --- fault (4 methods) ---
  bash "${SCRIPT_DIR}/fault/run_ds_hf_three_models.sh" "${m}"
  bash "${SCRIPT_DIR}/fault/run_checkfreq_hf_three_models.sh" "${m}"
  bash "${SCRIPT_DIR}/fault/run_memory_hf_three_models.sh" "${m}"
  bash "${SCRIPT_DIR}/fault/run_ipc_hf_three_models.sh" "${m}"

  # --- e2e (suite drives warmup + all four methods) ---
  bash "${SCRIPT_DIR}/e2e/run_e2e_ds_model_suite.sh" "${m}"
}

if [[ "${MODE}" == "all" ]]; then
  run_mode single
  run_mode multi
else
  run_mode "${MODE}"
fi
