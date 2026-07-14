#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =====================================================================
# Top-level vLLM runner: e2e model suite + ShareGPT overhead.
#
# Usage:
#   ./run_vllm.sh [single|multi|all]
#
#   single       single-GPU only
#   multi        multi-GPU only
#   all  (default) single-GPU then multi-GPU
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
  bash "${SCRIPT_DIR}/e2e/run_e2e_vllm_model_suite.sh" "${m}"
  bash "${SCRIPT_DIR}/overhead/run_vllm_sharegpt_overhead_models.sh" "${m}"
}

if [[ "${MODE}" == "all" ]]; then
  run_mode single
  run_mode multi
else
  run_mode "${MODE}"
fi
