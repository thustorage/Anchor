#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Anchor artifact driver: run the DeepSpeed (training) and vLLM (inference) suites.
# IPC_TOOL_VERBOSE=1 makes the Anchor daemons print startup/handshake diagnostics
# (PID + socket), which also helps distinguish per-GPU daemons in multi-GPU runs.
#
# Usage:
#   ./run.sh [single|multi|all]
#
#   single       single-GPU only
#   multi        multi-GPU only
#   all  (default) single-GPU then multi-GPU
#
# The mode is passed through to run_ds.sh / run_vllm.sh unchanged.
export IPC_TOOL_VERBOSE=1

MODE="${1:-all}"
case "${MODE}" in
  single|multi|all) ;;
  *)
    echo "Unknown mode '${MODE}'. Usage: $0 [single|multi|all]" >&2
    exit 1
    ;;
esac

bash "${SCRIPT_DIR}/IPC/ds_script/run_ds.sh" "${MODE}"
bash "${SCRIPT_DIR}/IPC/vllm_script/run_vllm.sh" "${MODE}"
