#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# =====================================================================
# Unified IPC / Anchor live-daemon recovery runner (ZeRO-3). The bench
# launches torch.distributed.run itself (nproc = number of --gpu-ids)
# and spawns one IPC daemon per GPU, so single-GPU is just GPU_IDS with
# one id (world_size 1).
#
# Usage:
#   ./run_ipc_hf_three_models.sh [single|multi]
#
#   single  (default) single-GPU, models llama3_2 1b / qwen3 1.7b / llama3_2 3b
#   multi             multi-GPU (torchrun), model llama3_2 3b
#
# Every value below is still overridable from the environment, e.g.:
#   GPU_IDS=0,1,2,3 ./run_ipc_hf_three_models.sh multi
# =====================================================================

MODE="${1:-single}"
case "${MODE}" in
  single)
    GPU_IDS="${GPU_IDS:-0}"
    BASELINE_NAME="${BASELINE_NAME:-ipc}"
    MASTER_PORT_BASE="${MASTER_PORT_BASE:-29931}"
    ;;
  multi)
    GPU_IDS="${GPU_IDS:-0,1}"
    BASELINE_NAME="${BASELINE_NAME:-ipc_multigpu}"
    MASTER_PORT_BASE="${MASTER_PORT_BASE:-29961}"
    ;;
  *)
    echo "Unknown mode '${MODE}'. Usage: $0 [single|multi]" >&2
    exit 1
    ;;
esac

MODEL_STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-10}"
MODEL_LOG_INTERVAL="${LOG_INTERVAL:-0}"

# --- shared config --------------------------------------------------
REPORT_DIR="${REPORT_DIR:-${SCRIPT_DIR}/report1/${BASELINE_NAME}}"
MODEL_BATCH_SIZE="${BATCH_SIZE:-1}"
MODEL_SEQ_LEN="${SEQ_LEN:-1024}"
ZERO_STAGE="${ZERO_STAGE:-3}"
DTYPE="${DTYPE:-bfloat16}"
IPC_FAULT_SUB_GROUP="${IPC_FAULT_SUB_GROUP:-}"
IPC_FAULT_STAGE="${IPC_FAULT_STAGE:-after_optimizer_step}"

mkdir -p "${REPORT_DIR}"

# Remove run artifacts on exit (IPC daemon, sockets, CUDA-IPC shm); reports kept.
cleanup() {
  rm -f "$HOME"/.cache/torch_extensions/*/*/lock 2>/dev/null || true  # stale DeepSpeed JIT-compile lock
  pkill -f "ds_tool.py --run-server" 2>/dev/null || true
  rm -f /tmp/uipc_socket_v3_dsmp* 2>/dev/null || true
  rm -f /dev/shm/cuda.shm.* /dev/shm/torch_* 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[mode] ${MODE}: gpu_ids=${GPU_IDS} baseline_name=${BASELINE_NAME} master_port_base=${MASTER_PORT_BASE}"

run_model() {
  local label="$1"
  local pretty_label="$2"
  local model_path="$3"
  local port="$4"

  local state_file="${REPORT_DIR}/${label}_state.json"
  local report_file="${REPORT_DIR}/${label}.json"
  local ipc_group="${BASELINE_NAME}_${label}"
  local extra_args=()

  if [[ -n "${IPC_FAULT_SUB_GROUP}" ]]; then
    extra_args+=(--inject-fault-sub-group "${IPC_FAULT_SUB_GROUP}" --inject-fault-stage "${IPC_FAULT_STAGE}")
  fi

  echo "[run] label=${label} model=${model_path} gpu_ids=${GPU_IDS} batch_size=${MODEL_BATCH_SIZE} seq_len=${MODEL_SEQ_LEN} steps_per_epoch=${MODEL_STEPS_PER_EPOCH}"
  # The bench spawns torch.distributed.run and one IPC daemon per --gpu-ids id.
  PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/ipc_hf_attach_bench.py" \
    --model "${model_path}" \
    --model-label "${pretty_label}" \
    --gpu-ids "${GPU_IDS}" \
    --ipc-group "${ipc_group}" \
    --state-file "${state_file}" \
    --report-file "${report_file}" \
    --master-port "${port}" \
    --batch-size "${MODEL_BATCH_SIZE}" \
    --seq-len "${MODEL_SEQ_LEN}" \
    --steps-per-epoch "${MODEL_STEPS_PER_EPOCH}" \
    --log-interval "${MODEL_LOG_INTERVAL}" \
    --zero-stage "${ZERO_STAGE}" \
    --dtype "${DTYPE}" \
    --local-files-only \
    "${extra_args[@]}"
}

# --- models to run --------------------------------------------------
if [[ "${MODE}" == "single" ]]; then
  run_model "llama3_2-1b" "llama3_2 1B" "/public/huggingface-models/meta-llama/Llama-3.2-1B" "$((MASTER_PORT_BASE + 0))"
  run_model "qwen3_1_7b" "qwen3 1_7B" "/public/huggingface-models/Qwen/Qwen3-1.7B" "$((MASTER_PORT_BASE + 1))"
  run_model "llama3_2-3b" "llama3_2 3B" "/public/huggingface-models/meta-llama/Llama-3.2-3B" "$((MASTER_PORT_BASE + 2))"
else
  run_model "llama3_2-3b" "llama3_2 3B" "/public/huggingface-models/meta-llama/Llama-3.2-3B" "$((MASTER_PORT_BASE + 2))"
fi
