#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# =====================================================================
# Unified DeepSpeed train-overhead runner. Single-GPU is just GPU_IDS
# with one id: it launches torch.distributed.run --nproc_per_node=1, so
# single-GPU is the world_size=1 special case of the multi-GPU path.
#
# Usage:
#   ./run_train_overhead_three_models.sh [single|multi]
#
#   single  (default) single-GPU (nproc=1), models llama3_2 1b / qwen3 1.7b / llama3_2 3b
#   multi             multi-GPU (nproc=visible GPUs), model llama3_2 3b
#
# Every value below is still overridable from the environment, e.g.:
#   GPU_IDS=0,1,2,3 ./run_train_overhead_three_models.sh multi
#
# Note: overhead is measured post-warmup (steady-state step time), so the
# torchrun/ds_mp_tool launch stack does not affect the reported numbers.
# =====================================================================

MODE="${1:-single}"
case "${MODE}" in
  single)
    GPU_IDS="${GPU_IDS:-0}"
    REPORT_ROOT="${REPORT_ROOT:-${SCRIPT_DIR}/report1}"
    MASTER_PORT_BASE="${MASTER_PORT_BASE:-29731}"
    ;;
  multi)
    GPU_IDS="${GPU_IDS:-0,1}"
    REPORT_ROOT="${REPORT_ROOT:-${SCRIPT_DIR}/report1_multigpu}"
    MASTER_PORT_BASE="${MASTER_PORT_BASE:-29871}"
    ;;
  *)
    echo "Unknown mode '${MODE}'. Usage: $0 [single|multi]" >&2
    exit 1
    ;;
esac

MODEL_BATCH_SIZE="${BATCH_SIZE:-10}"
MODEL_STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-200}"

# --- shared config --------------------------------------------------
METHODS="${METHODS:-baseline ipc}"
MODEL_SEQ_LEN="${SEQ_LEN:-1024}"
MODEL_WARMUP_STEPS="${WARMUP_STEPS:-40}"
ZERO_STAGE="${ZERO_STAGE:-3}"
DTYPE="${DTYPE:-bfloat16}"
RECOMPUTATION="${RECOMPUTATION:-1}"

mkdir -p "${REPORT_ROOT}/baseline" "${REPORT_ROOT}/ipc"

# Remove run artifacts on exit (IPC daemon, sockets, CUDA-IPC shm); reports kept.
cleanup() {
  rm -f "$HOME"/.cache/torch_extensions/*/*/lock 2>/dev/null || true  # stale DeepSpeed JIT-compile lock
  pkill -f "ds_tool.py --run-server" 2>/dev/null || true
  rm -f /tmp/uipc_socket_v3_dsmp* 2>/dev/null || true
  rm -f /dev/shm/cuda.shm.* /dev/shm/torch_* 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# nproc_per_node is derived from the GPU_IDS count (=1 in single-GPU mode).
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
GPU_COUNT="${#GPU_ID_ARRAY[@]}"
if [[ "${GPU_COUNT}" -le 0 ]]; then
  echo "GPU_IDS must contain at least one GPU id." >&2
  exit 1
fi

echo "[mode] ${MODE}: gpu_ids=${GPU_IDS} nproc=${GPU_COUNT} report_root=${REPORT_ROOT}"

run_model() {
  local method="$1"
  local label="$2"
  local pretty_label="$3"
  local model_path="$4"
  local port="$5"

  local state_file="${REPORT_ROOT}/${method}/${label}_state.json"
  local report_file="${REPORT_ROOT}/${method}/${label}.json"
  local extra_args=()

  if [[ "${RECOMPUTATION}" == "1" ]]; then
    extra_args+=(--recomputation)
  fi
  if [[ "${method}" == "ipc" ]]; then
    extra_args+=(--ipc-group "lab3_ipc_multigpu_${label}")
  fi

  echo "[run] method=${method} label=${label} model=${model_path} gpu_ids=${GPU_IDS} nproc=${GPU_COUNT} batch_size=${MODEL_BATCH_SIZE} seq_len=${MODEL_SEQ_LEN} warmup_steps=${MODEL_WARMUP_STEPS} steps_per_epoch=${MODEL_STEPS_PER_EPOCH} recomputation=${RECOMPUTATION}"
  PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}" \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "${GPU_COUNT}" \
    --master_port "${port}" \
    "${SCRIPT_DIR}/train_overhead_hf_bench.py" \
    --method "${method}" \
    --model "${model_path}" \
    --model-label "${pretty_label}" \
    --gpu-ids "${GPU_IDS}" \
    --state-file "${state_file}" \
    --report-file "${report_file}" \
    --master-port "${port}" \
    --batch-size "${MODEL_BATCH_SIZE}" \
    --seq-len "${MODEL_SEQ_LEN}" \
    --warmup-steps "${MODEL_WARMUP_STEPS}" \
    --steps-per-epoch "${MODEL_STEPS_PER_EPOCH}" \
    --zero-stage "${ZERO_STAGE}" \
    --dtype "${DTYPE}" \
    --local-files-only \
    "${extra_args[@]}"
}

run_all_methods_for_model() {
  local label="$1"
  local pretty_label="$2"
  local model_path="$3"
  local model_offset="$4"
  local method_index=0

  for method in ${METHODS}; do
    local port="$((MASTER_PORT_BASE + method_index * 20 + model_offset))"
    run_model "${method}" "${label}" "${pretty_label}" "${model_path}" "${port}"
    method_index="$((method_index + 1))"
  done
}

# --- models to run --------------------------------------------------
if [[ "${MODE}" == "single" ]]; then
  run_all_methods_for_model "llama3_2-1b" "llama3_2 1B" "/public/huggingface-models/meta-llama/Llama-3.2-1B" 1
  run_all_methods_for_model "qwen3_1_7b" "qwen3 1_7B" "/public/huggingface-models/Qwen/Qwen3-1.7B" 2
  run_all_methods_for_model "llama3_2-3b" "llama3_2 3B" "/public/huggingface-models/meta-llama/Llama-3.2-3B" 3
else
  run_all_methods_for_model "llama3_2-3b" "llama3_2 3B" "/public/huggingface-models/meta-llama/Llama-3.2-3B" 3
fi
