#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# =====================================================================
# Unified ShareGPT overhead runner for sharegpt_overhead_vllm_bench.py.
# Single-GPU is just tensor_parallel_size=1, so both modes share one bench.
#
# Usage:
#   ./run_vllm_sharegpt_overhead_models.sh [single|multi]
#
#   single  (default) single-GPU (TP=1), models qwen3 8b / 14b / 32b
#   multi             multi-GPU (TP=visible GPUs), model qwen3 32b
#
# Every value below is still overridable from the environment, e.g.:
#   GPU_IDS=0,1,2,3 ./run_vllm_sharegpt_overhead_models.sh multi
# =====================================================================

MODE="${1:-single}"
case "${MODE}" in
  single)
    GPU_IDS="${GPU_IDS:-0}"
    REPORT_ROOT="${REPORT_ROOT:-${SCRIPT_DIR}/report}"
    ;;
  multi)
    GPU_IDS="${GPU_IDS:-0,1}"
    REPORT_ROOT="${REPORT_ROOT:-${SCRIPT_DIR}/report_multigpu}"
    ;;
  *)
    echo "Unknown mode '${MODE}'. Usage: $0 [single|multi]" >&2
    exit 1
    ;;
esac

# --- config (shared; per-mode defaults set above) -------------------
METHODS="${METHODS:-baseline ipc}"
NUM_REQUESTS="${NUM_REQUESTS:-512}"
WARMUP_PROCESSED_TOKENS="${WARMUP_PROCESSED_TOKENS:-40000}"
MEASURE_PROCESSED_TOKENS="${MEASURE_PROCESSED_TOKENS:-200000}"
LOG_INTERVAL_TOKENS="${LOG_INTERVAL_TOKENS:-10000}"

SHAREGPT_DATASET_PATH="${SHAREGPT_DATASET_PATH:-/root/data/mhy/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json}"
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
SEND_EXIT_ON_FINISH="${SEND_EXIT_ON_FINISH:-true}"

mkdir -p "${REPORT_ROOT}/baseline" "${REPORT_ROOT}/ipc"

# Remove run artifacts on exit (vLLM IPC daemon, sockets, CUDA-IPC shm); reports kept.
cleanup() {
  pkill -f "vllm_tool.py --run-server" 2>/dev/null || true
  rm -f /tmp/uipc_socket_v3_vllmmp* 2>/dev/null || true
  rm -f /dev/shm/cuda.shm.* /dev/shm/torch_* 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ -z "${SHAREGPT_DATASET_PATH}" ]]; then
  echo "Set SHAREGPT_DATASET_PATH to a ShareGPT JSON file before running." >&2
  exit 1
fi

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
GPU_COUNT="${#GPU_ID_ARRAY[@]}"
if [[ "${GPU_COUNT}" -le 0 ]]; then
  echo "GPU_IDS must contain at least one GPU id." >&2
  exit 1
fi

# TP defaults to the number of visible GPUs (=1 in single-GPU mode).
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${GPU_COUNT}}"
if [[ "${TENSOR_PARALLEL_SIZE}" -le 0 ]]; then
  echo "TENSOR_PARALLEL_SIZE must be positive." >&2
  exit 1
fi
if [[ "${TENSOR_PARALLEL_SIZE}" -gt "${GPU_COUNT}" ]]; then
  echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} exceeds visible GPU count ${GPU_COUNT}." >&2
  exit 1
fi

echo "[mode] ${MODE}: gpu_ids=${GPU_IDS} tensor_parallel_size=${TENSOR_PARALLEL_SIZE} report_root=${REPORT_ROOT}"

run_model() {
  local method="$1"
  local label="$2"
  local pretty_label="$3"
  local model_path="$4"

  local state_file="${REPORT_ROOT}/${method}/${label}_state.json"
  local report_file="${REPORT_ROOT}/${method}/${label}.json"
  local -a extra_args=()

  if [[ -n "${DISABLE_SHUFFLE:-}" ]]; then
    extra_args+=(--disable-shuffle "${DISABLE_SHUFFLE}")
  fi

  if [[ "${ENFORCE_EAGER:-1}" == "1" ]]; then
    extra_args+=(--enforce-eager)
  fi

  echo "[run] method=${method} label=${label} model=${model_path} gpu_ids=${GPU_IDS} tensor_parallel_size=${TENSOR_PARALLEL_SIZE} num_requests=${NUM_REQUESTS} warmup_processed_tokens=${WARMUP_PROCESSED_TOKENS} measure_processed_tokens=${MEASURE_PROCESSED_TOKENS}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" "${SCRIPT_DIR}/sharegpt_overhead_vllm_bench.py" \
    --method "${method}" \
    --model "${model_path}" \
    --model-label "${pretty_label}" \
    --state-file "${state_file}" \
    --report-file "${report_file}" \
    --dtype "${DTYPE}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --trust-remote-code "${TRUST_REMOTE_CODE}" \
    --enable-prefix-caching "${ENABLE_PREFIX_CACHING}" \
    --send-exit-on-finish "${SEND_EXIT_ON_FINISH}" \
    --dataset-path "${SHAREGPT_DATASET_PATH}" \
    --num-requests "${NUM_REQUESTS}" \
    --warmup-processed-tokens "${WARMUP_PROCESSED_TOKENS}" \
    --measure-processed-tokens "${MEASURE_PROCESSED_TOKENS}" \
    --log-interval-tokens "${LOG_INTERVAL_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    "${extra_args[@]}"
}

run_all_methods_for_model() {
  local label="$1"
  local pretty_label="$2"
  local model_path="$3"

  for method in ${METHODS}; do
    run_model "${method}" "${label}" "${pretty_label}" "${model_path}"
  done
}

if [[ "${MODE}" == "single" ]]; then
  # --- single-GPU models ----------------------------------------------
  run_all_methods_for_model "qwen3 8b" "qwen3 8B" "/public/huggingface-models/Qwen/Qwen3-8B"
  run_all_methods_for_model "qwen3 14b" "qwen3 14B" "/public/huggingface-models/Qwen/Qwen3-14B"
  run_all_methods_for_model "qwen3 32b" "qwen3 32B" "/public/huggingface-models/Qwen/Qwen3-32B"
else
  # --- multi-GPU models -----------------------------------------------
  run_all_methods_for_model "qwen3 32b" "qwen3 32B" "/public/huggingface-models/Qwen/Qwen3-32B"
  # run_all_methods_for_model "qwen3 235b" "qwen3 235B" "/public/huggingface-models/Qwen/Qwen3-235B-A22B"
fi
