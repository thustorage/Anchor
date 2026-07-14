#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# =====================================================================
# Unified IPC e2e fault runner. Single-GPU is just GPU_IDS with one id
# (tensor_parallel_size derived from the count). Pick ONE mode below.
# Mode A (single-GPU) is active; comment it and uncomment Mode B for TP>1.
# =====================================================================

# --- Mode A: single-GPU (TP=1) --------------------------------------
GPU_IDS="${GPU_IDS:-0}"
REPORT_ROOT="${REPORT_ROOT:-${SCRIPT_DIR}/report1/e2e_vllm_ipc}"

# --- Mode B: multi-GPU (TP=visible GPUs) ----------------------------
# GPU_IDS="${GPU_IDS:-0,1}"
# REPORT_ROOT="${REPORT_ROOT:-${SCRIPT_DIR}/report1/e2e_vllm_ipc_multigpu}"

# --- shared config --------------------------------------------------
SHAREGPT_DATASET_PATH="${SHAREGPT_DATASET_PATH:-/root/data/mhy/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json}"
TOTAL_REQUESTS="${TOTAL_REQUESTS:-5000}"
CONCURRENCY="${CONCURRENCY:-256}"
FAULT_FREQUENCY="${FAULT_FREQUENCY:-40}"
FAULT_CHECK_INTERVAL_SECONDS="${FAULT_CHECK_INTERVAL_SECONDS:-10}"
LOG_INTERVAL_REQUESTS="${LOG_INTERVAL_REQUESTS:-10}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"

# Derive tensor-parallel-size from the GPU_IDS count (=1 in single-GPU mode).
IFS=',' read -ra _GPU_ARRAY <<< "${GPU_IDS}"
AUTO_TP_SIZE="${#_GPU_ARRAY[@]}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${AUTO_TP_SIZE}}"

mkdir -p "${REPORT_ROOT}"

# Remove run artifacts (vLLM IPC daemon, sockets, CUDA-IPC shm) + sweep any stale
# /dev/shm segments left by a prior crashed run (incl. a kv_memory buffer) that would
# otherwise fill /dev/shm and ENOSPC engine init. Named-segment only. Reports kept.
sweep_shm() {
  rm -f /tmp/uipc_socket_v3_vllmmp* 2>/dev/null || true
  rm -f /dev/shm/cuda.shm.* /dev/shm/torch_* 2>/dev/null || true
  rm -f /dev/shm/vllm_kv_cpu_buffer_* /dev/shm/vllm_kv_memory_* /dev/shm/kv_cpu_buffer* \
        /dev/shm/vllm_kv_cpu_checkpoint* 2>/dev/null || true
}
cleanup() {
  pkill -f "vllm_tool.py --run-server" 2>/dev/null || true
  sweep_shm
}
trap cleanup EXIT INT TERM
# Pre-clean stale segments from a prior crashed/OOM-killed run before we start.
sweep_shm

if [[ -z "${SHAREGPT_DATASET_PATH}" ]]; then
  echo "Set SHAREGPT_DATASET_PATH to a ShareGPT JSON file before running." >&2
  exit 1
fi

run_model() {
  local label="$1"
  local pretty_label="$2"
  local model_path="$3"

  local state_file="${REPORT_ROOT}/${label}_state.json"
  local report_file="${REPORT_ROOT}/${label}.json"
  local fault_file="${REPORT_ROOT}/${label}_faults.jsonl"
  local resume_state_file="${REPORT_ROOT}/${label}_resume_state.json"
  local -a extra_args=()

  if [[ -n "${DISABLE_SHUFFLE:-}" ]]; then
    extra_args+=(--disable-shuffle "${DISABLE_SHUFFLE}")
  fi

  echo "[run] label=${label} model=${model_path} gpu_ids=${GPU_IDS} tp=${TENSOR_PARALLEL_SIZE} total_requests=${TOTAL_REQUESTS} concurrency=${CONCURRENCY} fault_frequency=${FAULT_FREQUENCY} fault_check_interval_seconds=${FAULT_CHECK_INTERVAL_SECONDS} enable_prefix_caching=${ENABLE_PREFIX_CACHING}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" "${SCRIPT_DIR}/e2e_vllm_ipc_fault_bench.py" \
    --model "${model_path}" \
    --model-label "${pretty_label}" \
    --dtype "${DTYPE:-float16}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
    --trust-remote-code "${TRUST_REMOTE_CODE:-true}" \
    --enable-prefix-caching "${ENABLE_PREFIX_CACHING}" \
    --dataset-path "${SHAREGPT_DATASET_PATH}" \
    --total-requests "${TOTAL_REQUESTS}" \
    --concurrency "${CONCURRENCY}" \
    --fault-frequency "${FAULT_FREQUENCY}" \
    --fault-check-interval-seconds "${FAULT_CHECK_INTERVAL_SECONDS}" \
    --log-interval-requests "${LOG_INTERVAL_REQUESTS}" \
    --state-file "${state_file}" \
    --report-file "${report_file}" \
    --fault-file "${fault_file}" \
    --resume-state-file "${resume_state_file}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
    --max-num-seqs "${MAX_NUM_SEQS:-512}" \
    --max-model-len "${MAX_MODEL_LEN:-16384}" \
    --enforce-eager \
    "${extra_args[@]}"
}

if [[ -n "${MODEL_LABEL:-}" || -n "${MODEL_PRETTY_LABEL:-}" || -n "${MODEL_PATH:-}" ]]; then
  if [[ -z "${MODEL_LABEL:-}" || -z "${MODEL_PRETTY_LABEL:-}" || -z "${MODEL_PATH:-}" ]]; then
    echo "MODEL_LABEL, MODEL_PRETTY_LABEL, and MODEL_PATH must be set together." >&2
    exit 1
  fi
  run_model "${MODEL_LABEL}" "${MODEL_PRETTY_LABEL}" "${MODEL_PATH}"
  exit 0
fi

# --- model to run (uncomment one) -----------------------------------
# run_model "qwen2_5_7b" "qwen2.5 7B" "/mnt/mhy/model/qwen2.5-7b"
# run_model "qwen3_14b" "qwen3 14B" "/mnt/mhy/model/qwen3-14b"
# run_model "llama3_1_8b" "llama3.1 8B" "/mnt/mhy/model/llama3.1-8b"          # Mode B typical
run_model "deepseekv2_16b" "deepseekv2 16B" "/mnt/mhy/model/deepseekv2-16b"   # Mode A typical
