#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# =====================================================================
# Unified e2e suite orchestrator (warmup + replay/kv_memory/ipc).
# Single-GPU is just GPU_IDS with one id; the per-method scripts derive
# tensor_parallel_size from the GPU_IDS count.
#
# Usage:
#   ./run_e2e_vllm_model_suite.sh [single|multi]
#
#   single  (default) single-GPU (TP=1), models qwen3 8b / 14b / 32b
#   multi             multi-GPU (TP=visible GPUs), model qwen3 14b
#
# Every value below is still overridable from the environment, e.g.:
#   GPU_IDS=0,1,2,3 ./run_e2e_vllm_model_suite.sh multi
# =====================================================================

MODE="${1:-single}"
case "${MODE}" in
  single)
    GPU_IDS="${GPU_IDS:-0}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
    REPORT_SUBDIR_IPC="${REPORT_SUBDIR_IPC:-ipc}"
    REPORT_SUBDIR_REPLAY="${REPORT_SUBDIR_REPLAY:-replay}"
    REPORT_SUBDIR_KV_MEMORY="${REPORT_SUBDIR_KV_MEMORY:-kv_memory}"
    ;;
  multi)
    GPU_IDS="${GPU_IDS:-0,1}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
    REPORT_SUBDIR_IPC="${REPORT_SUBDIR_IPC:-e2e_vllm_ipc_multigpu}"
    REPORT_SUBDIR_REPLAY="${REPORT_SUBDIR_REPLAY:-e2e_vllm_replay_multigpu}"
    REPORT_SUBDIR_KV_MEMORY="${REPORT_SUBDIR_KV_MEMORY:-e2e_vllm_kv_memory_multigpu}"
    ;;
  *)
    echo "Unknown mode '${MODE}'. Usage: $0 [single|multi]" >&2
    exit 1
    ;;
esac

METHODS="${METHODS:-ipc}"

# Per-method GPU sets default to the shared GPU_IDS.
GPU_IDS_IPC="${GPU_IDS_IPC:-${GPU_IDS}}"
GPU_IDS_REPLAY="${GPU_IDS_REPLAY:-${GPU_IDS}}"
GPU_IDS_KV_MEMORY="${GPU_IDS_KV_MEMORY:-${GPU_IDS}}"
WARMUP_GPU_IDS="${WARMUP_GPU_IDS:-${GPU_IDS}}"

# --- shared config --------------------------------------------------
REPORT_ROOT="${REPORT_ROOT:-${SCRIPT_DIR}/report1}"
SHAREGPT_DATASET_PATH="${SHAREGPT_DATASET_PATH:-/root/data/mhy/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json}"
TOTAL_REQUESTS="${TOTAL_REQUESTS:-10000}"
CONCURRENCY="${CONCURRENCY:-1024}"
FAULT_FREQUENCY="${FAULT_FREQUENCY:-10}"
FAULT_CHECK_INTERVAL_SECONDS="${FAULT_CHECK_INTERVAL_SECONDS:-10}"
LOG_INTERVAL_REQUESTS="${LOG_INTERVAL_REQUESTS:-10}"
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
# KV flush/commit cadence K — affects ONLY the kv_memory method. K=1: cuda-sync + commit every
# step (original, max recovery completeness). K>1: block copies overlap decode (async on
# copy_stream), sync+commit every K steps → lower steady-state overhead; recovery re-decodes up
# to K extra tail steps. e.g. KV_FLUSH_INTERVAL=8 bash run_e2e_vllm_model_suite.sh. Default 1 reverts.
KV_FLUSH_INTERVAL="${KV_FLUSH_INTERVAL:-20}"

METHODS="${METHODS//,/ }"

# Note: tensor_parallel_size is intentionally NOT exported here; each
# per-method script derives it from the GPU_IDS it receives, so single-GPU
# (one id) becomes TP=1 automatically.
export PYTHON_BIN
export SHAREGPT_DATASET_PATH
export TOTAL_REQUESTS
export CONCURRENCY
export FAULT_FREQUENCY
export FAULT_CHECK_INTERVAL_SECONDS
export LOG_INTERVAL_REQUESTS
export DTYPE
export GPU_MEMORY_UTILIZATION
export MAX_NUM_BATCHED_TOKENS
export MAX_NUM_SEQS
export MAX_MODEL_LEN
export ENABLE_PREFIX_CACHING
export TRUST_REMOTE_CODE
export CUDA_DEVICE_ORDER
export KV_FLUSH_INTERVAL  # picked up by run_e2e_vllm_kv_memory_models.sh → VLLM_KV_FLUSH_INTERVAL

mkdir -p \
  "${REPORT_ROOT}/${REPORT_SUBDIR_IPC}" \
  "${REPORT_ROOT}/${REPORT_SUBDIR_REPLAY}" \
  "${REPORT_ROOT}/${REPORT_SUBDIR_KV_MEMORY}"

if [[ -z "${SHAREGPT_DATASET_PATH}" ]]; then
  echo "Set SHAREGPT_DATASET_PATH to a ShareGPT JSON file before running." >&2
  exit 1
fi

echo "[mode] ${MODE}: gpu_ids=${GPU_IDS} max_model_len=${MAX_MODEL_LEN} report_root=${REPORT_ROOT}"

run_warmup_for_model() {
  local label="$1"
  local pretty_label="$2"
  local model_path="$3"

  echo "================================================================"
  echo "[suite] warmup label=${label} model=${model_path} gpu_ids=${WARMUP_GPU_IDS}"
  echo "================================================================"
  MODEL_LABEL="${label}" \
  MODEL_PRETTY_LABEL="${pretty_label}" \
  MODEL_PATH="${model_path}" \
  GPU_IDS="${WARMUP_GPU_IDS}" \
  bash "${SCRIPT_DIR}/run_e2e_vllm_engine_init.sh"
}

run_method() {
  local method="$1"
  local label="$2"
  local pretty_label="$3"
  local model_path="$4"
  local method_gpu_ids
  local method_report_root
  local method_script

  case "${method}" in
    ipc)
      method_gpu_ids="${GPU_IDS_IPC}"
      method_report_root="${REPORT_ROOT}/${REPORT_SUBDIR_IPC}"
      method_script="${SCRIPT_DIR}/run_e2e_vllm_ipc_models.sh"
      ;;
    replay)
      method_gpu_ids="${GPU_IDS_REPLAY}"
      method_report_root="${REPORT_ROOT}/${REPORT_SUBDIR_REPLAY}"
      method_script="${SCRIPT_DIR}/run_e2e_vllm_replay_models.sh"
      ;;
    kv_memory)
      method_gpu_ids="${GPU_IDS_KV_MEMORY}"
      method_report_root="${REPORT_ROOT}/${REPORT_SUBDIR_KV_MEMORY}"
      method_script="${SCRIPT_DIR}/run_e2e_vllm_kv_memory_models.sh"
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 1
      ;;
  esac

  echo "----------------------------------------------------------------"
  echo "[suite] method=${method} label=${label} model=${model_path} gpu_ids=${method_gpu_ids}"
  echo "----------------------------------------------------------------"
  MODEL_LABEL="${label}" \
  MODEL_PRETTY_LABEL="${pretty_label}" \
  MODEL_PATH="${model_path}" \
  REPORT_ROOT="${method_report_root}" \
  GPU_IDS="${method_gpu_ids}" \
  bash "${method_script}"
}

run_all_methods_for_model() {
  local label="$1"
  local pretty_label="$2"
  local model_path="$3"
  # Optional 4th arg: per-model gpu_memory_utilization override (empty = shared default).
  # Exported once so warmup + all three methods use the same value -> fair for a given model.
  local model_gpu_util="${4:-${GPU_MEMORY_UTILIZATION}}"
  export GPU_MEMORY_UTILIZATION="${model_gpu_util}"
  echo "[suite] model=${label} gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} (applied to warmup + all methods)"

  run_warmup_for_model "${label}" "${pretty_label}" "${model_path}"
  for method in ${METHODS}; do
    run_method "${method}" "${label}" "${pretty_label}" "${model_path}"
  done
}

# --- models to run --------------------------------------------------
# 4th arg = per-model gpu_memory_utilization (empty = shared default 0.8). 32B single-GPU
# weights ~64GB; 0.8 barely covers weights with no KV headroom -> raise to 0.9. All methods
# share the same value for fairness.
if [[ "${MODE}" == "single" ]]; then
  run_all_methods_for_model "qwen3 8b" "qwen3 8B" "/public/huggingface-models/Qwen/Qwen3-8B"
  #run_all_methods_for_model "qwen3 14b" "qwen3 14B" "/public/huggingface-models/Qwen/Qwen3-14B"
  #run_all_methods_for_model "qwen3 32b" "qwen3 32B" "/public/huggingface-models/Qwen/Qwen3-32B" "0.9"
else
  run_all_methods_for_model "qwen3 14b" "qwen3 14B" "/public/huggingface-models/Qwen/Qwen3-14B"
  # run_all_methods_for_model "qwen3 235b" "qwen3 235B" "/public/huggingface-models/Qwen/Qwen3-235B-A22B"
fi
