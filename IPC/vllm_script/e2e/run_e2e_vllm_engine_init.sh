#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Single-GPU is just GPU_IDS with one id; multi-GPU uses a comma list
# (e.g. GPU_IDS=0,1). tensor_parallel_size is derived from the count.
GPU_IDS="${GPU_IDS:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"

# Derive tensor-parallel-size from the GPU_IDS count (=1 in single-GPU mode).
IFS=',' read -ra _GPU_ARRAY <<< "${GPU_IDS}"
AUTO_TP_SIZE="${#_GPU_ARRAY[@]}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${AUTO_TP_SIZE}}"

if [[ -z "${MODEL_LABEL:-}" || -z "${MODEL_PRETTY_LABEL:-}" || -z "${MODEL_PATH:-}" ]]; then
  echo "Set MODEL_LABEL, MODEL_PRETTY_LABEL, and MODEL_PATH before running warmup." >&2
  exit 1
fi

echo "[warmup-run] label=${MODEL_LABEL} model=${MODEL_PATH} gpu_ids=${GPU_IDS} tp=${TENSOR_PARALLEL_SIZE}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" "${SCRIPT_DIR}/warmup_vllm_engine_init.py" \
  --model "${MODEL_PATH}" \
  --model-label "${MODEL_PRETTY_LABEL}" \
  --dtype "${DTYPE:-float16}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --trust-remote-code "${TRUST_REMOTE_CODE:-true}" \
  --enable-prefix-caching "${ENABLE_PREFIX_CACHING}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
  --max-num-seqs "${MAX_NUM_SEQS:-512}" \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --enforce-eager
