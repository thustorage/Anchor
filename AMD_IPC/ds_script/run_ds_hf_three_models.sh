#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
BASELINE_NAME="${BASELINE_NAME:-ds_native}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRIPT_DIR}/ckpt/${BASELINE_NAME}}"
REPORT_DIR="${REPORT_DIR:-${SCRIPT_DIR}/report1/${BASELINE_NAME}}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29631}"

mkdir -p "${CHECKPOINT_ROOT}" "${REPORT_DIR}"

run_model() {
  local label="$1"
  local pretty_label="$2"
  local model_path="$3"
  local port="$4"

  local checkpoint_dir="${CHECKPOINT_ROOT}/${label}"
  local state_file="${REPORT_DIR}/${label}_state.json"
  local report_file="${REPORT_DIR}/${label}.json"
  local model_batch_size="${BATCH_SIZE:-1}"
  local model_seq_len="${SEQ_LEN:-1024}"
  local model_steps_per_epoch="${STEPS_PER_EPOCH:-1}"

  echo "[run] label=${label} model=${model_path} batch_size=${model_batch_size} seq_len=${model_seq_len} steps_per_epoch=${model_steps_per_epoch}"
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  HIP_VISIBLE_DEVICES="${GPU_ID}" ROCR_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/ds_hf_checkpoint_bench.py" \
    --model "${model_path}" \
    --model-label "${pretty_label}" \
    --checkpoint-dir "${checkpoint_dir}" \
    --state-file "${state_file}" \
    --report-file "${report_file}" \
    --master-port "${port}" \
    --batch-size "${model_batch_size}" \
    --seq-len "${model_seq_len}" \
    --steps-per-epoch "${model_steps_per_epoch}" \
    --zero-stage "${ZERO_STAGE:-3}" \
    --dtype "${DTYPE:-bfloat16}" \
    --gradient-checkpointing \
    --local-files-only
  rm -rf "${checkpoint_dir}"
}

run_model "qwen3_0_6b" "qwen3 0.6B" "/root/mhy/model/qwen3-0.6b" "$((MASTER_PORT_BASE + 0))"
run_model "llama3_2_1b" "llama3.2 1B" "/root/mhy/model/llama3.2-1b" "$((MASTER_PORT_BASE + 1))"
run_model "qwen2_5_1_5b" "qwen2.5 1.5B" "/root/mhy/model/qwen2.5-1.5b-it" "$((MASTER_PORT_BASE + 2))"
