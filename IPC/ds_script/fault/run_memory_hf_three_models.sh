#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# =====================================================================
# Unified in-memory (shared-memory) checkpoint baseline runner (ZeRO-3).
# The bench launches torch.distributed.run itself (nproc = number of
# --gpu-ids), so single-GPU is just GPU_IDS with one id (world_size 1).
#
# Usage:
#   ./run_memory_hf_three_models.sh [single|multi]
#
#   single  (default) single-GPU, models llama3_2 1b / qwen3 1.7b / llama3_2 3b
#   multi             multi-GPU (torchrun), model llama3_2 3b
#
# Every value below is still overridable from the environment, e.g.:
#   GPU_IDS=0,1,2,3 ./run_memory_hf_three_models.sh multi
# =====================================================================

MODE="${1:-single}"
case "${MODE}" in
  single)
    GPU_IDS="${GPU_IDS:-0}"
    BASELINE_NAME="${BASELINE_NAME:-memory}"
    MASTER_PORT_BASE="${MASTER_PORT_BASE:-29671}"
    ;;
  multi)
    GPU_IDS="${GPU_IDS:-0,1}"
    BASELINE_NAME="${BASELINE_NAME:-memory_multigpu}"
    MASTER_PORT_BASE="${MASTER_PORT_BASE:-29941}"
    ;;
  *)
    echo "Unknown mode '${MODE}'. Usage: $0 [single|multi]" >&2
    exit 1
    ;;
esac

MODEL_STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-10}"

# --- shared config --------------------------------------------------
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRIPT_DIR}/ckpt/${BASELINE_NAME}}"
REPORT_DIR="${REPORT_DIR:-${SCRIPT_DIR}/report1/${BASELINE_NAME}}"
MODEL_BATCH_SIZE="${BATCH_SIZE:-1}"
MODEL_SEQ_LEN="${SEQ_LEN:-1024}"
ZERO_STAGE="${ZERO_STAGE:-3}"
DTYPE="${DTYPE:-bfloat16}"

mkdir -p "${CHECKPOINT_ROOT}" "${REPORT_DIR}"

# Remove run artifacts on exit (checkpoints + shared-memory segments); reports kept.
cleanup() {
  rm -f "$HOME"/.cache/torch_extensions/*/*/lock 2>/dev/null || true  # stale DeepSpeed JIT-compile lock
  rm -rf "${CHECKPOINT_ROOT}" 2>/dev/null || true
  rm -f /dev/shm/mem_ckpt_* 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[mode] ${MODE}: gpu_ids=${GPU_IDS} baseline_name=${BASELINE_NAME} master_port_base=${MASTER_PORT_BASE}"

run_model() {
  local label="$1"
  local pretty_label="$2"
  local model_path="$3"
  local port="$4"

  local checkpoint_dir="${CHECKPOINT_ROOT}/${label}"
  local state_file="${REPORT_DIR}/${label}_state.json"
  local report_file="${REPORT_DIR}/${label}.json"

  echo "[run] label=${label} model=${model_path} gpu_ids=${GPU_IDS} batch_size=${MODEL_BATCH_SIZE} seq_len=${MODEL_SEQ_LEN} steps_per_epoch=${MODEL_STEPS_PER_EPOCH}"
  # The bench spawns torch.distributed.run and sets CUDA_VISIBLE_DEVICES from --gpu-ids.
  PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/memory_hf_checkpoint_bench.py" \
    --model "${model_path}" \
    --model-label "${pretty_label}" \
    --gpu-ids "${GPU_IDS}" \
    --checkpoint-dir "${checkpoint_dir}" \
    --state-file "${state_file}" \
    --report-file "${report_file}" \
    --master-port "${port}" \
    --batch-size "${MODEL_BATCH_SIZE}" \
    --seq-len "${MODEL_SEQ_LEN}" \
    --steps-per-epoch "${MODEL_STEPS_PER_EPOCH}" \
    --zero-stage "${ZERO_STAGE}" \
    --dtype "${DTYPE}" \
    --local-files-only
  rm -rf "${checkpoint_dir}"
}

# --- models to run --------------------------------------------------
if [[ "${MODE}" == "single" ]]; then
  run_model "llama3_2-1b" "llama3_2 1B" "/public/huggingface-models/meta-llama/Llama-3.2-1B" "$((MASTER_PORT_BASE + 0))"
  run_model "qwen3_1_7b" "qwen3 1_7B" "/public/huggingface-models/Qwen/Qwen3-1.7B" "$((MASTER_PORT_BASE + 1))"
  run_model "llama3_2-3b" "llama3_2 3B" "/public/huggingface-models/meta-llama/Llama-3.2-3B" "$((MASTER_PORT_BASE + 2))"
else
  run_model "llama3_2-3b" "llama3_2 3B" "/public/huggingface-models/meta-llama/Llama-3.2-3B" "$((MASTER_PORT_BASE + 2))"
fi
