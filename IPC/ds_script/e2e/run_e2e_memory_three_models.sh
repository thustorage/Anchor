#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# =====================================================================
# Unified e2e in-memory (shared-memory) checkpoint fault runner.
# The bench derives its launch mode from the --gpu-ids count (one id ->
# single-GPU direct launch, several -> torchrun multi-GPU) and sets
# CUDA_VISIBLE_DEVICES for the worker itself.
#
# Mode is chosen by the first positional arg (default single); the suite
# (run_e2e_ds_model_suite.sh) drives it by exporting MODE:
#   ./run_e2e_memory_three_models.sh [single|multi]
# =====================================================================

MODE="${1:-${MODE:-single}}"    # positional arg wins; suite drives via exported MODE; default single
case "${MODE}" in
  single)
    GPU_IDS="${GPU_IDS:-0}"
    REPORT_ROOT="${REPORT_ROOT:-${SCRIPT_DIR}/report1/e2e_memory}"
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRIPT_DIR}/ckpt/e2e_memory}"
    MASTER_PORT_BASE="${MASTER_PORT_BASE:-29851}"
    ;;
  multi)
    GPU_IDS="${GPU_IDS:-0,1}"
    REPORT_ROOT="${REPORT_ROOT:-${SCRIPT_DIR}/report1/e2e_memory_multigpu}"
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRIPT_DIR}/ckpt/e2e_memory_multigpu}"
    MASTER_PORT_BASE="${MASTER_PORT_BASE:-30161}"
    ;;
  *)
    echo "Unknown mode '${MODE}'. Usage: $0 [single|multi]" >&2
    exit 1
    ;;
esac

# --- mode-independent (identical for single and multi) --------------
MODEL_TOTAL_STEPS="${TOTAL_STEPS:-2000}"
MODEL_CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
FAULT_FREQUENCY="${FAULT_FREQUENCY:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"

# --- shared config --------------------------------------------------
MODEL_BATCH_SIZE="${BATCH_SIZE:-4}"
MODEL_SEQ_LEN="${SEQ_LEN:-1024}"
FAULT_CHECK_INTERVAL_SECONDS="${FAULT_CHECK_INTERVAL_SECONDS:-10}"
ZERO_STAGE="${ZERO_STAGE:-3}"
DTYPE="${DTYPE:-bfloat16}"
RECOMPUTATION="${RECOMPUTATION:-1}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"

mkdir -p "${REPORT_ROOT}" "${CHECKPOINT_ROOT}"

# Reap detached worker processes (launched with start_new_session, so they
# survive the manager) and remove run artifacts on exit; reports are kept.
cleanup() {
  rm -f "$HOME"/.cache/torch_extensions/*/*/lock 2>/dev/null || true  # stale DeepSpeed JIT-compile lock
  pkill -f "e2e_memory_checkpoint_fault_bench.py --phase worker" 2>/dev/null || true
  rm -rf "${CHECKPOINT_ROOT}" 2>/dev/null || true
  rm -f /dev/shm/e2e_mem_* 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[mode] ${MODE}: gpu_ids=${GPU_IDS} report_root=${REPORT_ROOT} master_port_base=${MASTER_PORT_BASE}"

run_model() {
  local label="$1"
  local pretty_label="$2"
  local model_path="$3"
  local port="$4"

  local checkpoint_dir="${CHECKPOINT_ROOT}/${label}"
  local state_file="${REPORT_ROOT}/${label}_state.json"
  local report_file="${REPORT_ROOT}/${label}.json"
  local fault_file="${REPORT_ROOT}/${label}_faults.jsonl"
  local extra_args=()

  if [[ "${RECOMPUTATION}" == "1" ]]; then
    extra_args+=(--recomputation)
  fi
  if [[ "${KEEP_CHECKPOINTS}" == "1" ]]; then
    extra_args+=(--keep-checkpoints)
  fi

  echo "[run] label=${label} model=${model_path} gpu_ids=${GPU_IDS} total_steps=${MODEL_TOTAL_STEPS} checkpoint_interval=${MODEL_CHECKPOINT_INTERVAL} fault_frequency=${FAULT_FREQUENCY} fault_check_interval_seconds=${FAULT_CHECK_INTERVAL_SECONDS} recomputation=${RECOMPUTATION}"
  # The bench derives single/multi launch from --gpu-ids and sets CUDA_VISIBLE_DEVICES itself.
  PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/e2e_memory_checkpoint_fault_bench.py" \
    --model "${model_path}" \
    --model-label "${pretty_label}" \
    --gpu-ids "${GPU_IDS}" \
    --checkpoint-dir "${checkpoint_dir}" \
    --state-file "${state_file}" \
    --report-file "${report_file}" \
    --fault-file "${fault_file}" \
    --master-port "${port}" \
    --log-interval "${LOG_INTERVAL}" \
    --batch-size "${MODEL_BATCH_SIZE}" \
    --seq-len "${MODEL_SEQ_LEN}" \
    --total-steps "${MODEL_TOTAL_STEPS}" \
    --checkpoint-interval "${MODEL_CHECKPOINT_INTERVAL}" \
    --fault-frequency "${FAULT_FREQUENCY}" \
    --fault-check-interval-seconds "${FAULT_CHECK_INTERVAL_SECONDS}" \
    --zero-stage "${ZERO_STAGE}" \
    --dtype "${DTYPE}" \
    --local-files-only \
    "${extra_args[@]}"
}

# --- suite single-model override (used by run_e2e_ds_model_suite.sh) ---
# When MODEL_LABEL/MODEL_PRETTY_LABEL/MODEL_PATH are exported (by the suite),
# run just that one model and exit; otherwise fall through to the default below.
if [[ -n "${MODEL_LABEL:-}" || -n "${MODEL_PRETTY_LABEL:-}" || -n "${MODEL_PATH:-}" ]]; then
  if [[ -z "${MODEL_LABEL:-}" || -z "${MODEL_PRETTY_LABEL:-}" || -z "${MODEL_PATH:-}" ]]; then
    echo "MODEL_LABEL, MODEL_PRETTY_LABEL, and MODEL_PATH must be set together." >&2
    exit 1
  fi
  run_model "${MODEL_LABEL}" "${MODEL_PRETTY_LABEL}" "${MODEL_PATH}" "${MASTER_PORT_BASE}"
  exit 0
fi

#  --- Mode A models (active) -----------------------------------------
# run_model "llama3_2-1b" "llama3_2 1B" "/public/huggingface-models/meta-llama/Llama-3.2-1B" "$((MASTER_PORT_BASE + 0))"
# run_model "qwen3_1_7b" "qwen3 1_7B" "/public/huggingface-models/Qwen/Qwen3-1.7B" "$((MASTER_PORT_BASE + 1))"
run_model "llama3_2-3b" "llama3_2 3B" "/public/huggingface-models/meta-llama/Llama-3.2-3B" "$((MASTER_PORT_BASE + 2))"

# --- Mode B model (uncomment together with the Mode B config) -------
# run_model "qwen2.5-14b" "qwen2.5 14B" "/public/huggingface-models/Qwen/Qwen2.5-14B" "$((MASTER_PORT_BASE))"
