#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# =====================================================================
# Unified DeepSpeed e2e suite orchestrator.
# For a chosen model: first a WARMUP run (fully build the engine once to
# warm the OS page cache + FusedAdam JIT + CUDA), then each fault method
# (ds_native / checkfreq / memory / ipc) in turn so none of them pays the
# cold-start cost.
#
# Mode is the first positional arg (default single). It sets GPU_IDS and
# is exported as MODE so each per-method script picks the matching report/
# checkpoint dirs and master-port base:
#   ./run_e2e_ds_model_suite.sh [single|multi]
#
# multi keeps the same params as the standalone per-method multi runs
# (TOTAL_STEPS=2000, CHECKPOINT_INTERVAL=100, GPU_IDS=0,1) so results stay
# comparable. Override any value from the environment as usual.
# =====================================================================

SUITE_MODE="${1:-single}"
case "${SUITE_MODE}" in
  single) GPU_IDS="${GPU_IDS:-0}" ;;
  multi)  GPU_IDS="${GPU_IDS:-0,1}" ;;
  *) echo "Unknown mode '${SUITE_MODE}'. Usage: $0 [single|multi]" >&2; exit 1 ;;
esac

METHODS="${METHODS:-ds_native checkfreq memory ipc}"
TOTAL_STEPS="${TOTAL_STEPS:-2000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"

# --- shared config (exported so every per-method script is consistent) ---
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQ_LEN="${SEQ_LEN:-1024}"
ZERO_STAGE="${ZERO_STAGE:-3}"
DTYPE="${DTYPE:-bfloat16}"
RECOMPUTATION="${RECOMPUTATION:-1}"
FAULT_FREQUENCY="${FAULT_FREQUENCY:-1}"
FAULT_CHECK_INTERVAL_SECONDS="${FAULT_CHECK_INTERVAL_SECONDS:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"

METHODS="${METHODS//,/ }"

# MODE is exported so each per-method script resolves the matching mode
# (single vs multi) report/checkpoint dirs and master-port base.
export MODE="${SUITE_MODE}"
export PYTHON_BIN GPU_IDS
export TOTAL_STEPS CHECKPOINT_INTERVAL BATCH_SIZE SEQ_LEN ZERO_STAGE DTYPE
export RECOMPUTATION FAULT_FREQUENCY FAULT_CHECK_INTERVAL_SECONDS LOG_INTERVAL

echo "[mode] ${SUITE_MODE}: gpu_ids=${GPU_IDS} methods='${METHODS}' total_steps=${TOTAL_STEPS} checkpoint_interval=${CHECKPOINT_INTERVAL}"

run_warmup_for_model() {
  local label="$1" pretty_label="$2" model_path="$3"
  echo "================================================================"
  echo "[suite] warmup label=${label} model=${model_path} gpu_ids=${GPU_IDS}"
  echo "================================================================"
  MODEL_LABEL="${label}" MODEL_PRETTY_LABEL="${pretty_label}" MODEL_PATH="${model_path}" \
    bash "${SCRIPT_DIR}/run_e2e_ds_engine_init.sh"
}

run_method() {
  local method="$1" label="$2" pretty_label="$3" model_path="$4"
  local method_script
  case "${method}" in
    ds_native) method_script="${SCRIPT_DIR}/run_e2e_ds_native_three_models.sh" ;;
    checkfreq) method_script="${SCRIPT_DIR}/run_e2e_checkfreq_three_models.sh" ;;
    memory)    method_script="${SCRIPT_DIR}/run_e2e_memory_three_models.sh" ;;
    ipc)       method_script="${SCRIPT_DIR}/run_e2e_ipc_three_models.sh" ;;
    *) echo "Unknown method: ${method}" >&2; exit 1 ;;
  esac

  echo "----------------------------------------------------------------"
  echo "[suite] method=${method} label=${label} model=${model_path} gpu_ids=${GPU_IDS}"
  echo "----------------------------------------------------------------"
  MODEL_LABEL="${label}" MODEL_PRETTY_LABEL="${pretty_label}" MODEL_PATH="${model_path}" \
    bash "${method_script}"
}

run_all_methods_for_model() {
  local label="$1" pretty_label="$2" model_path="$3"
  run_warmup_for_model "${label}" "${pretty_label}" "${model_path}"
  for method in ${METHODS}; do
    run_method "${method}" "${label}" "${pretty_label}" "${model_path}"
  done
}

# --- model to run (per mode) ----------------------------------------
# single -> qwen3 1.7B (current suite default); multi -> llama3_2 3B (matches
# the standalone per-method multi runs, so multi results stay comparable).
if [[ "${SUITE_MODE}" == "single" ]]; then
  run_all_methods_for_model "qwen3_1_7b" "qwen3 1_7B" "/public/huggingface-models/Qwen/Qwen3-1.7B"
  #run_all_methods_for_model "llama3_2-1b" "llama3_2 1B" "/public/huggingface-models/meta-llama/Llama-3.2-1B"
  #run_all_methods_for_model "llama3_2-3b" "llama3_2 3B" "/public/huggingface-models/meta-llama/Llama-3.2-3B"
else
  run_all_methods_for_model "llama3_2-3b" "llama3_2 3B" "/public/huggingface-models/meta-llama/Llama-3.2-3B"
  # run_all_methods_for_model "qwen2.5-14b" "qwen2.5 14B" "/public/huggingface-models/Qwen/Qwen2.5-14B"
fi
