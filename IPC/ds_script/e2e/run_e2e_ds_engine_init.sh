#!/usr/bin/env bash
set -euo pipefail

# =====================================================================
# DeepSpeed e2e warmup: fully build the engine once to warm the OS page
# cache (model weights) + FusedAdam JIT compile + CUDA context, so the
# first *measured* method does not pay those cold-start costs.
# Single-GPU (one --gpu-id) runs directly; multi-GPU runs under torchrun
# (nproc = number of gpu-ids), matching the method benches.
# Driven by env: MODEL_LABEL, MODEL_PRETTY_LABEL, MODEL_PATH, GPU_IDS.
# =====================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

GPU_IDS="${GPU_IDS:-0}"
MASTER_PORT="${WARMUP_MASTER_PORT:-29800}"
ZERO_STAGE="${ZERO_STAGE:-3}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-1024}"
RECOMPUTATION="${RECOMPUTATION:-1}"

if [[ -z "${MODEL_LABEL:-}" || -z "${MODEL_PRETTY_LABEL:-}" || -z "${MODEL_PATH:-}" ]]; then
  echo "Set MODEL_LABEL, MODEL_PRETTY_LABEL, and MODEL_PATH before running warmup." >&2
  exit 1
fi

# Derive nproc / launch mode from the GPU_IDS count (one id -> single).
IFS=',' read -ra _GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${#_GPU_ARRAY[@]}"

phase_args=(
  --model "${MODEL_PATH}"
  --model-label "${MODEL_PRETTY_LABEL}"
  --gpu-ids "${GPU_IDS}"
  --zero-stage "${ZERO_STAGE}"
  --dtype "${DTYPE}"
  --batch-size "${BATCH_SIZE}"
  --seq-len "${SEQ_LEN}"
  --local-files-only
)
if [[ "${RECOMPUTATION}" == "1" ]]; then
  phase_args+=(--recomputation)
fi

echo "[warmup-run] label=${MODEL_LABEL} model=${MODEL_PATH} gpu_ids=${GPU_IDS} nproc=${NPROC}"
cleanup() {
  rm -f "$HOME"/.cache/torch_extensions/*/*/lock 2>/dev/null || true  # stale DeepSpeed JIT-compile lock
}
trap cleanup EXIT INT TERM

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

if [[ "${NPROC}" -le 1 ]]; then
  # Single-GPU: direct python (world_size 1 via ensure_single_rank_env).
  "${PYTHON_BIN}" "${SCRIPT_DIR}/warmup_ds_engine_init.py" \
    --launch-mode single --master-port "${MASTER_PORT}" "${phase_args[@]}"
else
  # Multi-GPU: torchrun (ZeRO-3 needs all ranks).
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" \
    "${SCRIPT_DIR}/warmup_ds_engine_init.py" \
    --launch-mode torchrun --master-port "${MASTER_PORT}" "${phase_args[@]}"
fi
