#!/usr/bin/env python3
"""Warm the OS page cache and CUDA context by initializing a vLLM engine for a model."""
from __future__ import annotations

import argparse
import gc
import os
import time
from datetime import datetime, timezone
from typing import Any

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_IPC_TOOL"] = "0"
os.environ.pop("VLLM_DEBUG_SCHEDULE_PREFILL_TOKENS", None)
os.environ.pop("VLLM_KV_CPU_CHECKPOINT_PATH", None)
os.environ.pop("VLLM_KV_CPU_BUFFER_METADATA", None)

from vllm import EngineArgs, LLMEngine


def str2bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Warm the OS page cache by fully initializing a vLLM engine for a model."
        )
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-label", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--trust-remote-code", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--enable-prefix-caching", type=str2bool, default=False)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def maybe_shutdown_engine(engine: LLMEngine) -> None:
    engine_core = getattr(engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if not callable(shutdown):
        return

    try:
        shutdown()
    except Exception as exc:
        print(f"[warmup] shutdown skipped: {exc}", flush=True)


def main() -> int:
    """Entry point: initialize a vLLM engine to warm the page cache and report timings."""
    args = parse_args()
    process_start_time = time.time()
    engine_init_start_time = time.time()

    engine_args = EngineArgs(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
        disable_log_stats=True,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
    )

    engine: LLMEngine | None = None
    try:
        engine = LLMEngine.from_engine_args(engine_args)
        engine_ready_time = time.time()
        print(
            "[warmup] "
            f"model={args.model} "
            f"model_label={args.model_label or args.model} "
            f"engine_init_seconds={engine_ready_time - engine_init_start_time:.6f} "
            f"total_process_seconds={engine_ready_time - process_start_time:.6f}",
            flush=True,
        )
    finally:
        cleanup_start_time = time.time()
        if engine is not None:
            maybe_shutdown_engine(engine)
            del engine
        gc.collect()
        cleanup_end_time = time.time()
        print(
            "[warmup] "
            f"cleanup_seconds={cleanup_end_time - cleanup_start_time:.6f} "
            f"finished_utc={datetime.now(timezone.utc).isoformat()}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
