#!/usr/bin/env python3
"""Warm the OS page cache (+ DeepSpeed JIT + CUDA context) for a model before the e2e suite."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
FAULT_DIR = WORKSPACE_ROOT / "IPC" / "ds_script" / "fault"


def load_native_module() -> Any:
    if str(FAULT_DIR) not in sys.path:
        sys.path.insert(0, str(FAULT_DIR))
    import ds_hf_checkpoint_bench as native

    return native


def str2bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm OS page cache / DeepSpeed JIT / CUDA by building a DeepSpeed engine once.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--gpu-ids", default="0", help="Comma-separated physical GPU ids (=1 id -> single).")
    parser.add_argument("--launch-mode", choices=("auto", "single", "torchrun"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--zero-stage", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--gradient-checkpointing", "--recomputation",
        dest="gradient_checkpointing", action="store_true",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--master-port", type=int, default=29800)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    """Entry point: build the DeepSpeed engine once to warm caches/JIT/CUDA, then tear down."""
    args = parse_args()
    native = load_native_module()

    process_start = time.perf_counter()
    engine_init_start = time.perf_counter()
    torch_mod, engine, _, _, _tokenizer, _load_dataset_fn, _stats = native.create_engine(args)
    engine_init_seconds = time.perf_counter() - engine_init_start

    if native.rank0():
        label = native.model_label_from_arg(args.model, args.model_label)
        print(
            f"[warmup] label={label} gpu_ids={args.gpu_ids} "
            f"engine_init_seconds={engine_init_seconds:.3f} "
            f"process_seconds={time.perf_counter() - process_start:.3f}",
            flush=True,
        )

    native.destroy_engine(torch_mod, engine)
    native.destroy_process_group(torch_mod)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
