#!/usr/bin/env python3
"""Measure steady-state DeepSpeed training overhead with and without IPC (single + multi GPU)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
FAULT_MP_DIR = WORKSPACE_ROOT / "IPC" / "ds_script" / "fault"
DEFAULT_REPORT_ROOT = SCRIPT_DIR / "report1_multigpu"
DEFAULT_BASELINE_NAME = "baseline"
DEFAULT_IPC_NAME = "ipc"


def add_fault_mp_path() -> None:
    if str(FAULT_MP_DIR) not in sys.path:
        sys.path.insert(0, str(FAULT_MP_DIR))


def load_bench_modules() -> tuple[Any, Any]:
    add_fault_mp_path()
    import ds_hf_checkpoint_bench as native
    import ipc_hf_attach_bench as ipc

    return native, ipc


def parse_args() -> argparse.Namespace:
    native, _ = load_bench_modules()

    parser = argparse.ArgumentParser(
        description=(
            "Measure steady-state DeepSpeed training overhead with and without IPC "
            "on multiple local GPUs. This lab3 benchmark does not save checkpoints "
            "or simulate recovery."
        )
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=("baseline", "ipc"),
        help="Training method to measure.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Local HF causal LM path or model identifier.",
    )
    parser.add_argument(
        "--model-label",
        default=None,
        help="Short label used in report filenames and plots.",
    )
    parser.add_argument(
        "--gpu-ids",
        default=os.environ.get("GPU_IDS", "2,3"),
        help="Comma-separated local physical GPU ids, for example 2,3 or 0,1,2,3.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Optional JSON snapshot path. Defaults to report1_multigpu/<method>/<label>_state.json.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Final JSON report path. Defaults to report1_multigpu/<method>/<label>.json.",
    )
    parser.add_argument(
        "--ipc-group",
        default=None,
        help="IPC tensor group prefix used only when --method ipc.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29871,
        help="Port used for the multi-rank DeepSpeed runtime.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Print one training log every N timed steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=10,
        help="How many timed optimizer steps to run.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Optional untimed warmup steps before the measured epoch.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=native.DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--dataset-split",
        choices=("train", "validation", "test"),
        default=native.DATASET_SPLIT,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--zero-stage", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        "--recomputation",
        dest="gradient_checkpointing",
        action="store_true",
        help="Enable activation recomputation via the model's gradient checkpointing support.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument(
        "--keep-ipc-server",
        action="store_true",
        help="Do not ask the IPC daemons to exit after an IPC run.",
    )
    return parser.parse_args()


def normalize_run_paths(args: argparse.Namespace, native: Any) -> argparse.Namespace:
    label = native.model_label_from_arg(args.model, args.model_label)
    method_dir = DEFAULT_REPORT_ROOT / args.method
    if args.state_file is None:
        args.state_file = method_dir / f"{label}_state.json"
    if args.report_file is None:
        args.report_file = method_dir / f"{label}.json"
    return args


def resolve_ipc_group(args: argparse.Namespace, native: Any) -> str:
    if args.ipc_group:
        return str(args.ipc_group)
    label = native.model_label_from_arg(args.model, args.model_label).replace(" ", "_")
    return f"lab3_ipc_multigpu_{label}"


def validate_args(args: argparse.Namespace, native: Any) -> None:
    native.validate_args(args)
    native.parse_gpu_ids(args.gpu_ids)
    if args.warmup_steps < 0:
        native.fail("--warmup-steps must be non-negative.")
    if args.method == "ipc" and args.zero_stage != 3:
        native.fail("IPC training overhead benchmark currently requires --zero-stage 3.")


def clear_ipc_env() -> None:
    for key in (
        "DEEPSPEED_ZERO3_IPC_GROUP",
        "DEEPSPEED_ZERO3_IPC_TOOL_MODULE",
        "DEEPSPEED_ZERO3_IPC_REQUIRE_EXISTING",
        "DEEPSPEED_ZERO3_IPC_FAULT_SUBGROUP",
        "DEEPSPEED_ZERO3_IPC_FAULT_STAGE",
        "DEEPSPEED_ZERO3_IPC_FAULT_EXIT_CODE",
    ):
        os.environ.pop(key, None)


def sample_gpu_used_gib_avg(torch: Any, native: Any, device: Any) -> float:
    """Sample steady-state GPU memory: return used memory (GiB) averaged across ranks."""
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return 0.0
    native.sync_device(torch, device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    used_bytes = float(total_bytes - free_bytes)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sum_tensor = torch.tensor([used_bytes], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(sum_tensor, op=torch.distributed.ReduceOp.SUM)
        used_bytes = float(sum_tensor.item() / max(torch.distributed.get_world_size(), 1))
    return used_bytes / (1024 ** 3)


def run_warmup(
    native: Any,
    ipc: Any,
    torch: Any,
    engine: Any,
    blocks: list[dict[str, list[int]]],
    batch_size: int,
    seq_len: int,
    pad_token_id: int,
    warmup_steps: int,
    log_interval: int,
) -> float:
    if warmup_steps <= 0:
        return 0.0

    available_steps = ipc.full_batches_per_epoch(blocks, batch_size)
    native.distributed_barrier(torch)
    native.sync_device(torch, engine.device)
    last_loss = 0.0

    for step_idx in range(warmup_steps):
        batch_index = step_idx % available_steps
        batch = native.build_batch_from_blocks(
            torch,
            blocks,
            batch_index,
            batch_size,
            seq_len,
            pad_token_id,
            engine.device,
        )
        last_loss = ipc.run_one_step(torch, engine, batch, 0, step_idx, warmup_steps)
        if log_interval > 0 and ((step_idx + 1) % log_interval == 0 or step_idx + 1 == warmup_steps):
            native.log(f"[warmup] step={step_idx + 1}/{warmup_steps} loss={last_loss:.6f}")

    native.sync_device(torch, engine.device)
    native.distributed_barrier(torch)
    return float(last_loss)


def run_measured_epoch(
    native: Any,
    ipc: Any,
    torch: Any,
    engine: Any,
    blocks: list[dict[str, list[int]]],
    batch_size: int,
    seq_len: int,
    pad_token_id: int,
    steps_per_epoch: int,
    epoch_idx: int,
    log_interval: int,
) -> tuple[float, float, float]:
    """Run the timed measured epoch and return epoch seconds, last loss, and midpoint GPU memory."""
    native.distributed_barrier(torch)
    native.sync_device(torch, engine.device)
    epoch_start = time.perf_counter()
    last_loss = 0.0
    sample_step = max(1, (steps_per_epoch + 1) // 2)
    mid_gib_avg: float | None = None

    for step_idx in range(steps_per_epoch):
        batch = native.build_batch_from_blocks(
            torch,
            blocks,
            step_idx,
            batch_size,
            seq_len,
            pad_token_id,
            engine.device,
        )
        last_loss = ipc.run_one_step(torch, engine, batch, epoch_idx, step_idx, steps_per_epoch)
        current_step = step_idx + 1
        if current_step == sample_step:
            mid_gib_avg = sample_gpu_used_gib_avg(torch, native, engine.device)
            native.log(
                f"[memory] epoch={epoch_idx} midpoint_step={current_step}/{steps_per_epoch} "
                f"used_gib_avg={mid_gib_avg:.4f}"
            )
        if log_interval > 0 and (current_step % log_interval == 0 or current_step == steps_per_epoch):
            native.log(
                f"[train] epoch={epoch_idx} step={current_step}/{steps_per_epoch} loss={last_loss:.6f}"
            )

    native.sync_device(torch, engine.device)
    native.distributed_barrier(torch)
    epoch_seconds = time.perf_counter() - epoch_start
    epoch_seconds = native.max_across_ranks(torch, epoch_seconds, engine.device)

    native.log(
        f"[train] epoch={epoch_idx} completed epoch_seconds={epoch_seconds:.6f} "
        f"last_loss={last_loss:.6f}"
    )
    if mid_gib_avg is None:
        mid_gib_avg = sample_gpu_used_gib_avg(torch, native, engine.device)
    return float(epoch_seconds), float(last_loss), float(mid_gib_avg)


def common_report_metadata(
    args: argparse.Namespace,
    native: Any,
    stats: dict[str, Any],
    epoch_step_count: int,
) -> dict[str, Any]:
    return {
        "experiment": "train_overhead",
        "method": args.method,
        "model": args.model,
        "model_label": native.model_label_from_arg(args.model, args.model_label),
        "dtype": args.dtype,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "recomputation": bool(args.gradient_checkpointing),
        "zero_stage": args.zero_stage,
        "zero3_optimizer_subgroup_count": int(stats.get("zero3_optimizer_subgroup_count", 0)),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "warmup_steps": int(args.warmup_steps),
        "steps_per_epoch": int(epoch_step_count),
    }


def finalize_train_metrics(
    payload: dict[str, Any],
    epoch_seconds: float,
    mid_gib_avg: float,
) -> dict[str, Any]:
    steps = max(int(payload["steps_per_epoch"]), 1)
    payload["train_step_milliseconds"] = float(epoch_seconds) / steps * 1000.0
    payload["mid_training_gpu_used_gib_avg"] = float(mid_gib_avg)
    return payload


def run_baseline(args: argparse.Namespace, native: Any, ipc: Any) -> dict[str, Any]:
    clear_ipc_env()
    torch, deepspeed, auto_model_cls, auto_tokenizer_cls, load_dataset_fn = native.load_runtime_modules()
    device = native.init_distributed(args, torch, deepspeed)
    native.seed_everything(args.seed, torch)

    model = native.load_model(args, torch, auto_model_cls)
    engine = ipc.create_engine(args, torch, deepspeed, model)

    tokenizer = native.load_tokenizer(args, auto_tokenizer_cls)
    blocks, _ = native.build_token_blocks(args, tokenizer, load_dataset_fn)
    available_step_count = ipc.full_batches_per_epoch(blocks, args.batch_size)
    epoch_step_count = native.resolve_epoch_steps(available_step_count, args.steps_per_epoch)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        native.fail("Tokenizer must provide either pad_token_id or eos_token_id.")

    stats = {**ipc.summarize_zero3_layout(engine)}

    native.log(
        f"[baseline-multigpu] model={native.model_label_from_arg(args.model, args.model_label)} "
        f"gpus={args.gpu_ids} steps_per_epoch={epoch_step_count} "
        f"available_steps_per_epoch={available_step_count}"
    )

    try:
        run_warmup(
            native, ipc, torch, engine, blocks,
            args.batch_size, args.seq_len, int(pad_token_id),
            args.warmup_steps, args.log_interval,
        )
        epoch_seconds, _, mid_gib_avg = run_measured_epoch(
            native, ipc, torch, engine, blocks,
            args.batch_size, args.seq_len, int(pad_token_id),
            epoch_step_count, 1, args.log_interval,
        )
    finally:
        native.destroy_engine(torch, engine)
        native.destroy_process_group(torch)

    payload = common_report_metadata(args, native, stats, epoch_step_count)
    return finalize_train_metrics(payload, epoch_seconds, mid_gib_avg)


def run_ipc(args: argparse.Namespace, native: Any, ipc: Any) -> dict[str, Any]:
    clear_ipc_env()
    (
        torch,
        deepspeed,
        auto_model_cls,
        auto_tokenizer_cls,
        _auto_config_cls,
        load_dataset_fn,
        tool_cls,
        ipc_socket_cls,
        socket_path_for_gpu_id_fn,
    ) = ipc.load_runtime_modules()
    device = native.init_distributed(args, torch, deepspeed)
    native.seed_everything(args.seed, torch)

    group_name = resolve_ipc_group(args, native)

    tool = ipc.build_tool(device, tool_cls)
    if ipc.optimizer_ipc_group_exists(tool, group_name):
        native.fail(f"IPC group '{group_name}' already exists. Use a fresh group name.")

    model = ipc.build_creator_model(args, torch, auto_model_cls)

    ipc.configure_zero3_ipc_env(group_name, "ds_tool", require_existing=False)
    ipc.configure_zero3_ipc_fault_env(args, enabled=False)
    engine = ipc.create_engine(args, torch, deepspeed, model)

    tokenizer = native.load_tokenizer(args, auto_tokenizer_cls)
    blocks, _ = native.build_token_blocks(args, tokenizer, load_dataset_fn)
    available_step_count = ipc.full_batches_per_epoch(blocks, args.batch_size)
    epoch_step_count = native.resolve_epoch_steps(available_step_count, args.steps_per_epoch)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        native.fail("Tokenizer must provide either pad_token_id or eos_token_id.")

    stats = {**ipc.summarize_zero3_layout(engine)}

    native.log(
        f"[ipc-multigpu] model={native.model_label_from_arg(args.model, args.model_label)} "
        f"gpus={args.gpu_ids} ipc_group={group_name} steps_per_epoch={epoch_step_count} "
        f"available_steps_per_epoch={available_step_count}"
    )

    try:
        run_warmup(
            native, ipc, torch, engine, blocks,
            args.batch_size, args.seq_len, int(pad_token_id),
            args.warmup_steps, args.log_interval,
        )
        epoch_seconds, _, mid_gib_avg = run_measured_epoch(
            native, ipc, torch, engine, blocks,
            args.batch_size, args.seq_len, int(pad_token_id),
            epoch_step_count, 1, args.log_interval,
        )
    finally:
        ipc.destroy_training_state(torch, engine)
        if not args.keep_ipc_server:
            ipc.shutdown_ipc_servers(ipc_socket_cls, socket_path_for_gpu_id_fn, native.parse_gpu_ids(args.gpu_ids))
        clear_ipc_env()

    payload = common_report_metadata(args, native, stats, epoch_step_count)
    return finalize_train_metrics(payload, epoch_seconds, mid_gib_avg)


def maybe_write_json(path: Path, payload: dict[str, Any], native: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    native.maybe_write_json(path, payload)


def main() -> int:
    """Parse CLI args and run the DeepSpeed training overhead benchmark for the chosen method."""
    args = parse_args()
    native, ipc = load_bench_modules()
    args = normalize_run_paths(args, native)
    validate_args(args, native)

    if args.method == "baseline":
        payload = run_baseline(args, native, ipc)
    else:
        payload = run_ipc(args, native, ipc)

    maybe_write_json(args.state_file.resolve(), payload, native)
    maybe_write_json(args.report_file.resolve(), payload, native)
    if native.rank0():
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
