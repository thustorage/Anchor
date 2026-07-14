#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp

import ds_hf_checkpoint_bench as native
from checkfreq_hf_checkpoint_bench import (
    load_model_state,
    move_optimizer_state_to_device,
    prepare_optimizer_state_for_load,
    snapshot_model_state,
)


CREATOR_FAILURE_EXIT_CODE = native.CREATOR_FAILURE_EXIT_CODE
FAULT_EPOCH = native.FAULT_EPOCH
RESUME_EPOCHS = native.RESUME_EPOCHS
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_ROOT = SCRIPT_DIR / "pccheck_ckpt"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "report_pccheck"
CHECKPOINT_FILE_NAME = "pccheck_checkpoint.chk"
AUX_STATE_FILE_NAME = "aux_state.pt"
OFFSET_SIZE_INTS = 4096
INT_BYTES = 4
FLOAT32_SLOT_BYTES = 4
RECOVERY_META_OFFSET_BYTES = 16 * INT_BYTES
RECOVERY_META_STRUCT = struct.Struct("<qq")
FLAT_INDEX_KEY = "__pccheck_flat_index__"
TUPLE_KEY = "__pccheck_tuple__"
RAW_TENSOR_ALIGNMENT_BYTES = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one HF causal LM with DeepSpeed, persist a PCcheck parallel "
            "checkpoint at the failure boundary, relaunch a connector, and "
            "report snapshot/recovery timing."
        )
    )
    parser.add_argument("--phase", choices=["creator", "connector", "all"], default="all")
    parser.add_argument("--model",
                        required=True,
                        help="Local HF causal LM path or model identifier.")
    parser.add_argument("--model-label",
                        default=None,
                        help="Short label used in report filenames and JSON output.")
    parser.add_argument("--checkpoint-dir",
                        type=Path,
                        default=DEFAULT_CHECKPOINT_ROOT,
                        help="Directory used to save and reload PCcheck checkpoints.")
    parser.add_argument("--state-file",
                        type=Path,
                        default=DEFAULT_REPORT_DIR / "state.json",
                        help="Creator/connector shared state JSON. Kept outside ckpt to avoid accidental deletion.")
    parser.add_argument("--report-file",
                        type=Path,
                        default=DEFAULT_REPORT_DIR / "report.json",
                        help="Final JSON report path.")
    parser.add_argument("--master-port",
                        type=int,
                        default=29651,
                        help="Port used for the single-rank DeepSpeed runtime.")
    parser.add_argument("--c-lib-path",
                        type=Path,
                        default=Path("/root/mhy/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"),
                        help="Path to the compiled PCcheck shared library.")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--max-async", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-interval",
                        type=int,
                        default=0,
                        help="Print one training log every N steps. Use 0 to disable.")
    parser.add_argument("--steps-per-epoch",
                        type=int,
                        default=10,
                        help="How many optimizer steps to run in one benchmark epoch.")
    parser.add_argument("--dataset-dir",
                        type=Path,
                        default=native.DEFAULT_DATASET_DIR,
                        help="Local directory containing WikiText-2 parquet shards.")
    parser.add_argument("--dataset-split",
                        choices=("train", "validation", "test"),
                        default=native.DATASET_SPLIT,
                        help="Local parquet split used for timed training.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--zero-stage", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--dtype",
                        choices=("auto", "float16", "bfloat16", "float32"),
                        default="bfloat16")
    parser.add_argument("--gradient-checkpointing",
                        action="store_true",
                        help="Enable gradient checkpointing when the model supports it.")
    parser.add_argument("--trust-remote-code",
                        action="store_true",
                        help="Forwarded to transformers.from_pretrained.")
    parser.add_argument("--local-files-only",
                        action="store_true",
                        help="Avoid remote downloads when loading the model.")
    parser.add_argument("--attn-implementation",
                        default=None,
                        help="Optional transformers attention implementation override.")
    parser.add_argument("--keep-checkpoints",
                        action="store_true",
                        help="Keep checkpoint files after the final report is written.")
    parser.add_argument("--fault-start-time",
                        type=float,
                        default=None,
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def normalize_run_paths(args: argparse.Namespace) -> argparse.Namespace:
    label = native.model_label_from_arg(args.model, args.model_label)
    if args.checkpoint_dir == DEFAULT_CHECKPOINT_ROOT:
        args.checkpoint_dir = DEFAULT_CHECKPOINT_ROOT / label
    if args.state_file == DEFAULT_REPORT_DIR / "state.json":
        args.state_file = DEFAULT_REPORT_DIR / f"{label}_state.json"
    if args.report_file == DEFAULT_REPORT_DIR / "report.json":
        args.report_file = DEFAULT_REPORT_DIR / f"{label}.json"
    return args


def validate_args(args: argparse.Namespace) -> None:
    native.validate_args(args)
    if args.num_threads <= 0 or args.max_async <= 0:
        native.fail("PCcheck thread and async parameters must be positive.")
    if not args.c_lib_path.exists():
        native.fail(f"PCcheck shared library not found: {args.c_lib_path}")


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def destroy_engine(torch_mod: Any, engine: Any) -> None:
    del engine
    gc.collect()
    if torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()


def clone_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(item) for item in value)
    return value


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if dtype is None:
        native.fail(f"Unsupported dtype in checkpoint metadata: {name}")
    return dtype


def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def tensor_byte_view(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "untyped_storage"):
        byte_tensor = torch.empty(0, dtype=torch.uint8, device=tensor.device)
        return byte_tensor.set_(
            tensor.untyped_storage(),
            tensor.storage_offset() * tensor.element_size(),
            (tensor.numel() * tensor.element_size(),),
            (1,),
        )
    try:
        return tensor.view(torch.uint8)
    except (AttributeError, RuntimeError, TypeError) as exc:
        native.fail(f"Unable to reinterpret tensor storage as raw bytes: {exc}")


def shape_numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def tensor_from_byte_view(byte_tensor: torch.Tensor,
                          dtype: torch.dtype,
                          shape: list[int]) -> torch.Tensor:
    element_size = torch.empty((), dtype=dtype).element_size()
    numel = shape_numel(shape)
    expected_bytes = numel * element_size
    if byte_tensor.numel() != expected_bytes:
        native.fail(
            "Raw checkpoint slice size does not match tensor metadata: "
            f"expected {expected_bytes} bytes, got {byte_tensor.numel()}."
        )

    if hasattr(byte_tensor, "untyped_storage"):
        storage_offset_bytes = int(byte_tensor.storage_offset())
        if storage_offset_bytes % element_size != 0:
            native.fail(
                "Raw checkpoint slice is not aligned for dtype reconstruction: "
                f"offset_bytes={storage_offset_bytes}, element_size={element_size}."
            )
        typed_tensor = torch.empty(0, dtype=dtype, device=byte_tensor.device)
        typed_tensor = typed_tensor.set_(
            byte_tensor.untyped_storage(),
            storage_offset_bytes // element_size,
            (numel,),
            (1,),
        )
        return typed_tensor.view(shape)

    try:
        return byte_tensor.view(dtype).view(shape)
    except (AttributeError, RuntimeError, TypeError) as exc:
        native.fail(f"Unable to rebuild tensor from raw bytes: {exc}")


def resolve_flat_tensor_size(aux_payload: dict[str, Any], flat_entries: list[dict[str, Any]]) -> int:
    stored_size = int(aux_payload.get("flat_total_size", 0) or 0)
    if stored_size > 0:
        return stored_size

    stored_bytes = int(aux_payload.get("flat_total_bytes", 0) or 0)
    if stored_bytes > 0:
        return max(1, (stored_bytes + FLOAT32_SLOT_BYTES - 1) // FLOAT32_SLOT_BYTES)

    total_slots = 0
    total_bytes = 0
    for entry in flat_entries:
        if "byte_offset" in entry and "num_bytes" in entry:
            total_bytes = max(total_bytes, int(entry["byte_offset"]) + int(entry["num_bytes"]))
            continue
        total_slots = max(total_slots, int(entry["offset"]) + int(entry["numel"]))

    if total_bytes > 0:
        return max(1, (total_bytes + FLOAT32_SLOT_BYTES - 1) // FLOAT32_SLOT_BYTES)
    return total_slots


def encode_structure(obj: Any, flat_sources: list[dict[str, Any]]) -> Any:
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            flat_sources.append(
                {
                    "tensor": obj.detach(),
                    "shape": list(obj.shape),
                    "numel": int(obj.numel()),
                    "element_size": int(obj.element_size()),
                    "dtype": dtype_name(obj.dtype),
                }
            )
            return {FLAT_INDEX_KEY: len(flat_sources) - 1}
        return clone_to_cpu(obj)
    if isinstance(obj, dict):
        return {key: encode_structure(value, flat_sources) for key, value in obj.items()}
    if isinstance(obj, list):
        return [encode_structure(value, flat_sources) for value in obj]
    if isinstance(obj, tuple):
        return {TUPLE_KEY: [encode_structure(value, flat_sources) for value in obj]}
    return obj


def flatten_checkpoint_payload(model_state: dict[str, Any] | None,
                               optimizer_state: dict[str, Any],
                               device: torch.device) -> tuple[dict[str, Any], torch.Tensor, int]:
    flat_sources: list[dict[str, Any]] = []
    encoded_model = encode_structure(model_state, flat_sources)
    encoded_optimizer = encode_structure(optimizer_state, flat_sources)

    flat_entries: list[dict[str, Any]] = []
    for entry in flat_sources:
        byte_offset = align_up(
            int(flat_entries[-1]["byte_offset"]) + int(flat_entries[-1]["num_bytes"])
            if flat_entries else 0,
            RAW_TENSOR_ALIGNMENT_BYTES,
        )
        num_bytes = int(entry["numel"]) * int(entry["element_size"])
        flat_entries.append(
            {
                "byte_offset": byte_offset,
                "num_bytes": num_bytes,
                "numel": int(entry["numel"]),
                "shape": list(entry["shape"]),
                "dtype": str(entry["dtype"]),
            }
        )
    total_bytes = (
        int(flat_entries[-1]["byte_offset"]) + int(flat_entries[-1]["num_bytes"])
        if flat_entries else 0
    )
    total_size = max(1, (total_bytes + FLOAT32_SLOT_BYTES - 1) // FLOAT32_SLOT_BYTES)
    if device.type == "cpu":
        try:
            flat_tensor = torch.empty(total_size, dtype=torch.float32, device=device, pin_memory=True)
            flat_tensor.zero_()
        except RuntimeError:
            flat_tensor = torch.zeros(total_size, dtype=torch.float32, device=device)
    else:
        flat_tensor = torch.zeros(total_size, dtype=torch.float32, device=device)
    flat_tensor_bytes = tensor_byte_view(flat_tensor)
    for entry, meta in zip(flat_sources, flat_entries):
        source_tensor = entry["tensor"].detach()
        if source_tensor.device != device:
            source_tensor = source_tensor.to(device=device)
        if not source_tensor.is_contiguous():
            source_tensor = source_tensor.contiguous()
        source_bytes = tensor_byte_view(source_tensor)
        start = int(meta["byte_offset"])
        end = start + int(meta["num_bytes"])
        flat_tensor_bytes[start:end].copy_(source_bytes)

    aux_payload = {
        "model_state": encoded_model,
        "optimizer_state": encoded_optimizer,
        "flat_entries": flat_entries,
        "flat_total_bytes": int(total_bytes),
        "flat_total_size": int(total_size),
        "flat_encoding": "raw_bytes_v2",
    }
    return aux_payload, flat_tensor, int(total_size)


def decode_structure(obj: Any,
                     flat_tensor: torch.Tensor,
                     flat_entries: list[dict[str, Any]],
                     flat_tensor_bytes: torch.Tensor | None = None) -> Any:
    if flat_tensor_bytes is None:
        flat_tensor_bytes = tensor_byte_view(flat_tensor)
    if isinstance(obj, dict):
        if FLAT_INDEX_KEY in obj:
            meta = flat_entries[int(obj[FLAT_INDEX_KEY])]
            if "byte_offset" in meta and "num_bytes" in meta:
                start = int(meta["byte_offset"])
                end = start + int(meta["num_bytes"])
                tensor = tensor_from_byte_view(
                    flat_tensor_bytes[start:end],
                    dtype_from_name(str(meta["dtype"])),
                    list(meta["shape"]),
                )
                return tensor.clone()
            start = int(meta["offset"])
            end = start + int(meta["numel"])
            tensor = flat_tensor[start:end].view(meta["shape"]).to(dtype=dtype_from_name(str(meta["dtype"])))
            return tensor.clone()
        if TUPLE_KEY in obj:
            return tuple(
                decode_structure(value, flat_tensor, flat_entries, flat_tensor_bytes)
                for value in obj[TUPLE_KEY]
            )
        return {
            key: decode_structure(value, flat_tensor, flat_entries, flat_tensor_bytes)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [
            decode_structure(value, flat_tensor, flat_entries, flat_tensor_bytes)
            for value in obj
        ]
    return obj


def persist_torch_payload(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    with open(tmp_path, "rb") as handle:
        import os
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def checkpoint_size_bytes(checkpoint_dir: Path, apparent: bool = False) -> int:
    return native.path_size_bytes(checkpoint_dir, apparent=apparent)


def add_pccheck_repo_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    pccheck_root = repo_root / "pccheck"
    if str(pccheck_root) not in sys.path:
        sys.path.insert(0, str(pccheck_root))
    return pccheck_root


def build_chk_monitor(args: argparse.Namespace,
                      checkpoint_file: Path,
                      total_size: int,
                      flat_tensor: torch.Tensor):
    add_pccheck_repo_path()

    from checkpoint_eval.pccheck.chk_monitor import Chk_monitor

    return Chk_monitor(
        str(args.c_lib_path),
        total_size,
        args.num_threads,
        args.max_async,
        True,
        gpu_ar=flat_tensor,
        bsize=max(1, total_size),
        memory_saving=True,
        checkpoint_path=str(checkpoint_file),
        model={},
        optimizer={},
    )


def checkpoint_slot_padding_bytes(total_size: int) -> int:
    payload_bytes = total_size * FLOAT32_SLOT_BYTES
    remainder = payload_bytes % 4096
    return 4096 - remainder if remainder != 0 else 4096


def allocate_cpu_staging_tensor(total_size: int) -> torch.Tensor:
    try:
        return torch.empty(total_size, dtype=torch.float32, pin_memory=True)
    except RuntimeError:
        return torch.empty(total_size, dtype=torch.float32)


def read_latest_pccheck_tensor(checkpoint_file: Path, total_size: int, max_async: int) -> torch.Tensor:
    if not checkpoint_file.exists():
        native.fail(f"PCcheck checkpoint file not found: {checkpoint_file}")

    with checkpoint_file.open("rb") as handle:
        handle.seek(RECOVERY_META_OFFSET_BYTES)
        area, counter = RECOVERY_META_STRUCT.unpack(handle.read(RECOVERY_META_STRUCT.size))
        if area < 0 or counter <= 0:
            native.fail("PCcheck recovery metadata does not point to a completed checkpoint.")

        data_region_offset = (max_async + 3) * OFFSET_SIZE_INTS * INT_BYTES
        slot_size_bytes = total_size * FLOAT32_SLOT_BYTES + checkpoint_slot_padding_bytes(total_size)
        slot_data_offset = data_region_offset + area * slot_size_bytes
        handle.seek(slot_data_offset)
        payload_bytes = total_size * FLOAT32_SLOT_BYTES
        payload = bytearray(handle.read(payload_bytes))
        if len(payload) != payload_bytes:
            native.fail("Incomplete PCcheck payload while reading checkpoint data.")

    source_tensor = torch.frombuffer(payload, dtype=torch.float32)
    cpu_tensor = allocate_cpu_staging_tensor(total_size)
    cpu_tensor.copy_(source_tensor)
    return cpu_tensor


def report_common_metadata(args: argparse.Namespace, stats: dict[str, Any]) -> dict[str, Any]:
    payload = native.report_common_metadata(args, stats)
    payload["checkpoint_backend"] = "pccheck_parallel"
    payload["pccheck_num_threads"] = int(args.num_threads)
    payload["pccheck_max_async"] = int(args.max_async)
    payload["c_lib_path"] = str(args.c_lib_path.resolve())
    return payload


def run_creator(args: argparse.Namespace) -> int:
    """Train to the fault boundary and persist a PCcheck checkpoint."""
    checkpoint_dir = args.checkpoint_dir.resolve()
    state_file = args.state_file.resolve()
    clear_directory(checkpoint_dir)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    if state_file.exists():
        state_file.unlink()

    checkpoint_file = checkpoint_dir / CHECKPOINT_FILE_NAME
    aux_state_path = checkpoint_dir / AUX_STATE_FILE_NAME

    torch_mod, engine, _, _, tokenizer, load_dataset_fn, stats = native.create_engine(args)
    blocks, data_stats = native.build_token_blocks(args, tokenizer, load_dataset_fn)
    available_step_count = native.batches_per_epoch(blocks, args.batch_size)
    epoch_step_count = native.resolve_epoch_steps(available_step_count, args.steps_per_epoch)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        native.fail("Tokenizer must provide either pad_token_id or eos_token_id.")
    model_label = native.model_label_from_arg(args.model, args.model_label)

    native.log(
        f"[creator] model={model_label} "
        f"steps_per_epoch={epoch_step_count} "
        f"available_steps_per_epoch={available_step_count} "
        f"dataset_rows={data_stats['dataset_rows']} "
        f"token_blocks={data_stats['token_blocks']}"
    )

    epoch_seconds = 0.0
    last_loss = 0.0
    chk_monitor = None
    try:
        epoch_seconds, last_loss = native.run_epoch(
            torch_mod,
            engine,
            blocks,
            args.batch_size,
            args.seq_len,
            int(pad_token_id),
            epoch_step_count,
            FAULT_EPOCH,
            args.log_interval,
        )

        native.distributed_barrier(torch_mod)
        native.sync_device(torch_mod, engine.device)
        save_start = time.perf_counter()

        state_extract_start = time.perf_counter()
        model_state = snapshot_model_state(
            engine,
            zero_stage=getattr(engine, "zero_optimization_stage", lambda: 0)(),
        )
        optimizer_state = engine.optimizer.state_dict()
        state_extract_seconds = time.perf_counter() - state_extract_start
        prepare_start = time.perf_counter()
        flatten_start = time.perf_counter()
        checkpoint_payload, flat_tensor, total_size = flatten_checkpoint_payload(
            model_state,
            optimizer_state,
            torch.device("cpu"),
        )
        flatten_seconds = time.perf_counter() - flatten_start
        checkpoint_payload.update(
            {
                "client_state": {
                    "train_epoch": FAULT_EPOCH,
                    "last_loss": last_loss,
                    "model_label": model_label,
                },
                "model_metadata": {
                    "model_label": model_label,
                    "parameter_count": int(stats["parameter_count"]),
                },
            }
        )
        aux_write_start = time.perf_counter()
        persist_torch_payload(checkpoint_payload, aux_state_path)
        aux_state_write_seconds = time.perf_counter() - aux_write_start
        monitor_init_start = time.perf_counter()
        chk_monitor = build_chk_monitor(args, checkpoint_file, total_size, flat_tensor)
        monitor_init_seconds = time.perf_counter() - monitor_init_start
        prepare_seconds = time.perf_counter() - prepare_start

        gpu_snapshot_start = time.perf_counter()
        chk_monitor.save()
        while chk_monitor.gpu_copy_in_progress():
            time.sleep(0.001)
        native.sync_device(torch_mod, engine.device)
        native.distributed_barrier(torch_mod)
        save_seconds = time.perf_counter() - save_start
        save_seconds = native.max_across_ranks(torch_mod, save_seconds, engine.device)
        buffer_handoff_seconds = time.perf_counter() - gpu_snapshot_start
        buffer_handoff_seconds = native.max_across_ranks(torch_mod, buffer_handoff_seconds, engine.device)
        state_extract_seconds = native.max_across_ranks(torch_mod, state_extract_seconds, engine.device)
        prepare_seconds = native.max_across_ranks(torch_mod, prepare_seconds, engine.device)
        flatten_seconds = native.max_across_ranks(torch_mod, flatten_seconds, engine.device)
        aux_state_write_seconds = native.max_across_ranks(torch_mod, aux_state_write_seconds, engine.device)
        monitor_init_seconds = native.max_across_ranks(torch_mod, monitor_init_seconds, engine.device)
        persist_start = time.perf_counter()
        while chk_monitor.persist_pending():
            time.sleep(0.01)
        persist_seconds = time.perf_counter() - persist_start
        persist_seconds = native.max_across_ranks(torch_mod, persist_seconds, engine.device)
        chk_monitor.kill_checkpoint()
        chk_monitor = None
        del flat_tensor
        del model_state
        del optimizer_state
        gc.collect()

        size_bytes = checkpoint_size_bytes(checkpoint_dir, apparent=False)
        save_overhead_seconds = flatten_seconds + aux_state_write_seconds
        adjusted_save_seconds = max(0.0, save_seconds - save_overhead_seconds)
        snapshot_seconds = state_extract_seconds + flatten_seconds + buffer_handoff_seconds
        payload = report_common_metadata(args, stats)
        payload.update({
            "dataset_rows": int(data_stats["dataset_rows"]),
            "nonempty_rows": int(data_stats["nonempty_rows"]),
            "dataset_parquet_files": int(data_stats["parquet_files"]),
            "total_tokens_with_eos": int(data_stats["total_tokens_with_eos"]),
            "token_blocks": int(data_stats["token_blocks"]),
            "tokens_used_for_blocks": int(data_stats["tokens_used_for_blocks"]),
            "padding_tokens_added": int(data_stats["padding_tokens_added"]),
            "remainder_tokens": int(data_stats["remainder_tokens"]),
            "available_steps_per_epoch": int(available_step_count),
            "steps_per_epoch": int(epoch_step_count),
            "checkpoint_tag": f"epoch_{FAULT_EPOCH:06d}",
            "checkpoint_file": CHECKPOINT_FILE_NAME,
            "epoch_train_seconds": float(epoch_seconds),
            "last_loss": float(last_loss),
            "checkpoint_save_seconds": float(adjusted_save_seconds),
            "checkpoint_save_seconds_raw": float(save_seconds),
            "checkpoint_snapshot_seconds": float(snapshot_seconds),
            "checkpoint_state_extract_seconds": float(state_extract_seconds),
            "checkpoint_pack_seconds": float(flatten_seconds),
            "checkpoint_buffer_handoff_seconds": float(buffer_handoff_seconds),
            "checkpoint_prepare_seconds": float(prepare_seconds),
            "checkpoint_aux_state_write_seconds": float(aux_state_write_seconds),
            "checkpoint_monitor_init_seconds": float(monitor_init_seconds),
            "checkpoint_save_overhead_seconds": float(save_overhead_seconds),
            "checkpoint_persist_seconds": float(persist_seconds),
            "checkpoint_size_bytes": int(size_bytes),
            "checkpoint_size_gib": float(size_bytes) / (1024**3),
            "checkpoint_metric_version": 3,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        native.maybe_write_json(state_file, payload)
        native.log(
            f"[creator] model={payload['model_label']} "
            f"snapshot_seconds={payload['checkpoint_snapshot_seconds']:.6f} "
            f"persist_seconds={payload['checkpoint_persist_seconds']:.6f} "
            f"checkpoint_size_gib={payload['checkpoint_size_gib']:.6f}"
        )
        return CREATOR_FAILURE_EXIT_CODE
    finally:
        if chk_monitor is not None:
            while chk_monitor.persist_pending():
                time.sleep(0.01)
            chk_monitor.kill_checkpoint()
        destroy_engine(torch_mod, engine)
        native.destroy_process_group(torch_mod)


def run_connector(args: argparse.Namespace) -> int:
    """Reload the PCcheck checkpoint and resume training, reporting recovery timing."""
    connector_entry_time = time.perf_counter()
    recovery_start_time = args.fault_start_time if args.fault_start_time is not None else connector_entry_time
    state = native.load_json(args.state_file.resolve())

    checkpoint_dir = Path(state["checkpoint_dir"]).resolve()
    checkpoint_file = checkpoint_dir / CHECKPOINT_FILE_NAME
    aux_state_path = checkpoint_dir / AUX_STATE_FILE_NAME
    if not aux_state_path.exists():
        native.fail(f"PCcheck aux state file not found: {aux_state_path}")

    engine_init_start = time.perf_counter()
    torch_mod, engine, _, _, tokenizer, load_dataset_fn, _ = native.create_engine(args)
    checkpoint_load_seconds = 0.0
    checkpoint_load_seconds_raw = 0.0
    recovery_total_seconds = 0.0
    connector_other_seconds = 0.0
    client_state: dict[str, Any] = {}
    aux_state_load_seconds = 0.0
    pccheck_read_seconds = 0.0
    decode_seconds = 0.0
    state_restore_seconds = 0.0

    try:
        engine_init_seconds = time.perf_counter() - engine_init_start
        engine_init_seconds = native.max_across_ranks(torch_mod, engine_init_seconds, engine.device)

        native.distributed_barrier(torch_mod)
        native.sync_device(torch_mod, engine.device)
        load_start = time.perf_counter()
        aux_load_start = time.perf_counter()
        aux_payload = torch.load(aux_state_path, map_location="cpu", weights_only=False)
        aux_state_load_seconds = time.perf_counter() - aux_load_start
        flat_entries = list(aux_payload["flat_entries"])
        total_size = resolve_flat_tensor_size(aux_payload, flat_entries)
        pccheck_read_start = time.perf_counter()
        flat_tensor_cpu = read_latest_pccheck_tensor(checkpoint_file, total_size, args.max_async)
        pccheck_read_seconds = time.perf_counter() - pccheck_read_start

        decode_start = time.perf_counter()
        model_state = decode_structure(aux_payload["model_state"], flat_tensor_cpu, flat_entries)
        optimizer_state = decode_structure(aux_payload["optimizer_state"], flat_tensor_cpu, flat_entries)
        decode_seconds = time.perf_counter() - decode_start
        client_state = dict(aux_payload.get("client_state", {}))
        restore_start = time.perf_counter()
        load_model_state(engine, model_state, args.zero_stage)
        engine.optimizer.load_state_dict(prepare_optimizer_state_for_load(engine.optimizer, optimizer_state))
        move_optimizer_state_to_device(engine.optimizer, engine.device)
        state_restore_seconds = time.perf_counter() - restore_start
        del model_state
        del optimizer_state
        del aux_payload
        del flat_tensor_cpu
        gc.collect()
        native.sync_device(torch_mod, engine.device)
        native.distributed_barrier(torch_mod)
        checkpoint_load_seconds_raw = time.perf_counter() - load_start
        checkpoint_load_seconds_raw = native.max_across_ranks(torch_mod, checkpoint_load_seconds_raw, engine.device)
        aux_state_load_seconds = native.max_across_ranks(torch_mod, aux_state_load_seconds, engine.device)
        pccheck_read_seconds = native.max_across_ranks(torch_mod, pccheck_read_seconds, engine.device)
        decode_seconds = native.max_across_ranks(torch_mod, decode_seconds, engine.device)
        state_restore_seconds = native.max_across_ranks(torch_mod, state_restore_seconds, engine.device)
        load_overhead_seconds = aux_state_load_seconds + decode_seconds
        checkpoint_load_seconds = max(0.0, checkpoint_load_seconds_raw - load_overhead_seconds)

        recovery_total_seconds = time.perf_counter() - recovery_start_time
        recovery_total_seconds = native.max_across_ranks(torch_mod, recovery_total_seconds, engine.device)
        connector_other_seconds = max(0.0, recovery_total_seconds - engine_init_seconds - checkpoint_load_seconds_raw)
        resumed_epoch = int(client_state.get("train_epoch", 0))

        blocks, data_stats = native.build_token_blocks(args, tokenizer, load_dataset_fn)
        available_step_count = native.batches_per_epoch(blocks, args.batch_size)
        epoch_step_count = native.resolve_epoch_steps(available_step_count, args.steps_per_epoch)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_token_id is None:
            native.fail("Tokenizer must provide either pad_token_id or eos_token_id.")

        resume_epoch_seconds = 0.0
        resume_last_loss = 0.0
        for resume_epoch_idx in range(1, RESUME_EPOCHS + 1):
            current_epoch = resumed_epoch + resume_epoch_idx
            resume_epoch_seconds, resume_last_loss = native.run_epoch(
                torch_mod,
                engine,
                blocks,
                args.batch_size,
                args.seq_len,
                int(pad_token_id),
                epoch_step_count,
                current_epoch,
                args.log_interval,
            )
    finally:
        destroy_engine(torch_mod, engine)
        native.destroy_process_group(torch_mod)

    payload = dict(state)
    payload.update({
        "fault_to_recovery_ready_seconds": float(recovery_total_seconds),
        "connector_engine_init_seconds": float(engine_init_seconds),
        "connector_checkpoint_load_seconds": float(checkpoint_load_seconds),
        "connector_aux_state_load_seconds": float(aux_state_load_seconds),
        "connector_pccheck_read_seconds": float(pccheck_read_seconds),
        "connector_decode_seconds": float(decode_seconds),
        "connector_state_restore_seconds": float(state_restore_seconds),
        "connector_other_seconds": float(connector_other_seconds),
        "loaded_train_epoch": int(client_state.get("train_epoch", 0)),
        "resume_dataset_rows": int(data_stats["dataset_rows"]),
        "resume_token_blocks": int(data_stats["token_blocks"]),
        "resume_padding_tokens_added": int(data_stats["padding_tokens_added"]),
        "resume_remainder_tokens": int(data_stats["remainder_tokens"]),
        "resume_available_steps_per_epoch": int(available_step_count),
        "resume_steps_per_epoch": int(epoch_step_count),
        "resume_epoch_seconds": float(resume_epoch_seconds),
        "resume_last_loss": float(resume_last_loss),
        "recovery_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    native.enrich_training_throughput_metrics(payload)
    native.maybe_write_json(args.report_file.resolve(), payload)
    if native.rank0():
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    if not args.keep_checkpoints:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return 0


def build_subprocess_args(args: argparse.Namespace, phase: str) -> list[str]:
    script_path = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(script_path),
        "--phase",
        phase,
        "--model",
        str(args.model),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--state-file",
        str(args.state_file),
        "--report-file",
        str(args.report_file),
        "--master-port",
        str(args.master_port),
        "--c-lib-path",
        str(args.c_lib_path),
        "--num-threads",
        str(args.num_threads),
        "--max-async",
        str(args.max_async),
        "--seed",
        str(args.seed),
        "--log-interval",
        str(args.log_interval),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--dataset-dir",
        str(args.dataset_dir),
        "--dataset-split",
        str(args.dataset_split),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.seq_len),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--zero-stage",
        str(args.zero_stage),
        "--dtype",
        str(args.dtype),
    ]
    if args.model_label:
        cmd.extend(["--model-label", str(args.model_label)])
    if args.gradient_checkpointing:
        cmd.append("--gradient-checkpointing")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.attn_implementation:
        cmd.extend(["--attn-implementation", str(args.attn_implementation)])
    if args.keep_checkpoints:
        cmd.append("--keep-checkpoints")
    return cmd


def phase_all(args: argparse.Namespace) -> int:
    creator_cmd = build_subprocess_args(args, "creator")
    native.log("[all] starting creator process")
    rc = subprocess.call(creator_cmd)
    if rc != CREATOR_FAILURE_EXIT_CODE:
        return rc

    fault_start_time = time.perf_counter()
    connector_cmd = build_subprocess_args(args, "connector")
    connector_cmd.extend(["--fault-start-time", f"{fault_start_time:.9f}"])
    native.log("[all] starting connector process")
    return subprocess.call(connector_cmd)


def main() -> int:
    """Parse args and dispatch to the creator, connector, or both phases."""
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    args = normalize_run_paths(args)
    validate_args(args)
    if args.phase == "all":
        return phase_all(args)
    if args.phase == "creator":
        return run_creator(args)
    return run_connector(args)


if __name__ == "__main__":
    raise SystemExit(main())
