#!/usr/bin/env python3
"""Multi-GPU in-memory checkpoint benchmark: per-rank GPU state → shared-memory buffer → restore."""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import torch

import ds_hf_checkpoint_bench as native
from checkfreq_hf_checkpoint_bench import (
    move_optimizer_state_to_device,
    prepare_optimizer_state_for_load,
)
from flat_buffer_utils import (
    FLOAT32_SLOT_BYTES,
    RAW_TENSOR_ALIGNMENT_BYTES,
    align_up,
    decode_structure,
    encode_structure,
    resolve_flat_tensor_size,
    tensor_byte_view,
)

try:
    from multiprocessing import resource_tracker
except Exception:
    resource_tracker = None


SAVE_EPOCH = native.SAVE_EPOCH
RESUME_EPOCHS = native.RESUME_EPOCHS
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_NAME = "memory"
DEFAULT_CHECKPOINT_ROOT = SCRIPT_DIR / "memory_ckpt"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "report1" / DEFAULT_BASELINE_NAME



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one HF causal LM with multi-GPU DeepSpeed ZeRO-2/3, snapshot per-rank "
            "model/optimizer state into shared CPU memory buffers, relaunch a connector, "
            "and report in-memory checkpoint/recovery timing."
        )
    )
    parser.add_argument("--phase", choices=["creator", "connector", "all"], default="all")
    parser.add_argument("--model", required=True,
                        help="Local HF causal LM path or model identifier.")
    parser.add_argument("--model-label", default=None,
                        help="Short label used in report filenames and JSON output.")
    parser.add_argument(
        "--gpu-ids",
        default=os.environ.get("GPU_IDS", "0"),
        help="Comma-separated local physical GPU ids. One id -> single-GPU (world_size 1).",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_ROOT,
                        help="Directory used to write tiny manifests for in-memory checkpoints.")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_REPORT_DIR / "state.json",
                        help="Creator/connector shared state JSON.")
    parser.add_argument("--report-file", type=Path, default=DEFAULT_REPORT_DIR / "report.json",
                        help="Final JSON report path.")
    parser.add_argument("--master-port", type=int, default=29941)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--dataset-dir", type=Path, default=native.DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset-split", choices=("train", "validation", "test"),
                        default=native.DATASET_SPLIT)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-GPU micro batch size.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--zero-stage", type=int, default=3, choices=[3])
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"),
                        default="bfloat16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Keep the shared-memory segments after the final report is written.")
    parser.add_argument("--restart-start-time", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def current_rank() -> int:
    return int(os.environ.get("RANK", "0"))


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
    native.parse_gpu_ids(args.gpu_ids)
    native.validate_args(args)
    if args.zero_stage != 3:
        native.fail("In-memory checkpoint baseline currently supports only --zero-stage 3.")




def _untrack_shm(shm: shared_memory.SharedMemory) -> None:
    """Prevent Python's resource tracker from unlinking the SHM on process exit."""
    if resource_tracker is None:
        return
    name = str(getattr(shm, "_name", shm.name))
    with contextlib.suppress(Exception):
        resource_tracker.unregister(name, "shared_memory")


def _safe_close(shm: shared_memory.SharedMemory | None) -> None:
    if shm is None:
        return
    with contextlib.suppress(Exception):
        shm.close()


def _safe_unlink_and_close(shm: shared_memory.SharedMemory | None) -> None:
    if shm is None:
        return
    with contextlib.suppress(FileNotFoundError):
        shm.unlink()
    _safe_close(shm)


def _try_cleanup_by_name(name: str | None) -> None:
    if not name:
        return
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
    except (FileNotFoundError, OSError):
        return
    _safe_unlink_and_close(shm)


def _cleanup_stale_state(state_file: Path) -> None:
    if not state_file.exists():
        return
    try:
        state = native.load_json(state_file)
    except Exception:
        return
    for key in list(state.keys()):
        if key.startswith("shm_data_name_rank") or key.startswith("shm_meta_name_rank"):
            _try_cleanup_by_name(str(state.get(key) or ""))
    _try_cleanup_by_name(str(state.get("shm_data_name") or state.get("shared_buffer_name") or ""))
    _try_cleanup_by_name(str(state.get("shm_meta_name") or ""))




def _compute_flat_layout(
    flat_sources: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Return (flat_entries, total_bytes, total_f32_slots) for a list of tensor descriptors."""
    flat_entries: list[dict[str, Any]] = []
    running_bytes = 0
    for src in flat_sources:
        byte_offset = align_up(running_bytes, RAW_TENSOR_ALIGNMENT_BYTES)
        num_bytes = int(src["numel"]) * int(src["element_size"])
        flat_entries.append({
            "byte_offset": int(byte_offset),
            "num_bytes": int(num_bytes),
            "numel": int(src["numel"]),
            "shape": list(src["shape"]),
            "dtype": str(src["dtype"]),
        })
        running_bytes = byte_offset + num_bytes
    total_bytes = running_bytes
    total_slots = max(1, (total_bytes + FLOAT32_SLOT_BYTES - 1) // FLOAT32_SLOT_BYTES)
    return flat_entries, total_bytes, total_slots


def _ensure_shm(name_hint: str, size_bytes: int,
                existing: shared_memory.SharedMemory | None) -> shared_memory.SharedMemory:
    """Create or reuse a SharedMemory segment large enough for *size_bytes*."""
    if existing is not None:
        if existing.size >= size_bytes:
            return existing
        _safe_unlink_and_close(existing)
    _try_cleanup_by_name(name_hint)
    shm = shared_memory.SharedMemory(create=True, size=max(1, size_bytes))
    _untrack_shm(shm)
    torch.frombuffer(shm.buf, dtype=torch.uint8, count=shm.size).zero_()
    return shm




def _snapshot_model_state(engine: Any, zero_stage: int) -> dict[str, Any] | None:
    """ZeRO-3: skip the model snapshot entirely (fp16 params are reconstructed from the optimizer partition on recovery)."""
    return None


def _pack_tensors_to_shm(
    flat_sources: list[dict[str, Any]],
    flat_entries: list[dict[str, Any]],
    shm: shared_memory.SharedMemory,
    total_slots: int,
    cuda_device: torch.device | None,
) -> None:
    """Copy raw tensor bytes into the SHM-backed float32 tensor (CUDA tensors staged through pinned CPU via synchronous D2H)."""
    flat_tensor = torch.frombuffer(shm.buf, dtype=torch.float32, count=total_slots)
    flat_bytes = tensor_byte_view(flat_tensor)

    for src, meta in zip(flat_sources, flat_entries):
        t = src["tensor"].detach()
        if not t.is_contiguous():
            t = t.contiguous()
        start = int(meta["byte_offset"])
        end = start + int(meta["num_bytes"])
        if t.device.type == "cuda":
            cpu_tmp = torch.empty_like(t, device="cpu", pin_memory=True)
            cpu_tmp.copy_(t, non_blocking=False)
            flat_bytes[start:end].copy_(tensor_byte_view(cpu_tmp))
        else:
            flat_bytes[start:end].copy_(tensor_byte_view(t))

    if cuda_device is not None and cuda_device.type == "cuda":
        torch.cuda.synchronize(cuda_device)


def _serialize_meta_to_shm(meta_payload: dict[str, Any]) -> tuple[shared_memory.SharedMemory, int]:
    """Serialize *meta_payload* via torch.save into a SharedMemory segment."""
    buf = io.BytesIO()
    torch.save(meta_payload, buf)
    data = buf.getvalue()
    size = len(data)
    shm = shared_memory.SharedMemory(create=True, size=max(1, size))
    _untrack_shm(shm)
    shm.buf[:size] = data
    return shm, size


def _deserialize_meta_from_shm(shm: shared_memory.SharedMemory, size: int) -> dict[str, Any]:
    """Inverse of _serialize_meta_to_shm — load from an SHM segment."""
    raw = bytes(shm.buf[:size])
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)




class DistributedSharedMemoryCheckpointer:
    """Manages per-rank SHM segments: one pair (data + meta) per rank."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.latest_file = checkpoint_dir / "latest"
        self._shm_data: shared_memory.SharedMemory | None = None
        self._shm_meta: shared_memory.SharedMemory | None = None


    def save(
        self,
        torch_mod: Any,
        engine: Any,
        epoch: int,
        last_loss: float,
        model_label: str,
        zero_stage: int,
    ) -> dict[str, Any]:
        native.distributed_barrier(torch_mod)
        native.sync_device(torch_mod, engine.device)

        rank = current_rank()

        encode_start = time.perf_counter()
        model_state = _snapshot_model_state(engine, zero_stage)
        optimizer_state = engine.optimizer.state_dict()

        flat_sources: list[dict[str, Any]] = []
        encoded_model = encode_structure(model_state, flat_sources)
        encoded_optimizer = encode_structure(optimizer_state, flat_sources)

        flat_entries, total_bytes, total_slots = _compute_flat_layout(flat_sources)
        encode_seconds = time.perf_counter() - encode_start

        alloc_start = time.perf_counter()
        required_data_bytes = total_slots * FLOAT32_SLOT_BYTES
        shm_data_name = f"mem_ckpt_data_rank{rank:02d}"
        self._shm_data = _ensure_shm(shm_data_name, required_data_bytes, self._shm_data)
        alloc_seconds = time.perf_counter() - alloc_start

        pack_start = time.perf_counter()
        _pack_tensors_to_shm(flat_sources, flat_entries, self._shm_data, total_slots, engine.device)
        pack_seconds = time.perf_counter() - pack_start

        meta_start = time.perf_counter()
        meta_payload = {
            "model_state": encoded_model,
            "optimizer_state": encoded_optimizer,
            "flat_entries": flat_entries,
            "flat_total_bytes": int(total_bytes),
            "flat_total_size": int(total_slots),
            "flat_encoding": "raw_bytes_v2",
            "client_state": {
                "train_epoch": epoch,
                "last_loss": last_loss,
                "model_label": model_label,
            },
            "rank": rank,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "zero_stage": zero_stage,
        }
        _safe_unlink_and_close(self._shm_meta)
        self._shm_meta, meta_size = _serialize_meta_to_shm(meta_payload)
        meta_seconds = time.perf_counter() - meta_start

        tag = f"epoch_{epoch:06d}"
        manifest = {
            "tag": tag,
            "rank": rank,
            "shm_data_name": self._shm_data.name,
            "shm_data_size_bytes": int(required_data_bytes),
            "shm_meta_name": self._shm_meta.name,
            "shm_meta_size": int(meta_size),
            "flat_total_size": int(total_slots),
            "flat_total_bytes": int(total_bytes),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        rank_manifest_path = self.checkpoint_dir / f"{tag}_rank{rank:02d}.manifest.json"
        _write_text_fsync(
            rank_manifest_path,
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )

        native.distributed_barrier(torch_mod)
        if rank0():
            _write_text_fsync(self.latest_file, tag + "\n")
        native.distributed_barrier(torch_mod)

        del flat_sources, encoded_model, encoded_optimizer, meta_payload
        del model_state, optimizer_state
        gc.collect()

        snapshot_seconds = encode_seconds + pack_seconds + meta_seconds
        total_checkpoint_bytes = int(total_bytes) + int(meta_size)
        return {
            "epoch": int(epoch),
            "tag": tag,
            "rank": rank,
            "manifest_path": rank_manifest_path,
            "snapshot_seconds": float(snapshot_seconds),
            "persist_seconds": 0.0,
            "size_bytes": int(total_checkpoint_bytes),
            "apparent_size_bytes": int(total_checkpoint_bytes),
            "flat_data_bytes": int(total_bytes),
            "meta_bytes": int(meta_size),
            "encode_seconds": float(encode_seconds),
            "alloc_seconds": float(alloc_seconds),
            "shm_init_seconds": float(alloc_seconds),
            "pack_seconds": float(pack_seconds),
            "meta_serialize_seconds": float(meta_seconds),
            "shm_data_name": self._shm_data.name,
            "shm_data_size_bytes": int(required_data_bytes),
            "shm_meta_name": self._shm_meta.name,
            "shm_meta_size": int(meta_size),
        }


    def release(self, unlink: bool = True) -> None:
        if unlink:
            _safe_unlink_and_close(self._shm_data)
            _safe_unlink_and_close(self._shm_meta)
        else:
            _safe_close(self._shm_data)
            _safe_close(self._shm_meta)
        self._shm_data = None
        self._shm_meta = None




def _restore_from_shm(
    manifest: dict[str, Any],
    torch_mod: Any,
    engine: Any,
    zero_stage: int,
    device: torch.device,
) -> tuple[dict[str, Any], shared_memory.SharedMemory]:
    """Attach to per-rank SHM segments, decode, and load state into the engine."""
    rank = current_rank()

    shm_data = shared_memory.SharedMemory(name=str(manifest["shm_data_name"]), create=False)
    shm_meta = shared_memory.SharedMemory(name=str(manifest["shm_meta_name"]), create=False)
    meta_payload = _deserialize_meta_from_shm(shm_meta, int(manifest["shm_meta_size"]))

    flat_entries = list(meta_payload["flat_entries"])
    total_size = resolve_flat_tensor_size(meta_payload, flat_entries)
    flat_tensor = torch.frombuffer(shm_data.buf, dtype=torch.float32, count=total_size)
    flat_tensor_bytes = tensor_byte_view(flat_tensor)

    optimizer_state = decode_structure(
        meta_payload["optimizer_state"], flat_tensor, flat_entries, flat_tensor_bytes,
        copy=False,
    )

    engine.optimizer.load_state_dict(
        prepare_optimizer_state_for_load(engine.optimizer, optimizer_state, rank),
    )
    move_optimizer_state_to_device(engine.optimizer, device)

    client_state = dict(meta_payload.get("client_state", {}))

    del optimizer_state, meta_payload, flat_tensor, flat_tensor_bytes
    gc.collect()
    _safe_close(shm_meta)

    return client_state, shm_data




def _write_text_fsync(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _latest_rank_manifest(checkpoint_dir: Path, rank: int) -> dict[str, Any]:
    latest = checkpoint_dir / "latest"
    if not latest.exists():
        native.fail(f"Latest checkpoint marker not found in {checkpoint_dir}")
    tag = latest.read_text(encoding="utf-8").strip()
    if not tag:
        native.fail(f"Latest marker empty in {checkpoint_dir}")
    path = checkpoint_dir / f"{tag}_rank{rank:02d}.manifest.json"
    if not path.exists():
        native.fail(f"Rank manifest file does not exist: {path}")
    return native.load_json(path)


def _clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _destroy_engine(torch_mod: Any, engine: Any) -> None:
    del engine
    gc.collect()
    if torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()




def _report_metadata(args: argparse.Namespace, stats: dict[str, Any]) -> dict[str, Any]:
    payload = native.report_common_metadata(args, stats)
    payload["checkpoint_backend"] = f"memory_shared_cpu_flat_buffer_v2_zero{args.zero_stage}"
    payload["checkpoint_encoding"] = "flat_raw_bytes_v2"
    payload["checkpoint_memory_resident"] = True
    payload["checkpoint_payload_uri"] = "memory://shared-cpu-buffer-per-rank"
    return payload




def run_creator(args: argparse.Namespace) -> int:
    """Train one epoch, snapshot per-rank state into shared-memory buffers, and record the manifest/state."""
    checkpoint_dir = args.checkpoint_dir.resolve()
    state_file = args.state_file.resolve()

    if rank0():
        _cleanup_stale_state(state_file)
        _clear_directory(checkpoint_dir)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        if state_file.exists():
            state_file.unlink()

    torch_mod, engine, _, _, tokenizer, load_dataset_fn, stats = native.create_engine(args)
    blocks, data_stats = native.build_token_blocks(args, tokenizer, load_dataset_fn)
    available_step_count = native.batches_per_epoch(blocks, args.batch_size)
    epoch_step_count = native.resolve_epoch_steps(available_step_count, args.steps_per_epoch)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        native.fail("Tokenizer must provide either pad_token_id or eos_token_id.")
    model_label = native.model_label_from_arg(args.model, args.model_label)
    zero_stage = int(getattr(engine, "zero_optimization_stage", lambda: 0)())
    checkpointer = DistributedSharedMemoryCheckpointer(checkpoint_dir)

    native.log(
        f"[creator] model={model_label} "
        f"steps_per_epoch={epoch_step_count} "
        f"available_steps_per_epoch={available_step_count} "
        f"dataset_rows={data_stats['dataset_rows']} "
        f"token_blocks={data_stats['token_blocks']}"
    )

    try:
        epoch_seconds, last_loss = native.run_epoch(
            torch_mod, engine, blocks,
            args.batch_size, args.seq_len, int(pad_token_id),
            epoch_step_count, SAVE_EPOCH, args.log_interval,
        )

        ckpt_meta = checkpointer.save(
            torch_mod, engine, SAVE_EPOCH, last_loss, model_label, zero_stage,
        )

        snapshot_seconds = native.max_across_ranks(
            torch_mod, ckpt_meta["snapshot_seconds"], engine.device,
        )

        payload = _report_metadata(args, stats)
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
            "checkpoint_tag": str(ckpt_meta["tag"]),
            "epoch_train_seconds": float(epoch_seconds),
            "last_loss": float(last_loss),
            "checkpoint_save_seconds": float(snapshot_seconds),
            "checkpoint_snapshot_seconds": float(snapshot_seconds),
            "checkpoint_persist_seconds": 0.0,
            "checkpoint_encode_seconds": float(ckpt_meta["encode_seconds"]),
            "checkpoint_alloc_seconds": float(ckpt_meta["alloc_seconds"]),
            "checkpoint_shm_init_seconds": float(ckpt_meta["shm_init_seconds"]),
            "checkpoint_pack_seconds": float(ckpt_meta["pack_seconds"]),
            "checkpoint_meta_serialize_seconds": float(ckpt_meta["meta_serialize_seconds"]),
            "checkpoint_size_bytes": int(ckpt_meta["size_bytes"]),
            "checkpoint_size_gib": float(ckpt_meta["size_bytes"]) / (1024**3),
            "checkpoint_apparent_size_bytes": int(ckpt_meta["apparent_size_bytes"]),
            "checkpoint_apparent_size_gib": float(ckpt_meta["apparent_size_bytes"]) / (1024**3),
            f"shm_data_name_rank{current_rank():02d}": str(ckpt_meta["shm_data_name"]),
            f"shm_data_size_bytes_rank{current_rank():02d}": int(ckpt_meta["shm_data_size_bytes"]),
            f"shm_meta_name_rank{current_rank():02d}": str(ckpt_meta["shm_meta_name"]),
            f"shm_meta_size_rank{current_rank():02d}": int(ckpt_meta["shm_meta_size"]),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        if rank0():
            native.maybe_write_json(state_file, payload)
        native.log(
            f"[creator] model={payload['model_label']} "
            f"snapshot_seconds={payload['checkpoint_save_seconds']:.6f} "
            f"shm_data_size_gib={ckpt_meta['shm_data_size_bytes'] / (1024**3):.6f}"
        )
        checkpointer.release(unlink=False)
        return 0
    finally:
        _destroy_engine(torch_mod, engine)
        native.destroy_process_group(torch_mod)




def run_connector(args: argparse.Namespace) -> int:
    """Relaunch, attach to the shared-memory buffers, restore state, resume training, and report recovery timing."""
    connector_entry_time = time.perf_counter()
    restart_start_time = (
        args.restart_start_time if args.restart_start_time is not None else connector_entry_time
    )
    state = native.load_json(args.state_file.resolve())

    engine_init_start = time.perf_counter()
    torch_mod, engine, _, _, tokenizer, load_dataset_fn, _ = native.create_engine(args)
    checkpoint_dir = Path(state["checkpoint_dir"]).resolve()
    rank = current_rank()
    zero_stage = int(getattr(engine, "zero_optimization_stage", lambda: 0)())

    shm_data: shared_memory.SharedMemory | None = None

    try:
        engine_init_seconds = time.perf_counter() - engine_init_start
        engine_init_seconds = native.max_across_ranks(torch_mod, engine_init_seconds, engine.device)

        native.distributed_barrier(torch_mod)
        native.sync_device(torch_mod, engine.device)
        load_start = time.perf_counter()

        manifest = _latest_rank_manifest(checkpoint_dir, rank)
        client_state, shm_data = _restore_from_shm(
            manifest, torch_mod, engine, zero_stage, engine.device,
        )

        native.sync_device(torch_mod, engine.device)
        native.distributed_barrier(torch_mod)
        checkpoint_load_seconds = time.perf_counter() - load_start
        checkpoint_load_seconds = native.max_across_ranks(
            torch_mod, checkpoint_load_seconds, engine.device,
        )

        recovery_total_seconds = time.perf_counter() - restart_start_time
        recovery_total_seconds = native.max_across_ranks(
            torch_mod, recovery_total_seconds, engine.device,
        )
        connector_other_seconds = max(
            0.0, recovery_total_seconds - engine_init_seconds - checkpoint_load_seconds,
        )
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
                torch_mod, engine, blocks,
                args.batch_size, args.seq_len, int(pad_token_id),
                epoch_step_count, current_epoch, args.log_interval,
            )
    finally:
        if shm_data is not None:
            if not args.keep_checkpoints:
                _safe_unlink_and_close(shm_data)
                meta_name = manifest.get("shm_meta_name") if manifest else None
                _try_cleanup_by_name(str(meta_name or ""))
            else:
                _safe_close(shm_data)
        _destroy_engine(torch_mod, engine)
        native.destroy_process_group(torch_mod)

    payload = native.slim_recovery_report(state, {
        "recovery_total_seconds": float(recovery_total_seconds),
        "engine_init_seconds": float(engine_init_seconds),
        "load_seconds": float(checkpoint_load_seconds),
        "other_seconds": float(connector_other_seconds),
        "train_block_seconds": float(state.get("checkpoint_snapshot_seconds", 0.0)),
    })
    if rank0():
        native.maybe_write_json(args.report_file.resolve(), payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
        if not args.keep_checkpoints:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return 0




def build_phase_args(args: argparse.Namespace, phase: str) -> list[str]:
    script_path = Path(__file__).resolve()
    cmd = [
        str(script_path),
        "--phase", phase,
        "--model", str(args.model),
        "--model-label", str(args.model_label or ""),
        "--gpu-ids", str(args.gpu_ids),
        "--checkpoint-dir", str(args.checkpoint_dir),
        "--state-file", str(args.state_file),
        "--report-file", str(args.report_file),
        "--master-port", str(args.master_port),
        "--seed", str(args.seed),
        "--log-interval", str(args.log_interval),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--dataset-dir", str(args.dataset_dir),
        "--dataset-split", str(args.dataset_split),
        "--batch-size", str(args.batch_size),
        "--seq-len", str(args.seq_len),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--zero-stage", str(args.zero_stage),
        "--dtype", str(args.dtype),
    ]
    if not args.model_label:
        idx = cmd.index("--model-label")
        del cmd[idx:idx + 2]
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


def build_torchrun_command(args: argparse.Namespace, phase: str) -> tuple[list[str], dict[str, str]]:
    gpu_list = native.parse_gpu_ids(args.gpu_ids)
    phase_args = build_phase_args(args, phase)
    cmd = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone",
        "--nproc_per_node", str(len(gpu_list)),
        "--master_port", str(args.master_port),
        *phase_args,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
    return cmd, env


def build_direct_command(args: argparse.Namespace, phase: str) -> tuple[list[str], dict[str, str]]:
    gpu_list = native.parse_gpu_ids(args.gpu_ids)
    cmd = [sys.executable, *build_phase_args(args, phase)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
    return cmd, env


def phase_all(args: argparse.Namespace) -> int:
    mode = native.launch_mode(args)
    build = build_direct_command if mode == "single" else build_torchrun_command

    creator_cmd, creator_env = build(args, "creator")
    native.log(f"[all] starting creator process ({mode}) on gpus={args.gpu_ids}")
    rc = subprocess.call(creator_cmd, env=creator_env)
    if rc != 0:
        return rc

    restart_start_time = time.perf_counter()
    connector_cmd, connector_env = build(args, "connector")
    connector_cmd.extend(["--restart-start-time", f"{restart_start_time:.9f}"])
    native.log(f"[all] starting connector process ({mode}) on gpus={args.gpu_ids}")
    return subprocess.call(connector_cmd, env=connector_env)


def main() -> int:
    """Entry point: dispatch to the creator or connector phase based on parsed args."""
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
