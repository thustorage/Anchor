#!/usr/bin/env python3
"""In-memory checkpoint benchmark: GPU state to shared-memory buffer and back."""
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
    load_model_state,
    move_optimizer_state_to_device,
    prepare_optimizer_state_for_load,
)
from pccheck_hf_checkpoint_bench import (
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


CREATOR_FAILURE_EXIT_CODE = native.CREATOR_FAILURE_EXIT_CODE
FAULT_EPOCH = native.FAULT_EPOCH
RESUME_EPOCHS = native.RESUME_EPOCHS
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_NAME = "memory"
DEFAULT_CHECKPOINT_ROOT = SCRIPT_DIR / "ckpt" / DEFAULT_BASELINE_NAME
DEFAULT_REPORT_DIR = SCRIPT_DIR / "report1" / DEFAULT_BASELINE_NAME



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one HF causal LM with DeepSpeed, snapshot model/optimizer state "
            "into a shared CPU memory buffer at the failure boundary, relaunch a "
            "connector, and report in-memory checkpoint/recovery timing."
        )
    )
    parser.add_argument("--phase", choices=["creator", "connector", "all"], default="all")
    parser.add_argument("--model", required=True,
                        help="Local HF causal LM path or model identifier.")
    parser.add_argument("--model-label", default=None,
                        help="Short label used in report filenames and JSON output.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_ROOT,
                        help="Directory used to write the tiny manifest for the in-memory checkpoint.")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_REPORT_DIR / "state.json",
                        help="Creator/connector shared state JSON.")
    parser.add_argument("--report-file", type=Path, default=DEFAULT_REPORT_DIR / "report.json",
                        help="Final JSON report path.")
    parser.add_argument("--master-port", type=int, default=29671)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--dataset-dir", type=Path, default=native.DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset-split", choices=("train", "validation", "test"),
                        default=native.DATASET_SPLIT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--zero-stage", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"),
                        default="bfloat16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Keep the shared-memory segments after the final report is written.")
    parser.add_argument("--fault-start-time", type=float, default=None, help=argparse.SUPPRESS)
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
    return shm


def _pack_tensors_to_shm(
    flat_sources: list[dict[str, Any]],
    flat_entries: list[dict[str, Any]],
    shm: shared_memory.SharedMemory,
    total_slots: int,
    cuda_device: torch.device | None,
) -> None:
    """Copy raw tensor bytes into the SHM-backed float32 tensor."""
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
            cpu_tmp.copy_(t, non_blocking=True)
            src_bytes = tensor_byte_view(cpu_tmp)
            flat_bytes[start:end].copy_(src_bytes)
        else:
            src_bytes = tensor_byte_view(t)
            flat_bytes[start:end].copy_(src_bytes)

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




def _snapshot_model_state(engine: Any, zero_stage: int) -> dict[str, Any] | None:
    if zero_stage == 3:
        return None
    return engine.module.state_dict()




class SharedMemoryCheckpointer:
    """Manages the lifecycle of two SHM segments: one for tensor data, one for metadata."""

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
        model_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        native.distributed_barrier(torch_mod)
        native.sync_device(torch_mod, engine.device)

        encode_start = time.perf_counter()
        zero_stage = getattr(engine, "zero_optimization_stage", lambda: 0)()
        model_state = _snapshot_model_state(engine, zero_stage)
        optimizer_state = engine.optimizer.state_dict()

        flat_sources: list[dict[str, Any]] = []
        encoded_model = encode_structure(model_state, flat_sources)
        encoded_optimizer = encode_structure(optimizer_state, flat_sources)

        flat_entries, total_bytes, total_slots = _compute_flat_layout(flat_sources)
        encode_seconds = time.perf_counter() - encode_start

        alloc_start = time.perf_counter()
        required_data_bytes = total_slots * FLOAT32_SLOT_BYTES
        self._shm_data = _ensure_shm("mem_ckpt_data", required_data_bytes, self._shm_data)
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
                "model_label": model_metadata["model_label"],
            },
        }
        _safe_unlink_and_close(self._shm_meta)
        self._shm_meta, meta_size = _serialize_meta_to_shm(meta_payload)
        meta_seconds = time.perf_counter() - meta_start

        tag = f"epoch_{epoch:06d}"
        manifest = {
            "tag": tag,
            "shm_data_name": self._shm_data.name,
            "shm_data_size_bytes": int(required_data_bytes),
            "shm_meta_name": self._shm_meta.name,
            "shm_meta_size": int(meta_size),
            "flat_total_size": int(total_slots),
            "flat_total_bytes": int(total_bytes),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.checkpoint_dir / f"{tag}.manifest.json"
        _write_text_fsync(manifest_path,
                          json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        _write_text_fsync(self.latest_file, manifest_path.name + "\n")

        del flat_sources, encoded_model, encoded_optimizer, meta_payload
        del model_state, optimizer_state
        gc.collect()

        snapshot_seconds = encode_seconds + alloc_seconds + pack_seconds + meta_seconds
        return {
            "epoch": int(epoch),
            "tag": tag,
            "manifest_path": manifest_path,
            "snapshot_seconds": float(snapshot_seconds),
            "persist_seconds": 0.0,
            "size_bytes": int(total_bytes),
            "apparent_size_bytes": int(total_bytes),
            "encode_seconds": float(encode_seconds),
            "alloc_seconds": float(alloc_seconds),
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
    engine: Any,
    zero_stage: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, float], shared_memory.SharedMemory]:
    """Attach to SHM segments, decode, and load state into the engine."""
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    shm_data = shared_memory.SharedMemory(name=str(manifest["shm_data_name"]), create=False)
    timings["shm_data_attach_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    shm_meta = shared_memory.SharedMemory(name=str(manifest["shm_meta_name"]), create=False)
    meta_payload = _deserialize_meta_from_shm(shm_meta, int(manifest["shm_meta_size"]))
    timings["meta_deserialize_seconds"] = time.perf_counter() - t0

    flat_entries = list(meta_payload["flat_entries"])
    total_size = resolve_flat_tensor_size(meta_payload, flat_entries)
    flat_tensor = torch.frombuffer(shm_data.buf, dtype=torch.float32, count=total_size)
    flat_tensor_bytes = tensor_byte_view(flat_tensor)

    t0 = time.perf_counter()
    model_state = decode_structure(
        meta_payload["model_state"], flat_tensor, flat_entries, flat_tensor_bytes,
    )
    optimizer_state = decode_structure(
        meta_payload["optimizer_state"], flat_tensor, flat_entries, flat_tensor_bytes,
    )
    timings["decode_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    load_model_state(engine, model_state, zero_stage)
    engine.optimizer.load_state_dict(
        prepare_optimizer_state_for_load(engine.optimizer, optimizer_state),
    )
    move_optimizer_state_to_device(engine.optimizer, device)
    timings["state_restore_seconds"] = time.perf_counter() - t0

    client_state = dict(meta_payload.get("client_state", {}))

    del model_state, optimizer_state, meta_payload, flat_tensor, flat_tensor_bytes
    gc.collect()
    _safe_close(shm_meta)

    return client_state, timings, shm_data




def _write_text_fsync(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _latest_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    latest = checkpoint_dir / "latest"
    if not latest.exists():
        native.fail(f"Latest checkpoint marker not found in {checkpoint_dir}")
    name = latest.read_text(encoding="utf-8").strip()
    if not name:
        native.fail(f"Latest marker empty in {checkpoint_dir}")
    path = checkpoint_dir / name
    if not path.exists():
        native.fail(f"Manifest file does not exist: {path}")
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
    payload["checkpoint_backend"] = "memory_shared_cpu_flat_buffer_v2"
    payload["checkpoint_encoding"] = "flat_raw_bytes_v2"
    payload["checkpoint_memory_resident"] = True
    payload["checkpoint_payload_uri"] = "memory://shared-cpu-buffer"
    return payload




def run_creator(args: argparse.Namespace) -> int:
    """Run the initial training that snapshots state into shared memory and injects the fault."""
    checkpoint_dir = args.checkpoint_dir.resolve()
    state_file = args.state_file.resolve()
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
    checkpointer = SharedMemoryCheckpointer(checkpoint_dir)

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
            epoch_step_count, FAULT_EPOCH, args.log_interval,
        )

        ckpt_meta = checkpointer.save(
            torch_mod, engine, FAULT_EPOCH, last_loss,
            {"model_label": model_label},
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
            "checkpoint_file": str(ckpt_meta["manifest_path"].name),
            "epoch_train_seconds": float(epoch_seconds),
            "last_loss": float(last_loss),
            "checkpoint_save_seconds": float(ckpt_meta["snapshot_seconds"]),
            "checkpoint_snapshot_seconds": float(ckpt_meta["snapshot_seconds"]),
            "checkpoint_persist_seconds": float(ckpt_meta["persist_seconds"]),
            "checkpoint_encode_seconds": float(ckpt_meta["encode_seconds"]),
            "checkpoint_alloc_seconds": float(ckpt_meta["alloc_seconds"]),
            "checkpoint_pack_seconds": float(ckpt_meta["pack_seconds"]),
            "checkpoint_meta_serialize_seconds": float(ckpt_meta["meta_serialize_seconds"]),
            "checkpoint_size_bytes": int(ckpt_meta["size_bytes"]),
            "checkpoint_size_gib": float(ckpt_meta["size_bytes"]) / (1024**3),
            "checkpoint_apparent_size_bytes": int(ckpt_meta["apparent_size_bytes"]),
            "checkpoint_apparent_size_gib": float(ckpt_meta["apparent_size_bytes"]) / (1024**3),
            "shm_data_name": str(ckpt_meta["shm_data_name"]),
            "shm_data_size_bytes": int(ckpt_meta["shm_data_size_bytes"]),
            "shm_meta_name": str(ckpt_meta["shm_meta_name"]),
            "shm_meta_size": int(ckpt_meta["shm_meta_size"]),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        native.maybe_write_json(state_file, payload)
        native.log(
            f"[creator] model={payload['model_label']} "
            f"snapshot_seconds={payload['checkpoint_save_seconds']:.6f} "
            f"shm_data_size_gib={payload['shm_data_size_bytes'] / (1024**3):.6f}"
        )
        checkpointer.release(unlink=False)
        return CREATOR_FAILURE_EXIT_CODE
    finally:
        _destroy_engine(torch_mod, engine)
        native.destroy_process_group(torch_mod)




def run_connector(args: argparse.Namespace) -> int:
    """Recover by restoring model/optimizer state from shared memory and resuming training."""
    connector_entry_time = time.perf_counter()
    recovery_start_time = (
        args.fault_start_time if args.fault_start_time is not None else connector_entry_time
    )
    state = native.load_json(args.state_file.resolve())

    engine_init_start = time.perf_counter()
    torch_mod, engine, _, _, tokenizer, load_dataset_fn, _ = native.create_engine(args)
    checkpoint_dir = Path(state["checkpoint_dir"]).resolve()

    shm_data: shared_memory.SharedMemory | None = None

    try:
        engine_init_seconds = time.perf_counter() - engine_init_start
        engine_init_seconds = native.max_across_ranks(torch_mod, engine_init_seconds, engine.device)

        native.distributed_barrier(torch_mod)
        native.sync_device(torch_mod, engine.device)
        load_start = time.perf_counter()

        manifest = _latest_manifest(checkpoint_dir)
        zero_stage = getattr(engine, "zero_optimization_stage", lambda: 0)()
        client_state, timings, shm_data = _restore_from_shm(
            manifest, engine, zero_stage, engine.device,
        )

        native.sync_device(torch_mod, engine.device)
        native.distributed_barrier(torch_mod)
        checkpoint_load_seconds = time.perf_counter() - load_start
        checkpoint_load_seconds = native.max_across_ranks(
            torch_mod, checkpoint_load_seconds, engine.device,
        )

        recovery_total_seconds = time.perf_counter() - recovery_start_time
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
        for i in range(1, RESUME_EPOCHS + 1):
            resume_epoch_seconds, resume_last_loss = native.run_epoch(
                torch_mod, engine, blocks,
                args.batch_size, args.seq_len, int(pad_token_id),
                epoch_step_count, resumed_epoch + i, args.log_interval,
            )
    finally:
        if shm_data is not None:
            if not args.keep_checkpoints:
                _safe_unlink_and_close(shm_data)
                _try_cleanup_by_name(str(state.get("shm_meta_name") or ""))
            else:
                _safe_close(shm_data)
        _destroy_engine(torch_mod, engine)
        native.destroy_process_group(torch_mod)

    payload = dict(state)
    payload.update({
        "fault_to_recovery_ready_seconds": float(recovery_total_seconds),
        "connector_engine_init_seconds": float(engine_init_seconds),
        "connector_checkpoint_load_seconds": float(checkpoint_load_seconds),
        "connector_shm_data_attach_seconds": float(timings.get("shm_data_attach_seconds", 0)),
        "connector_meta_deserialize_seconds": float(timings.get("meta_deserialize_seconds", 0)),
        "connector_decode_seconds": float(timings.get("decode_seconds", 0)),
        "connector_state_restore_seconds": float(timings.get("state_restore_seconds", 0)),
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
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--phase", phase,
        "--model", str(args.model),
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
    """Entry point dispatching to the creator, connector, or combined phase."""
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
