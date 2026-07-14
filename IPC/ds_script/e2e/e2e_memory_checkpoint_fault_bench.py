#!/usr/bin/env python3
"""End-to-end in-memory (SHM) checkpoint benchmark with random fault injection (single + multi GPU)."""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
LAB2_DIR = WORKSPACE_ROOT / "IPC" / "ds_script" / "fault"
DEFAULT_CHECKPOINT_ROOT = SCRIPT_DIR / "ckpt" / "e2e_memory"
DEFAULT_REPORT_ROOT = SCRIPT_DIR / "report1" / "e2e_memory"

try:
    from multiprocessing import resource_tracker
except Exception:
    resource_tracker = None


def _neutralize_shm_resource_tracker() -> None:
    """Stop Python's resource_tracker from auto-managing shared_memory (we own SHM lifetime explicitly)."""
    if resource_tracker is None:
        return
    rt = resource_tracker
    _orig_register = rt.register
    _orig_unregister = rt.unregister

    def register(name, rtype):
        if rtype == "shared_memory":
            return
        return _orig_register(name, rtype)

    def unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return _orig_unregister(name, rtype)

    rt.register = register
    rt.unregister = unregister
    with contextlib.suppress(Exception):
        rt._CLEANUP_FUNCS.pop("shared_memory", None)


_neutralize_shm_resource_tracker()




def add_lab2_path() -> None:
    if str(LAB2_DIR) not in sys.path:
        sys.path.insert(0, str(LAB2_DIR))


def load_bench_modules() -> tuple[Any, Any, Any]:
    """Return (native, checkfreq, flat-buffer utils) from fault/."""
    add_lab2_path()
    import ds_hf_checkpoint_bench as native
    import checkfreq_hf_checkpoint_bench as checkfreq
    import flat_buffer_utils as pccheck

    return native, checkfreq, pccheck


def load_native_module() -> Any:
    add_lab2_path()
    import ds_hf_checkpoint_bench as native

    return native




def parse_args() -> argparse.Namespace:
    native = load_native_module()
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end multi-GPU in-memory checkpoint benchmark with a manager "
            "process that injects random fail-stop crashes and restarts a fresh "
            "worker group via torchrun.  State is kept in per-rank POSIX SHM."
        )
    )
    parser.add_argument("--phase", choices=("manager", "worker"), default="manager")
    parser.add_argument("--model", required=True, help="Local HF causal LM path or model identifier.")
    parser.add_argument("--model-label", default=None, help="Short label used in filenames and reports.")
    parser.add_argument(
        "--gpu-ids",
        default=os.environ.get("GPU_IDS", "2,3"),
        help="Comma-separated local physical GPU ids, for example 2,3 or 0,1,2,3.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Directory used to keep manifest JSONs for the SHM checkpoint.",
    )
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--report-file", type=Path, default=None)
    parser.add_argument("--fault-file", type=Path, default=None)
    parser.add_argument("--master-port", type=int, default=30161)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-interval", type=int, default=0)
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
    parser.add_argument("--zero-stage", type=int, default=3, choices=[2, 3])
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
        "--total-steps",
        type=int,
        default=10000,
        help="Target committed global steps for the whole e2e run.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Snapshot to shared-memory every N completed steps.",
    )
    parser.add_argument(
        "--fault-frequency",
        type=float,
        default=10.0,
        help="Every fault check, inject a crash when a random draw in [0, 100) is below this value.",
    )
    parser.add_argument(
        "--fault-check-interval-seconds",
        type=float,
        default=1.0,
        help="How often the manager should draw a fault-injection random number.",
    )
    parser.add_argument(
        "--max-unexpected-worker-exits",
        type=int,
        default=5,
        help="Abort after this many non-injected worker failures.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep the SHM segments and manifest directory after the benchmark completes.",
    )
    parser.add_argument("--parent-start-wall-time", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-launch-index", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def current_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def normalize_run_paths(args: argparse.Namespace, native: Any) -> argparse.Namespace:
    label = native.model_label_from_arg(args.model, args.model_label)
    if args.checkpoint_dir == DEFAULT_CHECKPOINT_ROOT:
        args.checkpoint_dir = DEFAULT_CHECKPOINT_ROOT / label
    if args.state_file is None:
        args.state_file = DEFAULT_REPORT_ROOT / f"{label}_state.json"
    if args.report_file is None:
        args.report_file = DEFAULT_REPORT_ROOT / f"{label}.json"
    if args.fault_file is None:
        args.fault_file = DEFAULT_REPORT_ROOT / f"{label}_faults.jsonl"
    return args


def validate_args(args: argparse.Namespace, native: Any) -> None:
    if not hasattr(args, "steps_per_epoch"):
        args.steps_per_epoch = int(args.total_steps)
    native.validate_args(args)
    if args.total_steps <= 0:
        native.fail("--total-steps must be positive.")
    if args.checkpoint_interval <= 0:
        native.fail("--checkpoint-interval must be positive.")
    if args.fault_frequency < 0.0 or args.fault_frequency > 100.0:
        native.fail("--fault-frequency must be within [0, 100].")
    if args.fault_check_interval_seconds <= 0.0:
        native.fail("--fault-check-interval-seconds must be positive.")
    if args.max_unexpected_worker_exits < 0:
        native.fail("--max-unexpected-worker-exits must be non-negative.")


def derived_paths(args: argparse.Namespace) -> dict[str, Path]:
    checkpoint_dir = args.checkpoint_dir.resolve()
    state_file = args.state_file.resolve()
    report_file = args.report_file.resolve()
    fault_file = args.fault_file.resolve()
    return {
        "checkpoint_dir": checkpoint_dir,
        "state_file": state_file,
        "report_file": report_file,
        "fault_file": fault_file,
        "runtime_file": state_file.with_name(state_file.stem + "_runtime.json"),
    }




def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl_record(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()




def _untrack_shm(shm: shared_memory.SharedMemory) -> None:
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




def _compute_flat_layout(
    flat_sources: list[dict[str, Any]],
    pccheck: Any,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return (flat_entries, total_bytes, total_f32_slots)."""
    flat_entries: list[dict[str, Any]] = []
    running_bytes = 0
    for src in flat_sources:
        byte_offset = pccheck.align_up(running_bytes, pccheck.RAW_TENSOR_ALIGNMENT_BYTES)
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
    total_slots = max(1, (total_bytes + pccheck.FLOAT32_SLOT_BYTES - 1) // pccheck.FLOAT32_SLOT_BYTES)
    return flat_entries, total_bytes, total_slots


def _ensure_shm(
    name_hint: str,
    size_bytes: int,
    existing: shared_memory.SharedMemory | None,
) -> shared_memory.SharedMemory:
    if existing is not None:
        if existing.size >= size_bytes:
            return existing
        _safe_unlink_and_close(existing)
    _try_cleanup_by_name(name_hint)
    shm = shared_memory.SharedMemory(name=name_hint, create=True, size=max(1, size_bytes))
    _untrack_shm(shm)
    return shm


def _pack_tensors_to_shm(
    flat_sources: list[dict[str, Any]],
    flat_entries: list[dict[str, Any]],
    shm: shared_memory.SharedMemory,
    total_slots: int,
    cuda_device: Any,
    torch_mod: Any,
    pccheck: Any,
) -> None:
    flat_tensor = torch_mod.frombuffer(shm.buf, dtype=torch_mod.float32, count=total_slots)
    flat_bytes = pccheck.tensor_byte_view(flat_tensor)

    for src, meta in zip(flat_sources, flat_entries):
        t = src["tensor"].detach()
        if not t.is_contiguous():
            t = t.contiguous()
        start = int(meta["byte_offset"])
        end = start + int(meta["num_bytes"])
        if t.device.type == "cuda":
            cpu_tmp = torch_mod.empty_like(t, device="cpu", pin_memory=True)
            cpu_tmp.copy_(t, non_blocking=False)
            src_bytes = pccheck.tensor_byte_view(cpu_tmp)
            flat_bytes[start:end].copy_(src_bytes)
        else:
            src_bytes = pccheck.tensor_byte_view(t)
            flat_bytes[start:end].copy_(src_bytes)

    if cuda_device is not None and hasattr(cuda_device, "type") and cuda_device.type == "cuda":
        torch_mod.cuda.synchronize(cuda_device)


def _serialize_meta_to_shm(
    meta_payload: dict[str, Any], torch_mod: Any, name_hint: str
) -> tuple[shared_memory.SharedMemory, int]:
    buf = io.BytesIO()
    torch_mod.save(meta_payload, buf)
    data = buf.getvalue()
    size = len(data)
    _try_cleanup_by_name(name_hint)
    shm = shared_memory.SharedMemory(name=name_hint, create=True, size=max(1, size))
    _untrack_shm(shm)
    shm.buf[:size] = data
    return shm, size


def _deserialize_meta_from_shm(shm: shared_memory.SharedMemory, size: int, torch_mod: Any) -> dict[str, Any]:
    raw = bytes(shm.buf[:size])
    return torch_mod.load(io.BytesIO(raw), map_location="cpu", weights_only=False)


def _write_text_fsync(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)




def snapshot_model_state(engine: Any, zero_stage: int, checkfreq: Any) -> dict[str, Any] | None:
    """Snapshot the model state dict (returns None under ZeRO-3 — optimizer partition carries the fp32 master)."""
    if zero_stage != 3:
        return engine.module.state_dict()
    return None




def record_runtime_state(
    runtime_file: Path,
    step: int,
    latest_checkpoint_step: int,
    latest_checkpoint_tag: str | None,
    last_loss: float,
    parent_start_wall_time: float | None,
) -> None:
    now = time.time()
    payload = {
        "last_completed_step": int(step),
        "last_completed_wall_time": float(now),
        "last_completed_relative_time_seconds": (
            None if parent_start_wall_time is None else float(now - parent_start_wall_time)
        ),
        "latest_checkpoint_step": int(latest_checkpoint_step),
        "latest_checkpoint_tag": latest_checkpoint_tag,
        "last_loss": float(last_loss),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(runtime_file, payload)




def save_checkpoint_to_shm(
    torch_mod: Any,
    engine: Any,
    current_step: int,
    last_loss: float,
    model_label: str,
    paths: dict[str, Path],
    pccheck: Any,
    checkfreq: Any,
    *,
    shm_data: shared_memory.SharedMemory | None,
    shm_meta: shared_memory.SharedMemory | None,
) -> tuple[dict[str, Any], shared_memory.SharedMemory, shared_memory.SharedMemory]:
    """Encode and pack per-rank model+optimizer into SHM.  Returns (meta, shm_data, shm_meta)."""
    rank = current_rank()
    zero_stage = getattr(engine, "zero_optimization_stage", lambda: 0)()

    encode_start = time.perf_counter()
    model_state = snapshot_model_state(engine, zero_stage, checkfreq)
    optimizer_state = engine.optimizer.state_dict()

    flat_sources: list[dict[str, Any]] = []
    encoded_model = pccheck.encode_structure(model_state, flat_sources)
    encoded_optimizer = pccheck.encode_structure(optimizer_state, flat_sources)

    flat_entries, total_bytes, total_slots = _compute_flat_layout(flat_sources, pccheck)
    encode_seconds = time.perf_counter() - encode_start

    alloc_start = time.perf_counter()
    required_data_bytes = total_slots * pccheck.FLOAT32_SLOT_BYTES
    shm_data_name = f"e2e_mem_mg_data_rank{rank:02d}"
    shm_data = _ensure_shm(shm_data_name, required_data_bytes, shm_data)
    alloc_seconds = time.perf_counter() - alloc_start

    pack_start = time.perf_counter()
    _pack_tensors_to_shm(flat_sources, flat_entries, shm_data, total_slots, engine.device, torch_mod, pccheck)
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
            "global_step": int(current_step),
            "last_loss": float(last_loss),
            "model_label": model_label,
        },
        "rank": rank,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "zero_stage": zero_stage,
    }
    _safe_unlink_and_close(shm_meta)
    shm_meta_name = f"e2e_mem_mg_meta_rank{rank:02d}"
    shm_meta, meta_size = _serialize_meta_to_shm(meta_payload, torch_mod, shm_meta_name)
    meta_seconds = time.perf_counter() - meta_start

    tag = f"step_{current_step:08d}"
    manifest = {
        "tag": tag,
        "rank": rank,
        "shm_data_name": shm_data.name,
        "shm_data_size_bytes": int(required_data_bytes),
        "shm_meta_name": shm_meta.name,
        "shm_meta_size": int(meta_size),
        "flat_total_size": int(total_slots),
        "flat_total_bytes": int(total_bytes),
        "checkpoint_snapshot_seconds": float(encode_seconds + alloc_seconds + pack_seconds + meta_seconds),
        "checkpoint_size_bytes": int(total_bytes),
        "checkpoint_apparent_size_bytes": int(total_bytes),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    rank_manifest_path = paths["checkpoint_dir"] / f"{tag}_rank{rank:02d}.manifest.json"
    _write_text_fsync(
        rank_manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )

    native_mod = load_native_module()
    native_mod.distributed_barrier(torch_mod)
    if rank0():
        latest_file = paths["checkpoint_dir"] / "latest"
        _write_text_fsync(latest_file, tag + "\n")
        atomic_write_json(
            paths["checkpoint_dir"] / "resume_manifest.json",
            {
                "latest_checkpoint_tag": str(tag),
                "latest_checkpoint_step": int(current_step),
                "latest_last_loss": float(last_loss),
                "checkpoint_snapshot_seconds": float(encode_seconds + alloc_seconds + pack_seconds + meta_seconds),
                "checkpoint_size_bytes": int(total_bytes),
                "checkpoint_apparent_size_bytes": int(total_bytes),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    native_mod.distributed_barrier(torch_mod)

    del flat_sources, encoded_model, encoded_optimizer, meta_payload
    del model_state, optimizer_state
    gc.collect()

    snapshot_seconds = encode_seconds + alloc_seconds + pack_seconds + meta_seconds
    result_meta = {
        "tag": tag,
        "snapshot_seconds": float(snapshot_seconds),
        "size_bytes": int(total_bytes),
        "encode_seconds": float(encode_seconds),
        "alloc_seconds": float(alloc_seconds),
        "pack_seconds": float(pack_seconds),
        "meta_serialize_seconds": float(meta_seconds),
    }
    return result_meta, shm_data, shm_meta


def _latest_rank_manifest(checkpoint_dir: Path, rank: int) -> dict[str, Any] | None:
    """Read the per-rank manifest for the latest checkpoint tag."""
    latest = checkpoint_dir / "latest"
    if not latest.exists():
        return None
    tag = latest.read_text(encoding="utf-8").strip()
    if not tag:
        return None
    path = checkpoint_dir / f"{tag}_rank{rank:02d}.manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def restore_from_shm(
    manifest: dict[str, Any],
    torch_mod: Any,
    engine: Any,
    pccheck: Any,
    checkfreq: Any,
) -> tuple[dict[str, Any], dict[str, float], shared_memory.SharedMemory]:
    """Attach to per-rank SHM segments, decode, load into engine."""
    timings: dict[str, float] = {}
    rank = current_rank()
    zero_stage = getattr(engine, "zero_optimization_stage", lambda: 0)()

    t0 = time.perf_counter()
    shm_data = shared_memory.SharedMemory(name=str(manifest["shm_data_name"]), create=False)
    timings["shm_data_attach_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    shm_meta = shared_memory.SharedMemory(name=str(manifest["shm_meta_name"]), create=False)
    meta_payload = _deserialize_meta_from_shm(shm_meta, int(manifest["shm_meta_size"]), torch_mod)
    timings["meta_deserialize_seconds"] = time.perf_counter() - t0

    flat_entries = list(meta_payload["flat_entries"])
    total_size = pccheck.resolve_flat_tensor_size(meta_payload, flat_entries)
    flat_tensor = torch_mod.frombuffer(shm_data.buf, dtype=torch_mod.float32, count=total_size)
    flat_tensor_bytes = pccheck.tensor_byte_view(flat_tensor)

    t0 = time.perf_counter()
    optimizer_state = pccheck.decode_structure(
        meta_payload["optimizer_state"], flat_tensor, flat_entries, flat_tensor_bytes,
        copy=False,
    )
    timings["decode_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    engine.optimizer.load_state_dict(
        checkfreq.prepare_optimizer_state_for_load(engine.optimizer, optimizer_state, rank),
    )
    checkfreq.move_optimizer_state_to_device(engine.optimizer, engine.device)
    timings["state_restore_seconds"] = time.perf_counter() - t0

    client_state = dict(meta_payload.get("client_state", {}))

    del optimizer_state, meta_payload, flat_tensor, flat_tensor_bytes
    gc.collect()
    _safe_close(shm_meta)

    return client_state, timings, shm_data




def run_worker(args: argparse.Namespace, native: Any, checkfreq: Any, pccheck: Any) -> int:
    """Worker phase: train and snapshot per-rank optimizer state into SHM segments that survive kill -9."""
    paths = derived_paths(args)
    paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    torch_mod = None
    engine = None
    shm_data: shared_memory.SharedMemory | None = None
    shm_meta: shared_memory.SharedMemory | None = None

    try:
        resume_manifest = load_json_if_exists(paths["checkpoint_dir"] / "resume_manifest.json") or {}

        torch_mod, engine, _, _, tokenizer, load_dataset_fn, _ = native.create_engine(args)
        blocks, _ = native.build_token_blocks(args, tokenizer, load_dataset_fn)
        available_step_count = native.batches_per_epoch(blocks, args.batch_size)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_token_id is None:
            native.fail("Tokenizer must provide either pad_token_id or eos_token_id.")

        rank = current_rank()
        rank_manifest = _latest_rank_manifest(paths["checkpoint_dir"], rank)
        latest_tag = resume_manifest.get("latest_checkpoint_tag")
        latest_checkpoint_step = int(resume_manifest.get("latest_checkpoint_step", 0) or 0)
        start_step = 0
        last_loss = float(resume_manifest.get("latest_last_loss", 0.0) or 0.0)
        checkpoint_count = 0

        if rank_manifest is not None and latest_tag and rank_manifest.get("shm_data_name"):
            try:
                native.distributed_barrier(torch_mod)
                native.sync_device(torch_mod, engine.device)
                client_state, _timings, shm_data = restore_from_shm(
                    rank_manifest, torch_mod, engine, pccheck, checkfreq,
                )
                native.sync_device(torch_mod, engine.device)
                native.distributed_barrier(torch_mod)

                start_step = int(client_state.get("global_step", latest_checkpoint_step) or latest_checkpoint_step)
                latest_checkpoint_step = start_step
                last_loss = float(client_state.get("last_loss", last_loss) or last_loss)
                checkpoint_count = max(1, start_step // args.checkpoint_interval)
                native.log(
                    f"[worker] launch={args.worker_launch_index} rank={rank} resumed from SHM "
                    f"tag={latest_tag} step={start_step}/{args.total_steps}"
                )
            except (FileNotFoundError, OSError) as exc:
                native.log(
                    f"[worker] launch={args.worker_launch_index} rank={rank} SHM attach failed ({exc}), "
                    f"starting from scratch"
                )
                shm_data = None
                start_step = 0
                latest_checkpoint_step = 0
                latest_tag = None
        else:
            native.log(f"[worker] launch={args.worker_launch_index} started step=0/{args.total_steps}")

        if rank0():
            record_runtime_state(
                paths["runtime_file"],
                start_step,
                latest_checkpoint_step,
                latest_tag,
                last_loss,
                args.parent_start_wall_time,
            )

        model_label = native.model_label_from_arg(args.model, args.model_label)
        current_completed_step = start_step

        for step_idx in range(start_step, args.total_steps):
            batch_index = step_idx % available_step_count
            batch = native.build_batch_from_blocks(
                torch_mod,
                blocks,
                batch_index,
                args.batch_size,
                args.seq_len,
                int(pad_token_id),
                engine.device,
            )
            last_loss = native.run_one_step(engine, batch)
            current_completed_step = step_idx + 1

            if rank0():
                record_runtime_state(
                    paths["runtime_file"],
                    current_completed_step,
                    latest_checkpoint_step,
                    latest_tag,
                    last_loss,
                    args.parent_start_wall_time,
                )

            if args.log_interval > 0 and (
                current_completed_step % args.log_interval == 0 or current_completed_step == args.total_steps
            ):
                native.log(
                    f"[worker] launch={args.worker_launch_index} step={current_completed_step}/{args.total_steps} "
                    f"loss={last_loss:.6f}"
                )

            if current_completed_step % args.checkpoint_interval == 0:
                ckpt_meta, shm_data, shm_meta = save_checkpoint_to_shm(
                    torch_mod,
                    engine,
                    current_completed_step,
                    last_loss,
                    model_label,
                    paths,
                    pccheck,
                    checkfreq,
                    shm_data=shm_data,
                    shm_meta=shm_meta,
                )
                latest_tag = ckpt_meta["tag"]
                latest_checkpoint_step = current_completed_step
                checkpoint_count += 1

                if rank0():
                    record_runtime_state(
                        paths["runtime_file"],
                        current_completed_step,
                        latest_checkpoint_step,
                        latest_tag,
                        last_loss,
                        args.parent_start_wall_time,
                    )
                native.log(
                    f"[worker] SHM snapshot tag={latest_tag} step={current_completed_step} "
                    f"snapshot_seconds={ckpt_meta['snapshot_seconds']:.6f}"
                )

        if rank0():
            record_runtime_state(
                paths["runtime_file"],
                int(args.total_steps),
                latest_checkpoint_step,
                latest_tag,
                last_loss,
                args.parent_start_wall_time,
            )
        return 0
    finally:
        if shm_data is not None:
            _safe_close(shm_data)
        if shm_meta is not None:
            _safe_close(shm_meta)
        if engine is not None and torch_mod is not None:
            checkfreq.destroy_engine(torch_mod, engine)
            native.destroy_process_group(torch_mod)




def build_worker_cmd(
    args: argparse.Namespace,
    native: Any,
    worker_launch_index: int,
    parent_start_wall_time: float,
) -> tuple[list[str], dict[str, str]]:
    script_path = Path(__file__).resolve()
    gpu_list = native.parse_gpu_ids(args.gpu_ids)
    launch_port = int(args.master_port + (worker_launch_index % 1000))
    phase_args = [
        str(script_path),
        "--phase", "worker",
        "--model", str(args.model),
        "--gpu-ids", str(args.gpu_ids),
        "--checkpoint-dir", str(args.checkpoint_dir),
        "--state-file", str(args.state_file),
        "--report-file", str(args.report_file),
        "--fault-file", str(args.fault_file),
        "--master-port", str(launch_port),
        "--seed", str(args.seed),
        "--log-interval", str(args.log_interval),
        "--dataset-dir", str(args.dataset_dir),
        "--dataset-split", str(args.dataset_split),
        "--batch-size", str(args.batch_size),
        "--seq-len", str(args.seq_len),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--zero-stage", str(args.zero_stage),
        "--dtype", str(args.dtype),
        "--total-steps", str(args.total_steps),
        "--checkpoint-interval", str(args.checkpoint_interval),
        "--fault-frequency", str(args.fault_frequency),
        "--fault-check-interval-seconds", str(args.fault_check_interval_seconds),
        "--max-unexpected-worker-exits", str(args.max_unexpected_worker_exits),
        "--parent-start-wall-time", f"{parent_start_wall_time:.9f}",
        "--worker-launch-index", str(worker_launch_index),
    ]
    if args.model_label:
        phase_args.extend(["--model-label", str(args.model_label)])
    if args.gradient_checkpointing:
        phase_args.append("--gradient-checkpointing")
    if args.trust_remote_code:
        phase_args.append("--trust-remote-code")
    if args.local_files_only:
        phase_args.append("--local-files-only")
    if args.attn_implementation:
        phase_args.extend(["--attn-implementation", str(args.attn_implementation)])
    if args.keep_checkpoints:
        phase_args.append("--keep-checkpoints")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
    env.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "0")
    env.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    if native.launch_mode(args) == "single":
        cmd = [sys.executable, *phase_args]
    else:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone",
            "--nproc_per_node", str(len(gpu_list)),
            "--master_port", str(launch_port),
            *phase_args,
        ]
    return cmd, env


def kill_process_group(process: subprocess.Popen[str]) -> None:
    """Hard-kill the entire worker process tree (torchrun agent + every rank)."""
    import psutil

    try:
        parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return
    try:
        victims = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        victims = []
    victims = [parent, *victims]
    for proc in victims:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(victims, timeout=30)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def _cleanup_stale_shm(checkpoint_dir: Path, world_size: int) -> None:
    """Clean up SHM segments from a previous run by reading manifests."""
    latest = checkpoint_dir / "latest"
    if not latest.exists():
        return
    tag = latest.read_text(encoding="utf-8").strip()
    if not tag:
        return
    for r in range(world_size):
        manifest_path = checkpoint_dir / f"{tag}_rank{r:02d}.manifest.json"
        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text(encoding="utf-8"))
                _try_cleanup_by_name(str(m.get("shm_data_name") or ""))
                _try_cleanup_by_name(str(m.get("shm_meta_name") or ""))
            except Exception:
                pass


def initialize_manager_files(paths: dict[str, Path], world_size: int) -> None:
    native = load_native_module()
    _cleanup_stale_shm(paths["checkpoint_dir"], world_size)
    native.clear_directory(paths["checkpoint_dir"])
    for path in (
        paths["state_file"],
        paths["report_file"],
        paths["fault_file"],
        paths["runtime_file"],
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()


def run_manager(args: argparse.Namespace, native: Any) -> int:
    """Manager phase: inject random fail-stop crashes and relaunch the worker until the run completes."""
    paths = derived_paths(args)
    gpu_list = native.parse_gpu_ids(args.gpu_ids)
    world_size = len(gpu_list)
    initialize_manager_files(paths, world_size)

    rng = random.Random(args.seed)
    parent_start_wall_time = time.time()
    fault_injection_count = 0
    worker_launch_count = 0
    worker_restart_count = 0
    unexpected_worker_exit_count = 0
    fault_check_draw_count = 0

    with paths["fault_file"].open("a", encoding="utf-8") as fault_handle:
        append_jsonl_record(
            fault_handle,
            {
                "event": "manager_start",
                "wall_time": float(parent_start_wall_time),
                "fault_frequency": float(args.fault_frequency),
                "fault_check_interval_seconds": float(args.fault_check_interval_seconds),
                "total_steps": int(args.total_steps),
                "checkpoint_interval": int(args.checkpoint_interval),
                "gpu_ids": str(args.gpu_ids),
            },
        )

        def launch_worker() -> subprocess.Popen[str]:
            nonlocal worker_launch_count
            cmd, env = build_worker_cmd(args, native, worker_launch_count, parent_start_wall_time)
            native.log(
                f"[manager] launching worker index={worker_launch_count} "
                f"master_port={args.master_port + (worker_launch_count % 1000)} "
                f"gpu_ids={args.gpu_ids}"
            )
            process = subprocess.Popen(cmd, env=env, start_new_session=True)
            worker_launch_count += 1
            return process

        process = launch_worker()

        while True:
            rc = process.poll()
            if rc is not None:
                if rc == 0:
                    break
                unexpected_worker_exit_count += 1
                append_jsonl_record(
                    fault_handle,
                    {
                        "event": "worker_exit",
                        "return_code": int(rc),
                        "wall_time": float(time.time()),
                        "relative_time_seconds": float(time.time() - parent_start_wall_time),
                        "unexpected_worker_exit_count": int(unexpected_worker_exit_count),
                    },
                )
                if unexpected_worker_exit_count > args.max_unexpected_worker_exits:
                    native.fail(
                        "Worker exited unexpectedly too many times. "
                        f"limit={args.max_unexpected_worker_exits}"
                    )
                worker_restart_count += 1
                process = launch_worker()
                continue

            time.sleep(args.fault_check_interval_seconds)
            fault_check_draw_count += 1
            draw = rng.uniform(0.0, 100.0)
            if draw >= args.fault_frequency:
                continue

            runtime_state = load_json_if_exists(paths["runtime_file"]) or {}
            observed_step = int(runtime_state.get("last_completed_step", 0) or 0)
            fault_injection_count += 1
            append_jsonl_record(
                fault_handle,
                {
                    "event": "fault_injected",
                    "wall_time": float(time.time()),
                    "relative_time_seconds": float(time.time() - parent_start_wall_time),
                    "random_draw": float(draw),
                    "threshold": float(args.fault_frequency),
                    "observed_step_before_kill": int(observed_step),
                    "worker_launch_index": int(worker_launch_count - 1),
                    "pid": int(process.pid),
                },
            )
            native.log(
                f"[manager] fault injected draw={draw:.4f} threshold={args.fault_frequency:.4f} "
                f"observed_step={observed_step}"
            )
            kill_process_group(process)
            worker_restart_count += 1
            process = launch_worker()

    total_elapsed_seconds = time.time() - parent_start_wall_time
    runtime_state = load_json_if_exists(paths["runtime_file"]) or {}
    completed_steps = int(runtime_state.get("last_completed_step", 0) or 0)
    average_steps_per_second = (
        float(completed_steps / total_elapsed_seconds) if total_elapsed_seconds > 0.0 else 0.0
    )

    mode_suffix = "" if native.launch_mode(args) == "single" else "_multigpu"
    payload = {
        "experiment": f"e2e_memory_checkpoint_faults{mode_suffix}",
        "checkpoint_backend": f"memory_shared_cpu_flat_buffer_e2e{mode_suffix}",
        "model": args.model,
        "model_label": native.model_label_from_arg(args.model, args.model_label),
        "dtype": args.dtype,
        "recomputation": bool(args.gradient_checkpointing),
        "zero_stage": int(args.zero_stage),
        "gpu_ids": str(args.gpu_ids),
        "expected_world_size": int(native.expected_world_size(args)),
        "batch_size": int(args.batch_size),
        "seq_len": int(args.seq_len),
        "total_steps_target": int(args.total_steps),
        "checkpoint_interval": int(args.checkpoint_interval),
        "fault_frequency": float(args.fault_frequency),
        "fault_check_interval_seconds": float(args.fault_check_interval_seconds),
        "fault_injection_count": int(fault_injection_count),
        "worker_restart_count": int(worker_restart_count),
        "average_steps_per_second": float(average_steps_per_second),
    }

    atomic_write_json(paths["state_file"], payload)
    atomic_write_json(paths["report_file"], payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), flush=True)

    if not args.keep_checkpoints:
        _cleanup_stale_shm(paths["checkpoint_dir"], world_size)
        shutil.rmtree(paths["checkpoint_dir"], ignore_errors=True)
    return 0




def main() -> int:
    """Entry point: dispatch to the worker or manager phase based on parsed args."""
    args = parse_args()
    native, checkfreq, pccheck = load_bench_modules()
    args = normalize_run_paths(args, native)
    validate_args(args, native)
    if args.phase == "worker":
        return run_worker(args, native, checkfreq, pccheck)
    return run_manager(args, native)


if __name__ == "__main__":
    raise SystemExit(main())
