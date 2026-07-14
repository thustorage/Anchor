#!/usr/bin/env python3
"""Unified CheckFreq baseline (ZeRO-3 only): consistent CPU snapshot at the iteration boundary plus async CPU->disk persist."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import ds_hf_checkpoint_bench as native


SAVE_EPOCH = native.SAVE_EPOCH
FAULT_EPOCH = SAVE_EPOCH
RESUME_EPOCHS = native.RESUME_EPOCHS
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_ROOT = SCRIPT_DIR / "checkfreq_ckpt"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "report1" / "checkfreq"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one HF causal LM with DeepSpeed ZeRO-3, take a CheckFreq-style "
            "consistent CPU snapshot of the optimizer state (pinned buffers), "
            "persist it asynchronously to disk, relaunch a connector, and report "
            "snapshot/recovery timing. Single-GPU is world_size 1."
        )
    )
    parser.add_argument("--phase", choices=["creator", "connector", "all"], default="all")
    parser.add_argument("--model", required=True, help="Local HF causal LM path or model identifier.")
    parser.add_argument("--model-label", default=None, help="Short label used in report filenames and JSON output.")
    parser.add_argument(
        "--gpu-ids",
        default=os.environ.get("GPU_IDS", "0"),
        help="Comma-separated local physical GPU ids. One id -> single-GPU (world_size 1).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Directory used to save and reload asynchronous checkpoints.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_REPORT_DIR / "state.json",
        help="Creator/connector shared state JSON. Kept outside ckpt to avoid accidental deletion.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=DEFAULT_REPORT_DIR / "report.json",
        help="Final JSON report path.",
    )
    parser.add_argument("--master-port", type=int, default=29931)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--dataset-dir", type=Path, default=native.DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--dataset-split",
        choices=("train", "validation", "test"),
        default=native.DATASET_SPLIT,
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Per-GPU micro batch size.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--zero-stage", type=int, default=3, choices=[3])
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--keep-checkpoints", action="store_true")
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
        native.fail("CheckFreq baseline currently supports only --zero-stage 3.")


def clone_to_cpu(value: Any) -> Any:
    """Per-tensor detach().cpu().clone() to CPU (fresh allocation, blocking D2H)."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(item) for item in value)
    return value


def move_optimizer_state_to_device(optimizer: Any, device: Any) -> None:
    state = getattr(optimizer, "state", None)
    if isinstance(state, dict):
        for param_state in state.values():
            if isinstance(param_state, dict):
                for key, value in list(param_state.items()):
                    if torch.is_tensor(value):
                        param_state[key] = value.to(device=device, non_blocking=True)
    inner = getattr(optimizer, "optimizer", None)
    if inner is not None and inner is not optimizer:
        move_optimizer_state_to_device(inner, device)


def prepare_optimizer_state_for_load(optimizer: Any, optimizer_state: Any, rank: int) -> Any:
    if not isinstance(optimizer_state, dict):
        return optimizer_state
    if any(isinstance(key, int) for key in optimizer_state.keys()):
        return optimizer_state

    module_name = type(optimizer).__module__
    if "deepspeed.runtime.zero.stage3" in module_name:
        return {rank: optimizer_state}
    if "deepspeed.runtime.zero.stage_1_and_2" in module_name:
        return {rank: optimizer_state}
    return optimizer_state


def restore_model_partitions_from_zero3_state(torch_mod: Any, engine: Any) -> None:
    """Push restored fp32 master partitions back into the fp16 module params."""
    zero_optimizer = getattr(engine, "optimizer", None)
    if zero_optimizer is None:
        return
    restore_fn = getattr(zero_optimizer, "_reassign_or_swap_out_partitioned_parameters", None)
    fp32_groups = getattr(zero_optimizer, "fp32_partitioned_groups_flat", None)
    if restore_fn is None or fp32_groups is None:
        return
    with torch_mod.no_grad():
        for sub_group_id in range(len(fp32_groups)):
            restore_fn(sub_group_id)
    native.sync_device(torch_mod, engine.device)


class AsyncDistributedCheckpointer:
    """Per-rank async checkpointer: sync clone_to_cpu snapshot + background disk persist."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.worker: threading.Thread | None = None
        self.pending_meta: dict[str, Any] | None = None

    @staticmethod
    def _atomic_torch_save(payload: Any, path: Path) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        with open(tmp_path, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)

    def _persist_rank_snapshot(
        self,
        rank_snapshot: dict[str, Any],
        optimizer_path: Path,
        model_snapshot: dict[str, Any] | None,
        model_path: Path,
        meta: dict[str, Any],
    ) -> None:
        try:
            persist_start = time.perf_counter()
            optimizer_path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_torch_save(rank_snapshot, optimizer_path)
            if model_snapshot is not None:
                self._atomic_torch_save(model_snapshot, model_path)
            meta["persist_seconds"] = time.perf_counter() - persist_start
        except Exception as exc:
            meta["error"] = f"{type(exc).__name__}: {exc}"

    def wait(self) -> dict[str, Any] | None:
        if self.worker is None or self.pending_meta is None:
            return None
        self.worker.join()
        meta = self.pending_meta
        self.worker = None
        self.pending_meta = None
        if meta.get("error"):
            raise RuntimeError(f"Checkpoint persistence failed: {meta['error']}")
        return meta

    def save_async(
        self,
        torch_mod: Any,
        engine: Any,
        epoch: int,
        last_loss: float,
        model_label: str,
    ) -> dict[str, Any]:
        self.wait()
        native.distributed_barrier(torch_mod)
        native.sync_device(torch_mod, engine.device)

        rank = current_rank()
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        tag = f"epoch_{epoch:06d}"
        tag_dir = self.checkpoint_dir / tag

        snapshot_start = time.perf_counter()
        optimizer_snapshot = {
            "optimizer": clone_to_cpu(engine.optimizer.state_dict()),
            "rank": rank,
            "world_size": world_size,
            "zero_stage": int(getattr(engine, "zero_optimization_stage", lambda: 0)()),
        }
        snapshot_seconds = time.perf_counter() - snapshot_start

        model_snapshot = None
        if rank0():
            model_snapshot = {
                "model": None,
                "client_state": {
                    "train_epoch": epoch,
                    "last_loss": last_loss,
                    "model_label": model_label,
                },
                "model_label": model_label,
                "world_size": world_size,
                "zero_stage": int(getattr(engine, "zero_optimization_stage", lambda: 0)()),
            }

        optimizer_path = tag_dir / f"optim_rank_{rank:02d}.pt"
        model_path = tag_dir / "model.pt"
        meta = {
            "tag": tag,
            "tag_dir": tag_dir,
            "optimizer_path": optimizer_path,
            "model_path": model_path,
            "snapshot_seconds": snapshot_seconds,
            "persist_seconds": None,
            "error": None,
        }
        self.pending_meta = meta
        self.worker = threading.Thread(
            target=self._persist_rank_snapshot,
            args=(optimizer_snapshot, optimizer_path, model_snapshot, model_path, meta),
            daemon=True,
        )
        self.worker.start()
        return meta


def latest_checkpoint_dir(checkpoint_root: Path) -> Path:
    latest_file = checkpoint_root / "latest"
    if not latest_file.exists():
        native.fail(f"Latest checkpoint marker not found in {checkpoint_root}")
    checkpoint_name = latest_file.read_text(encoding="utf-8").strip()
    if not checkpoint_name:
        native.fail(f"Latest checkpoint marker is empty in {checkpoint_root}")
    checkpoint_dir = checkpoint_root / checkpoint_name
    if not checkpoint_dir.exists():
        native.fail(f"Latest checkpoint dir does not exist: {checkpoint_dir}")
    return checkpoint_dir


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def destroy_engine(torch_mod: Any, engine: Any) -> None:
    del engine
    gc.collect()
    if torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()


def report_common_metadata(args: argparse.Namespace, stats: dict[str, Any]) -> dict[str, Any]:
    payload = native.report_common_metadata(args, stats)
    payload["checkpoint_backend"] = "checkfreq_async_cpu_snapshot_zero3"
    return payload


def write_manifest(checkpoint_root: Path, tag: str, world_size: int, model_label: str) -> None:
    tag_dir = checkpoint_root / tag
    payload = {
        "tag": tag,
        "world_size": world_size,
        "model_file": "model.pt",
        "optimizer_files": [f"optim_rank_{rank:02d}.pt" for rank in range(world_size)],
        "model_label": model_label,
    }
    manifest_path = tag_dir / "manifest.json"
    latest_path = checkpoint_root / "latest"
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    latest_tmp = latest_path.with_name(latest_path.name + ".tmp")
    manifest_tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with open(manifest_tmp, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(manifest_tmp, manifest_path)
    latest_tmp.write_text(tag + "\n", encoding="utf-8")
    with open(latest_tmp, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(latest_tmp, latest_path)


def run_creator(args: argparse.Namespace) -> int:
    """Train one epoch, take an async CheckFreq snapshot of the optimizer state, and persist it to disk."""
    checkpoint_dir = args.checkpoint_dir.resolve()
    state_file = args.state_file.resolve()
    if rank0():
        clear_directory(checkpoint_dir)
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
    checkpointer = AsyncDistributedCheckpointer(checkpoint_dir)

    native.log(
        f"[creator] model={model_label} "
        f"steps_per_epoch={epoch_step_count} "
        f"available_steps_per_epoch={available_step_count} "
        f"dataset_rows={data_stats['dataset_rows']} "
        f"token_blocks={data_stats['token_blocks']}"
    )

    try:
        epoch_seconds, last_loss = native.run_epoch(
            torch_mod,
            engine,
            blocks,
            args.batch_size,
            args.seq_len,
            int(pad_token_id),
            epoch_step_count,
            SAVE_EPOCH,
            args.log_interval,
        )

        queued_meta = checkpointer.save_async(torch_mod, engine, SAVE_EPOCH, last_loss, model_label)
        native.log(
            f"[creator] queued async checkpoint tag={queued_meta['tag']} "
            f"snapshot_seconds={queued_meta['snapshot_seconds']:.6f}"
        )
        completed_meta = checkpointer.wait()
        if completed_meta is None or completed_meta["persist_seconds"] is None:
            native.fail("Failed to persist async checkpoint before creator exit.")

        snapshot_seconds = native.max_across_ranks(
            torch_mod,
            completed_meta["snapshot_seconds"],
            engine.device,
        )
        persist_seconds = native.max_across_ranks(
            torch_mod,
            completed_meta["persist_seconds"],
            engine.device,
        )
        native.distributed_barrier(torch_mod)
        if rank0():
            write_manifest(checkpoint_dir, completed_meta["tag"], int(os.environ.get("WORLD_SIZE", "1")), model_label)
        native.distributed_barrier(torch_mod)

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
            "checkpoint_tag": str(completed_meta["tag"]),
            "epoch_train_seconds": float(epoch_seconds),
            "last_loss": float(last_loss),
            "train_epoch": SAVE_EPOCH,
            "checkpoint_save_seconds": float(snapshot_seconds),
            "checkpoint_snapshot_seconds": float(snapshot_seconds),
            "checkpoint_persist_seconds": float(persist_seconds),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        if rank0():
            native.maybe_write_json(state_file, payload)
        native.log(
            f"[creator] model={payload['model_label']} "
            f"snapshot_seconds={payload['checkpoint_save_seconds']:.6f} "
            f"persist_seconds={payload['checkpoint_persist_seconds']:.6f}"
        )
        return 0
    finally:
        destroy_engine(torch_mod, engine)
        native.destroy_process_group(torch_mod)


def run_connector(args: argparse.Namespace) -> int:
    """Relaunch, reload the async CheckFreq checkpoint, restore state, and report recovery timing."""
    connector_entry_time = time.perf_counter()
    restart_start_time = args.restart_start_time if args.restart_start_time is not None else connector_entry_time
    state = native.load_json(args.state_file.resolve())

    engine_init_start = time.perf_counter()
    torch_mod, engine, _, _, tokenizer, load_dataset_fn, _ = native.create_engine(args)
    checkpoint_root = Path(state["checkpoint_dir"]).resolve()
    checkpoint_dir = latest_checkpoint_dir(checkpoint_root)
    rank = current_rank()
    optimizer_path = checkpoint_dir / f"optim_rank_{rank:02d}.pt"
    model_path = checkpoint_dir / "model.pt"
    checkpoint_load_seconds = 0.0
    recovery_total_seconds = 0.0
    connector_other_seconds = 0.0
    client_state: dict[str, Any] = {}

    try:
        engine_init_seconds = time.perf_counter() - engine_init_start
        engine_init_seconds = native.max_across_ranks(torch_mod, engine_init_seconds, engine.device)

        native.distributed_barrier(torch_mod)
        native.sync_device(torch_mod, engine.device)
        load_start = time.perf_counter()
        model_payload = torch.load(model_path, map_location="cpu", weights_only=False)
        optimizer_payload = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        engine.optimizer.load_state_dict(
            prepare_optimizer_state_for_load(engine.optimizer, optimizer_payload["optimizer"], rank)
        )
        move_optimizer_state_to_device(engine.optimizer, engine.device)
        client_state = dict(model_payload.get("client_state", {}))
        del model_payload
        del optimizer_payload
        gc.collect()
        native.sync_device(torch_mod, engine.device)
        native.distributed_barrier(torch_mod)
        checkpoint_load_seconds = time.perf_counter() - load_start
        checkpoint_load_seconds = native.max_across_ranks(torch_mod, checkpoint_load_seconds, engine.device)

        recovery_total_seconds = time.perf_counter() - restart_start_time
        recovery_total_seconds = native.max_across_ranks(torch_mod, recovery_total_seconds, engine.device)
        connector_other_seconds = max(0.0, recovery_total_seconds - engine_init_seconds - checkpoint_load_seconds)
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
            shutil.rmtree(checkpoint_root, ignore_errors=True)
    return 0


def build_phase_args(args: argparse.Namespace, phase: str) -> list[str]:
    script_path = Path(__file__).resolve()
    cmd = [
        str(script_path),
        "--phase",
        phase,
        "--model",
        str(args.model),
        "--model-label",
        str(args.model_label or ""),
        "--gpu-ids",
        str(args.gpu_ids),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--state-file",
        str(args.state_file),
        "--report-file",
        str(args.report_file),
        "--master-port",
        str(args.master_port),
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
    if not args.model_label:
        del cmd[6:8]
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
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(gpu_list)),
        "--master_port",
        str(args.master_port),
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
