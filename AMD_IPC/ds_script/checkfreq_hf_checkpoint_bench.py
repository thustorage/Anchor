#!/usr/bin/env python3
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


CREATOR_FAILURE_EXIT_CODE = native.CREATOR_FAILURE_EXIT_CODE
FAULT_EPOCH = native.FAULT_EPOCH
RESUME_EPOCHS = native.RESUME_EPOCHS
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_ROOT = SCRIPT_DIR / "checkfreq_ckpt"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "report_checkfreq"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one HF causal LM with DeepSpeed, snapshot model/optimizer state "
            "to CPU memory at the failure boundary, persist it asynchronously to "
            "disk, relaunch a connector, and report snapshot/recovery timing."
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
                        help="Directory used to save and reload asynchronous checkpoints.")
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
                        default=29641,
                        help="Port used for the single-rank DeepSpeed runtime.")
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


def prepare_optimizer_state_for_load(optimizer: Any, optimizer_state: Any) -> Any:
    if not isinstance(optimizer_state, dict):
        return optimizer_state

    if any(isinstance(key, int) for key in optimizer_state.keys()):
        return optimizer_state

    module_name = type(optimizer).__module__
    if "deepspeed.runtime.zero.stage_1_and_2" in module_name:
        return {0: optimizer_state}
    if "deepspeed.runtime.zero.stage3" in module_name:
        return {0: optimizer_state}

    return optimizer_state


def snapshot_model_state(engine: Any, zero_stage: int) -> dict[str, Any] | None:
    if zero_stage == 3:
        return None
    return clone_to_cpu(engine.module.state_dict())


def load_model_state(engine: Any, model_state: dict[str, Any] | None, zero_stage: int) -> None:
    if zero_stage != 3:
        if model_state is None:
            native.fail("Model state is required for non-ZeRO-3 CheckFreq recovery.")
        engine.module.load_state_dict(model_state)
        return
    return


class AsyncHFCheckpointer:

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.latest_file = checkpoint_dir / "latest"
        self.worker: threading.Thread | None = None
        self.pending_meta: dict[str, Any] | None = None

    def _persist_snapshot(self, snapshot: dict[str, Any], path: Path, meta: dict[str, Any]) -> None:
        try:
            persist_start = time.perf_counter()
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            torch.save(snapshot, tmp_path)
            with open(tmp_path, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)

            latest_tmp = self.latest_file.with_name(self.latest_file.name + ".tmp")
            latest_tmp.write_text(path.name + "\n", encoding="utf-8")
            with open(latest_tmp, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(latest_tmp, self.latest_file)

            meta["persist_seconds"] = time.perf_counter() - persist_start
            meta["size_bytes"] = native.path_size_bytes(path, apparent=False)
            meta["apparent_size_bytes"] = native.path_size_bytes(path, apparent=True)
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

    def save_async(self,
                   torch_mod: Any,
                   engine: Any,
                   epoch: int,
                   last_loss: float,
                   model_metadata: dict[str, Any]) -> dict[str, Any]:
        self.wait()
        native.distributed_barrier(torch_mod)
        native.sync_device(torch_mod, engine.device)

        snapshot_start = time.perf_counter()
        snapshot = {
            "model": snapshot_model_state(engine, zero_stage=getattr(engine, "zero_optimization_stage", lambda: 0)()),
            "optimizer": clone_to_cpu(engine.optimizer.state_dict()),
            "client_state": {
                "train_epoch": epoch,
                "last_loss": last_loss,
                "model_label": model_metadata["model_label"],
            },
        }
        snapshot_seconds = time.perf_counter() - snapshot_start

        tag = f"epoch_{epoch:06d}"
        path = self.checkpoint_dir / f"{tag}.pt"
        meta = {
            "epoch": epoch,
            "tag": tag,
            "path": path,
            "snapshot_seconds": snapshot_seconds,
            "persist_seconds": None,
            "size_bytes": None,
            "apparent_size_bytes": None,
            "error": None,
        }
        self.pending_meta = meta
        self.worker = threading.Thread(target=self._persist_snapshot, args=(snapshot, path, meta), daemon=True)
        self.worker.start()
        return meta


def latest_checkpoint_path(checkpoint_dir: Path) -> Path:
    latest_file = checkpoint_dir / "latest"
    if not latest_file.exists():
        native.fail(f"Latest checkpoint marker not found in {checkpoint_dir}")
    checkpoint_name = latest_file.read_text(encoding="utf-8").strip()
    if not checkpoint_name:
        native.fail(f"Latest checkpoint marker is empty in {checkpoint_dir}")
    checkpoint_path = checkpoint_dir / checkpoint_name
    if not checkpoint_path.exists():
        native.fail(f"Latest checkpoint file does not exist: {checkpoint_path}")
    return checkpoint_path


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
    payload["checkpoint_backend"] = "checkfreq_async_cpu_snapshot"
    return payload


def run_creator(args: argparse.Namespace) -> int:
    """Run initial training, snapshot state to CPU, persist it asynchronously, then inject the fault."""
    checkpoint_dir = args.checkpoint_dir.resolve()
    state_file = args.state_file.resolve()
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
    checkpointer = AsyncHFCheckpointer(checkpoint_dir)

    native.log(
        f"[creator] model={model_label} "
        f"steps_per_epoch={epoch_step_count} "
        f"available_steps_per_epoch={available_step_count} "
        f"dataset_rows={data_stats['dataset_rows']} "
        f"token_blocks={data_stats['token_blocks']}"
    )

    epoch_seconds = 0.0
    last_loss = 0.0
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

        queued_meta = checkpointer.save_async(
            torch_mod,
            engine,
            FAULT_EPOCH,
            last_loss,
            {"model_label": model_label},
        )
        native.log(
            f"[creator] queued async checkpoint tag={queued_meta['tag']} "
            f"snapshot_seconds={queued_meta['snapshot_seconds']:.6f}"
        )
        completed_meta = checkpointer.wait()
        if completed_meta is None or completed_meta["persist_seconds"] is None:
            native.fail("Failed to persist async checkpoint before fault injection.")

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
            "checkpoint_file": completed_meta["path"].name,
            "epoch_train_seconds": float(epoch_seconds),
            "last_loss": float(last_loss),
            "checkpoint_save_seconds": float(completed_meta["snapshot_seconds"]),
            "checkpoint_snapshot_seconds": float(completed_meta["snapshot_seconds"]),
            "checkpoint_persist_seconds": float(completed_meta["persist_seconds"]),
            "checkpoint_size_bytes": int(completed_meta["size_bytes"]),
            "checkpoint_size_gib": float(completed_meta["size_bytes"]) / (1024**3),
            "checkpoint_apparent_size_bytes": int(completed_meta["apparent_size_bytes"]),
            "checkpoint_apparent_size_gib": float(completed_meta["apparent_size_bytes"]) / (1024**3),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        native.maybe_write_json(state_file, payload)
        native.log(
            f"[creator] model={payload['model_label']} "
            f"snapshot_seconds={payload['checkpoint_save_seconds']:.6f} "
            f"persist_seconds={payload['checkpoint_persist_seconds']:.6f} "
            f"checkpoint_size_gib={payload['checkpoint_size_gib']:.6f}"
        )
        return CREATOR_FAILURE_EXIT_CODE
    finally:
        destroy_engine(torch_mod, engine)
        native.destroy_process_group(torch_mod)


def run_connector(args: argparse.Namespace) -> int:
    """Reload the latest async snapshot, restore model/optimizer state, and resume training."""
    connector_entry_time = time.perf_counter()
    recovery_start_time = args.fault_start_time if args.fault_start_time is not None else connector_entry_time
    state = native.load_json(args.state_file.resolve())

    engine_init_start = time.perf_counter()
    torch_mod, engine, _, _, tokenizer, load_dataset_fn, _ = native.create_engine(args)
    checkpoint_dir = Path(state["checkpoint_dir"]).resolve()
    checkpoint_path = latest_checkpoint_path(checkpoint_dir)
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
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        load_model_state(engine, payload["model"], args.zero_stage)
        engine.optimizer.load_state_dict(prepare_optimizer_state_for_load(engine.optimizer, payload["optimizer"]))
        move_optimizer_state_to_device(engine.optimizer, engine.device)
        client_state = dict(payload.get("client_state", {}))
        del payload
        gc.collect()
        native.sync_device(torch_mod, engine.device)
        native.distributed_barrier(torch_mod)
        checkpoint_load_seconds = time.perf_counter() - load_start
        checkpoint_load_seconds = native.max_across_ranks(torch_mod, checkpoint_load_seconds, engine.device)

        recovery_total_seconds = time.perf_counter() - recovery_start_time
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

    payload = dict(state)
    payload.update({
        "fault_to_recovery_ready_seconds": float(recovery_total_seconds),
        "connector_engine_init_seconds": float(engine_init_seconds),
        "connector_checkpoint_load_seconds": float(checkpoint_load_seconds),
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
    """Entry point dispatching to the creator, connector, or both phases."""
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
