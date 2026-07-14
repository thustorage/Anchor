#!/usr/bin/env python3
"""End-to-end DeepSpeed native checkpoint benchmark with random fault injection (single + multi GPU)."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
FAULT_DIR = WORKSPACE_ROOT / "IPC" / "ds_script" / "fault"
DEFAULT_CHECKPOINT_ROOT = SCRIPT_DIR / "ckpt" / "e2e_ds_native"
DEFAULT_REPORT_ROOT = SCRIPT_DIR / "report1" / "e2e_ds_native"


def add_fault_path() -> None:
    if str(FAULT_DIR) not in sys.path:
        sys.path.insert(0, str(FAULT_DIR))


def load_native_module() -> Any:
    add_fault_path()
    import ds_hf_checkpoint_bench as native

    return native


def parse_args() -> argparse.Namespace:
    native = load_native_module()
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end multi-GPU DeepSpeed native checkpoint benchmark with a manager "
            "process that injects random fail-stop crashes and restarts a fresh worker group."
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
        help="Directory used to keep only the latest DeepSpeed checkpoint.",
    )
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--report-file", type=Path, default=None)
    parser.add_argument("--fault-file", type=Path, default=None)
    parser.add_argument("--master-port", type=int, default=30131)
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
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--fault-frequency", type=float, default=10.0)
    parser.add_argument("--fault-check-interval-seconds", type=float, default=1.0)
    parser.add_argument("--max-unexpected-worker-exits", type=int, default=5)
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--parent-start-wall-time", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-launch-index", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


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
        "manifest_file": checkpoint_dir / "resume_manifest.json",
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


def build_ds_config(args: argparse.Namespace, native: Any) -> dict[str, Any]:
    return native.build_ds_config(args)


def create_engine(args: argparse.Namespace, native: Any, skip_pretrained: bool = False) -> tuple[Any, Any, Any, Any]:
    torch, deepspeed, auto_model_cls, auto_tokenizer_cls, load_dataset_fn = native.load_runtime_modules()
    device = native.init_distributed(args, torch, deepspeed)
    native.seed_everything(args.seed, torch)
    if skip_pretrained and hasattr(native, "load_model_from_config"):
        model = native.load_model_from_config(args, torch, auto_model_cls)
    else:
        model = native.load_model(args, torch, auto_model_cls)
    tokenizer = native.load_tokenizer(args, auto_tokenizer_cls)
    ds_config = build_ds_config(args, native)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=ds_config,
    )
    return torch, engine, tokenizer, load_dataset_fn


def cleanup_old_checkpoint(checkpoint_dir: Path, previous_tag: str | None, latest_tag: str) -> None:
    if not previous_tag or previous_tag == latest_tag:
        return
    shutil.rmtree(checkpoint_dir / previous_tag, ignore_errors=True)


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


def run_worker(args: argparse.Namespace, native: Any) -> int:
    """Worker phase: train and periodically save native DeepSpeed distributed checkpoints."""
    paths = derived_paths(args)
    torch = None
    engine = None
    try:
        worker_started_at = time.time()
        manifest = load_json_if_exists(paths["manifest_file"]) or {}
        latest_tag = manifest.get("latest_checkpoint_tag")
        has_checkpoint = latest_tag is not None

        skip_pretrained = has_checkpoint and native.launch_mode(args) != "single"
        torch, engine, tokenizer, load_dataset_fn = create_engine(
            args, native, skip_pretrained=skip_pretrained,
        )
        blocks, _ = native.build_token_blocks(args, tokenizer, load_dataset_fn)
        available_step_count = native.batches_per_epoch(blocks, args.batch_size)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_token_id is None:
            native.fail("Tokenizer must provide either pad_token_id or eos_token_id.")

        latest_checkpoint_step = int(manifest.get("latest_checkpoint_step", 0) or 0)
        start_step = latest_checkpoint_step
        last_loss = float(manifest.get("latest_last_loss", 0.0) or 0.0)

        if latest_tag:
            native.distributed_barrier(torch)
            native.sync_device(torch, engine.device)
            load_path, client_state = engine.load_checkpoint(str(paths["checkpoint_dir"]), tag=str(latest_tag))
            native.sync_device(torch, engine.device)
            native.distributed_barrier(torch)
            if load_path is None or client_state is None:
                native.fail(
                    f"Failed to load latest checkpoint tag={latest_tag} from {paths['checkpoint_dir']}"
                )
            start_step = int(client_state.get("global_step", latest_checkpoint_step) or latest_checkpoint_step)
            latest_checkpoint_step = start_step
            last_loss = float(client_state.get("last_loss", last_loss) or last_loss)
            native.log(
                f"[worker] launch={args.worker_launch_index} resumed tag={latest_tag} "
                f"step={start_step}/{args.total_steps}"
            )
        else:
            native.log(
                f"[worker] launch={args.worker_launch_index} started step=0/{args.total_steps}"
            )

        if native.rank0():
            record_runtime_state(
                paths["runtime_file"],
                start_step,
                latest_checkpoint_step,
                None if latest_tag is None else str(latest_tag),
                last_loss,
                args.parent_start_wall_time,
            )

        for step_idx in range(start_step, args.total_steps):
            batch_index = step_idx % available_step_count
            batch = native.build_batch_from_blocks(
                torch,
                blocks,
                batch_index,
                args.batch_size,
                args.seq_len,
                int(pad_token_id),
                engine.device,
            )
            last_loss = native.run_one_step(engine, batch)
            current_step = step_idx + 1
            if native.rank0():
                record_runtime_state(
                    paths["runtime_file"],
                    current_step,
                    latest_checkpoint_step,
                    None if latest_tag is None else str(latest_tag),
                    last_loss,
                    args.parent_start_wall_time,
                )
            if args.log_interval > 0 and (
                current_step % args.log_interval == 0 or current_step == args.total_steps
            ):
                native.log(
                    f"[worker] launch={args.worker_launch_index} step={current_step}/{args.total_steps} "
                    f"loss={last_loss:.6f}"
                )

            if current_step % args.checkpoint_interval == 0:
                tag = f"step_{current_step:08d}"
                client_state = {
                    "global_step": int(current_step),
                    "last_loss": float(last_loss),
                    "model_label": native.model_label_from_arg(args.model, args.model_label),
                }
                previous_tag = None if latest_tag is None else str(latest_tag)
                shutil.rmtree(paths["checkpoint_dir"] / tag, ignore_errors=True)
                save_seconds, size_bytes, apparent_size_bytes = native.save_checkpoint(
                    torch,
                    engine,
                    paths["checkpoint_dir"],
                    tag,
                    client_state,
                )
                latest_tag = tag
                latest_checkpoint_step = current_step
                if native.rank0():
                    atomic_write_json(
                        paths["manifest_file"],
                        {
                            "latest_checkpoint_tag": str(tag),
                            "latest_checkpoint_step": int(current_step),
                            "latest_last_loss": float(last_loss),
                            "checkpoint_save_seconds": float(save_seconds),
                            "checkpoint_size_bytes": int(size_bytes),
                            "checkpoint_apparent_size_bytes": int(apparent_size_bytes),
                            "worker_launch_index": int(args.worker_launch_index),
                            "worker_started_wall_time": float(worker_started_at),
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    record_runtime_state(
                        paths["runtime_file"],
                        current_step,
                        latest_checkpoint_step,
                        str(latest_tag),
                        last_loss,
                        args.parent_start_wall_time,
                    )
                    cleanup_old_checkpoint(paths["checkpoint_dir"], previous_tag, str(tag))

        if native.rank0():
            record_runtime_state(
                paths["runtime_file"],
                int(args.total_steps),
                latest_checkpoint_step,
                None if latest_tag is None else str(latest_tag),
                last_loss,
                args.parent_start_wall_time,
            )
        return 0
    finally:
        if engine is not None and torch is not None:
            native.destroy_engine(torch, engine)
            native.destroy_process_group(torch)


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


def initialize_manager_files(paths: dict[str, Path]) -> None:
    native = load_native_module()
    native.clear_directory(paths["checkpoint_dir"])
    for path in (
        paths["state_file"],
        paths["report_file"],
        paths["fault_file"],
        paths["runtime_file"],
        paths["manifest_file"],
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()


def run_manager(args: argparse.Namespace, native: Any) -> int:
    """Manager phase: inject random fail-stop crashes and relaunch the worker until the run completes."""
    paths = derived_paths(args)
    initialize_manager_files(paths)

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
        "experiment": f"e2e_ds_native_checkpoint_faults{mode_suffix}",
        "checkpoint_backend": f"deepspeed_native_latest_only{mode_suffix}",
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
        shutil.rmtree(paths["checkpoint_dir"], ignore_errors=True)
    return 0


def main() -> int:
    """Entry point: dispatch to the worker or manager phase based on parsed args."""
    args = parse_args()
    native = load_native_module()
    args = normalize_run_paths(args, native)
    validate_args(args, native)
    if args.phase == "worker":
        return run_worker(args, native)
    return run_manager(args, native)


if __name__ == "__main__":
    raise SystemExit(main())
