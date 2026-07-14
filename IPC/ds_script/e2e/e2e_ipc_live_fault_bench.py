#!/usr/bin/env python3
"""End-to-end IPC live-state benchmark with random fault injection (single + multi GPU)."""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
LAB2_DIR = WORKSPACE_ROOT / "IPC" / "ds_script" / "fault"
DEFAULT_REPORT_ROOT = SCRIPT_DIR / "report1" / "e2e_ipc"


def add_lab2_path() -> None:
    if str(LAB2_DIR) not in sys.path:
        sys.path.insert(0, str(LAB2_DIR))


def load_bench_modules() -> tuple[Any, Any]:
    add_lab2_path()
    import ds_hf_checkpoint_bench as native
    import ipc_hf_attach_bench as ipc

    return native, ipc




def parse_args() -> argparse.Namespace:
    native, _ = load_bench_modules()
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end multi-GPU IPC live-state benchmark with a manager process "
            "that injects random fail-stop crashes and restarts a fresh worker group "
            "via torchrun.  State lives in per-GPU IPC daemons."
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
        "--state-file",
        type=Path,
        default=None,
        help="Parent summary JSON path. Defaults to report1/e2e_ipc/<label>_state.json.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Final JSON report path. Defaults to report1/e2e_ipc/<label>.json.",
    )
    parser.add_argument(
        "--fault-file",
        type=Path,
        default=None,
        help="JSONL file containing manager-side fault injection events.",
    )
    parser.add_argument(
        "--ipc-group",
        default=None,
        help="IPC tensor group prefix. Defaults to a model-specific e2e name.",
    )
    parser.add_argument("--master-port", type=int, default=30151)
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
    parser.add_argument("--zero-stage", type=int, default=3, choices=[0, 3])
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
        "--keep-ipc-server",
        action="store_true",
        help="Keep the IPC daemon alive after the benchmark completes.",
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
    if args.state_file is None:
        args.state_file = DEFAULT_REPORT_ROOT / f"{label}_state.json"
    if args.report_file is None:
        args.report_file = DEFAULT_REPORT_ROOT / f"{label}.json"
    if args.fault_file is None:
        args.fault_file = DEFAULT_REPORT_ROOT / f"{label}_faults.jsonl"
    return args


def resolve_ipc_group(args: argparse.Namespace, native: Any) -> str:
    if args.ipc_group:
        return str(args.ipc_group)
    label = native.model_label_from_arg(args.model, args.model_label).replace(" ", "_")
    return f"e2e_ipc_{label}"


def validate_args(args: argparse.Namespace, native: Any) -> None:
    if not hasattr(args, "steps_per_epoch"):
        args.steps_per_epoch = int(args.total_steps)
    native.validate_args(args)
    if args.total_steps <= 0:
        native.fail("--total-steps must be positive.")
    if args.zero_stage != 3:
        native.fail("IPC e2e benchmark currently requires --zero-stage 3.")
    if args.fault_frequency < 0.0 or args.fault_frequency > 100.0:
        native.fail("--fault-frequency must be within [0, 100].")
    if args.fault_check_interval_seconds <= 0.0:
        native.fail("--fault-check-interval-seconds must be positive.")
    if args.max_unexpected_worker_exits < 0:
        native.fail("--max-unexpected-worker-exits must be non-negative.")




def derived_paths(args: argparse.Namespace) -> dict[str, Path]:
    state_file = args.state_file.resolve()
    report_file = args.report_file.resolve()
    fault_file = args.fault_file.resolve()
    return {
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


def normalize_step_value(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def summarize_persisted_zero3_step(engine: Any) -> dict[str, Any]:
    zero_optimizer = getattr(engine, "optimizer", None)
    get_step_fn = getattr(zero_optimizer, "_zero3_ipc_get_optimizer_step", None)
    fp16_groups = getattr(zero_optimizer, "fp16_groups", None)
    if not callable(get_step_fn) or fp16_groups is None:
        return {
            "min_step": 0,
            "max_step": 0,
            "step_count": 0,
            "steps_consistent": True,
        }

    steps: list[int] = []
    for sub_group_id in range(len(fp16_groups)):
        step_value = normalize_step_value(get_step_fn(sub_group_id))
        if step_value is not None:
            steps.append(step_value)

    if not steps:
        return {
            "min_step": 0,
            "max_step": 0,
            "step_count": 0,
            "steps_consistent": True,
        }

    min_step = min(steps)
    max_step = max(steps)
    return {
        "min_step": int(min_step),
        "max_step": int(max_step),
        "step_count": int(len(steps)),
        "steps_consistent": bool(min_step == max_step),
    }




def record_runtime_state(
    runtime_file: Path,
    step: int,
    last_loss: float,
    parent_start_wall_time: float | None,
    group_name: str,
    worker_role: str,
    resumed_inflight_step: bool,
    persisted_step_summary: dict[str, Any],
    resume_debug: dict[str, Any],
) -> None:
    now = time.time()
    payload = {
        "last_completed_step": int(step),
        "last_completed_wall_time": float(now),
        "last_completed_relative_time_seconds": (
            None if parent_start_wall_time is None else float(now - parent_start_wall_time)
        ),
        "last_loss": float(last_loss),
        "ipc_group": str(group_name),
        "worker_role": str(worker_role),
        "resumed_inflight_step": bool(resumed_inflight_step),
        "persisted_min_step": int(persisted_step_summary.get("min_step", 0) or 0),
        "persisted_max_step": int(persisted_step_summary.get("max_step", 0) or 0),
        "persisted_step_count": int(persisted_step_summary.get("step_count", 0) or 0),
        "persisted_steps_consistent": bool(persisted_step_summary.get("steps_consistent", True)),
        "resume_detected_active_sub_group": resume_debug.get("detected_active_sub_group"),
        "resume_rolled_back_sub_group": resume_debug.get("rolled_back_sub_group"),
        "resume_start_sub_group": resume_debug.get("resume_start_sub_group"),
        "resume_total_subgroups": resume_debug.get("resume_total_subgroups"),
        "resume_replayed_subgroup_count": int(len(resume_debug.get("replayed_sub_groups", []))),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(runtime_file, payload)




def run_worker(args: argparse.Namespace, native: Any, ipc: Any) -> int:
    """Worker phase: train with optimizer+model state kept live in the per-GPU IPC daemons."""
    paths = derived_paths(args)
    runtime_state = load_json_if_exists(paths["runtime_file"]) or {}
    previous_completed_step = int(runtime_state.get("last_completed_step", 0) or 0)
    previous_last_loss = float(runtime_state.get("last_loss", 0.0) or 0.0)
    group_name = resolve_ipc_group(args, native)

    engine = None
    torch = None
    worker_role = "creator"
    resumed_inflight_step = False
    persisted_step_summary: dict[str, Any] = {
        "min_step": 0,
        "max_step": 0,
        "step_count": 0,
        "steps_consistent": True,
    }
    resume_debug: dict[str, Any] = {"resumed": False}

    try:
        (
            torch,
            deepspeed,
            auto_model_cls,
            auto_tokenizer_cls,
            auto_config_cls,
            load_dataset_fn,
            tool_cls,
            _ipc_socket_cls,
            _socket_path_for_gpu_id_fn,
        ) = ipc.load_runtime_modules()
        device = native.init_distributed(args, torch, deepspeed)
        native.seed_everything(args.seed, torch)

        clear_ipc_env()
        tool = ipc.build_tool(device, tool_cls)
        group_exists = ipc.optimizer_ipc_group_exists(tool, group_name)

        if group_exists:
            worker_role = "connector"
            model = ipc.build_connector_model(args, torch, auto_model_cls, auto_config_cls)
            ipc.configure_zero3_ipc_env(group_name, "ds_tool", require_existing=True)
        else:
            if previous_completed_step > 0:
                native.fail(
                    f"IPC group '{group_name}' is missing while runtime already records "
                    f"step={previous_completed_step}. Refusing to restart from scratch."
                )
            model = ipc.build_creator_model(args, torch, auto_model_cls)
            ipc.configure_zero3_ipc_env(group_name, "ds_tool", require_existing=False)

        ipc.configure_zero3_ipc_fault_env(args, enabled=False)
        engine = ipc.create_engine(args, torch, deepspeed, model)

        if worker_role == "connector":
            resume_fn = getattr(engine.optimizer, "resume_ipc_inflight_step_if_needed", None)
            if callable(resume_fn):
                resumed_inflight_step = bool(resume_fn())
            resume_debug = getattr(engine.optimizer, "_ipc_resume_state", {"resumed": False})
            if not resumed_inflight_step:
                ipc.restore_model_partitions_from_zero3_state(torch, engine)

        persisted_step_summary = summarize_persisted_zero3_step(engine)
        start_step = max(previous_completed_step, int(persisted_step_summary["min_step"]))
        last_loss = previous_last_loss

        tokenizer = native.load_tokenizer(args, auto_tokenizer_cls)
        blocks, _ = native.build_token_blocks(args, tokenizer, load_dataset_fn)
        available_step_count = native.batches_per_epoch(blocks, args.batch_size)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_token_id is None:
            native.fail("Tokenizer must provide either pad_token_id or eos_token_id.")

        if rank0():
            record_runtime_state(
                paths["runtime_file"],
                start_step,
                last_loss,
                args.parent_start_wall_time,
                group_name,
                worker_role,
                resumed_inflight_step,
                persisted_step_summary,
                resume_debug,
            )
        native.log(
            f"[worker] launch={args.worker_launch_index} role={worker_role} step={start_step}/{args.total_steps} "
            f"ipc_group={group_name} resumed_inflight={resumed_inflight_step} gpu_ids={args.gpu_ids}"
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
            last_loss = ipc.run_one_step(torch, engine, batch, 0, step_idx, args.total_steps)
            current_step = step_idx + 1
            if rank0():
                record_runtime_state(
                    paths["runtime_file"],
                    current_step,
                    last_loss,
                    args.parent_start_wall_time,
                    group_name,
                    worker_role,
                    False,
                    summarize_persisted_zero3_step(engine),
                    {"resumed": False},
                )
            if args.log_interval > 0 and (
                current_step % args.log_interval == 0 or current_step == args.total_steps
            ):
                native.log(
                    f"[worker] launch={args.worker_launch_index} step={current_step}/{args.total_steps} "
                    f"loss={last_loss:.6f}"
                )
        return 0
    finally:
        clear_ipc_env()
        if engine is not None and torch is not None:
            ipc.destroy_training_state(torch, engine)




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
        "--state-file", str(args.state_file),
        "--report-file", str(args.report_file),
        "--fault-file", str(args.fault_file),
        "--ipc-group", str(resolve_ipc_group(args, native)),
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
    if args.keep_ipc_server:
        phase_args.append("--keep-ipc-server")

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
    """Hard-kill the worker process tree (torchrun agent + ranks) but spare the IPC daemon (``--run-server``)."""
    import psutil

    daemon_marker = "--run-server"
    try:
        parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return
    try:
        descendants = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    victims = []
    for proc in [parent, *descendants]:
        try:
            cmdline = " ".join(proc.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            cmdline = ""
        if daemon_marker in cmdline:
            continue
        victims.append(proc)
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


def initialize_manager_files(
    paths: dict[str, Path],
    ipc_socket_cls: Any,
    socket_path_for_gpu_id_fn: Any,
    ipc: Any,
    gpu_ids: list[str],
) -> None:
    ipc.shutdown_ipc_servers(ipc_socket_cls, socket_path_for_gpu_id_fn, gpu_ids)
    for path in (
        paths["state_file"],
        paths["report_file"],
        paths["fault_file"],
        paths["runtime_file"],
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()


def run_manager(args: argparse.Namespace, native: Any, ipc: Any) -> int:
    """Manager phase: inject random fail-stop crashes and relaunch the worker until the run completes."""
    paths = derived_paths(args)
    (
        _torch,
        _deepspeed,
        _auto_model_cls,
        _auto_tokenizer_cls,
        _auto_config_cls,
        _load_dataset_fn,
        _tool_cls,
        ipc_socket_cls,
        socket_path_for_gpu_id_fn,
    ) = ipc.load_runtime_modules()
    gpu_list = native.parse_gpu_ids(args.gpu_ids)
    initialize_manager_files(paths, ipc_socket_cls, socket_path_for_gpu_id_fn, ipc, gpu_list)

    rng = random.Random(args.seed)
    parent_start_wall_time = time.time()
    fault_injection_count = 0
    worker_launch_count = 0
    worker_restart_count = 0
    unexpected_worker_exit_count = 0
    fault_check_draw_count = 0

    try:
        with paths["fault_file"].open("a", encoding="utf-8") as fault_handle:
            append_jsonl_record(
                fault_handle,
                {
                    "event": "manager_start",
                    "wall_time": float(parent_start_wall_time),
                    "fault_frequency": float(args.fault_frequency),
                    "fault_check_interval_seconds": float(args.fault_check_interval_seconds),
                    "total_steps": int(args.total_steps),
                    "ipc_group": str(resolve_ipc_group(args, native)),
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
    finally:
        if not args.keep_ipc_server:
            ipc.shutdown_ipc_servers(ipc_socket_cls, socket_path_for_gpu_id_fn, gpu_list)

    total_elapsed_seconds = time.time() - parent_start_wall_time
    runtime_state = load_json_if_exists(paths["runtime_file"]) or {}
    completed_steps = int(runtime_state.get("last_completed_step", 0) or 0)
    average_steps_per_second = (
        float(completed_steps / total_elapsed_seconds) if total_elapsed_seconds > 0.0 else 0.0
    )

    mode_suffix = "" if native.launch_mode(args) == "single" else "_multigpu"
    payload = {
        "experiment": f"e2e_ipc_live_faults{mode_suffix}",
        "checkpoint_backend": f"ipc_live_daemon{mode_suffix}",
        "model": args.model,
        "model_label": native.model_label_from_arg(args.model, args.model_label),
        "dtype": args.dtype,
        "recomputation": bool(args.gradient_checkpointing),
        "zero_stage": int(args.zero_stage),
        "gpu_ids": str(args.gpu_ids),
        "expected_world_size": int(native.expected_world_size(args)),
        "batch_size": int(args.batch_size),
        "seq_len": int(args.seq_len),
        "ipc_group": str(resolve_ipc_group(args, native)),
        "total_steps_target": int(args.total_steps),
        "fault_frequency": float(args.fault_frequency),
        "fault_check_interval_seconds": float(args.fault_check_interval_seconds),
        "fault_injection_count": int(fault_injection_count),
        "worker_restart_count": int(worker_restart_count),
        "average_steps_per_second": float(average_steps_per_second),
    }

    atomic_write_json(paths["state_file"], payload)
    atomic_write_json(paths["report_file"], payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return 0




def main() -> int:
    """Entry point: dispatch to the worker or manager phase based on parsed args."""
    args = parse_args()
    native, ipc = load_bench_modules()
    args = normalize_run_paths(args, native)
    validate_args(args, native)
    if args.phase == "worker":
        return run_worker(args, native, ipc)
    return run_manager(args, native, ipc)


if __name__ == "__main__":
    raise SystemExit(main())
