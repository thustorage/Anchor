#!/usr/bin/env python3
"""End-to-end vLLM in-memory (shared-memory) KV checkpoint benchmark with random fault injection (single + multi GPU)."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_IPC_TOOL"] = "0"
os.environ.pop("VLLM_DEBUG_SCHEDULE_PREFILL_TOKENS", None)
os.environ.pop("VLLM_KV_CPU_CHECKPOINT_PATH", None)
os.environ.pop("VLLM_KV_CPU_BUFFER_METADATA", None)

from vllm import EngineArgs, LLMEngine, SamplingParams, _custom_ops as ops
from vllm.config import set_current_vllm_config
from vllm.sampling_params import RequestOutputKind
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.kv_cpu_buffer import (
    create_kv_cpu_buffer,
    prepare_kv_cpu_buffer,
    release_all_kv_cpu_buffers,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_ROOT = SCRIPT_DIR / "report1" / "e2e_vllm_kv_memory"
DEFAULT_SHAREGPT_DATASET_PATH = Path(
    "/root/mhy/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json"
)
REQUEST_ID_PREFIX = "sharegpt_request_"
RUNTIME_UPDATE_INTERVAL_SECONDS = 1.0
MANAGER_POLL_INTERVAL_SECONDS = 0.2
KV_BUFFER_ALIGNMENT_BYTES = 64


def str2bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def slugify_label(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "model"


def model_label_from_arg(model: str, model_label: str | None) -> str:
    if model_label:
        return str(model_label)
    model_path = Path(str(model))
    if model_path.name:
        return model_path.name
    return str(model)


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def uses_ranked_kv_checkpoint_files(tensor_parallel_size: int) -> bool:
    return int(tensor_parallel_size) > 1


def supports_shared_kv_buffer(tensor_parallel_size: int) -> bool:
    return not uses_ranked_kv_checkpoint_files(tensor_parallel_size)


def worker_kv_checkpoint_path(base_path: str | Path, rank: int) -> Path:
    base = Path(base_path)
    suffix = "".join(base.suffixes)
    stem = base.name[: -len(suffix)] if suffix else base.name
    filename = f"{stem}.rank{rank:02d}{suffix}" if suffix else f"{stem}.rank{rank:02d}"
    return base.with_name(filename)


def list_ranked_kv_checkpoint_paths(
    base_path: str | Path, tensor_parallel_size: int
) -> list[Path]:
    return [
        worker_kv_checkpoint_path(base_path, rank)
        for rank in range(int(tensor_parallel_size))
    ]


def cleanup_kv_checkpoint_files(base_path: str | Path, tensor_parallel_size: int) -> None:
    base = Path(base_path)
    try:
        base.unlink()
    except FileNotFoundError:
        pass

    if not uses_ranked_kv_checkpoint_files(tensor_parallel_size):
        return

    for path in list_ranked_kv_checkpoint_paths(base, tensor_parallel_size):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def ensure_kv_checkpoint_files_exist(
    base_path: str | Path, tensor_parallel_size: int
) -> None:
    base = Path(base_path)
    if not uses_ranked_kv_checkpoint_files(tensor_parallel_size):
        if not base.exists():
            raise FileNotFoundError(f"kv checkpoint file not found: {base}")
        return

    missing = [
        str(path)
        for path in list_ranked_kv_checkpoint_paths(base, tensor_parallel_size)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "worker kv checkpoint files not found: " + ", ".join(missing)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end vLLM KV-memory benchmark with a manager process that "
            "injects random faults and a lab1-style probe/creator/connector flow."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("manager", "probe", "creator", "connector"),
        default="manager",
    )
    parser.add_argument("--model", type=str, default="/root/mhy/model/llama3.1-8b")
    parser.add_argument("--model-label", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--trust-remote-code", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--enable-prefix-caching", type=str2bool, default=False)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_SHAREGPT_DATASET_PATH)
    parser.add_argument("--total-requests", type=int, default=10000)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--disable-shuffle", type=str2bool, default=False)
    parser.add_argument(
        "--fault-frequency",
        type=float,
        default=10.0,
        help=(
            "Every fault check, inject a fault when a random draw in [0, 100) "
            "is below this value."
        ),
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
        help="Abort after this many unexpected worker exits.",
    )
    parser.add_argument("--log-interval-requests", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Final manager summary JSON path.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Final benchmark report JSON path.",
    )
    parser.add_argument(
        "--fault-file",
        type=Path,
        default=None,
        help="JSONL file containing manager-side fault events.",
    )
    parser.add_argument(
        "--resume-state-file",
        type=Path,
        default=None,
        help="Creator/connector handoff JSON written when a fault is requested.",
    )
    parser.add_argument(
        "--recovery-profile-file",
        type=Path,
        default=None,
        help="Optional JSON file storing the first fault's recovery timing breakdown.",
    )
    parser.add_argument(
        "--recovery-profiles-file",
        type=Path,
        default=None,
        help="Optional JSONL file storing every fault's recovery timing breakdown.",
    )
    parser.add_argument(
        "--kv-checkpoint-file",
        type=Path,
        default=None,
        help="Fallback CPU KV checkpoint path when shared buffer is unavailable.",
    )
    parser.add_argument(
        "--kv-buffer-names",
        type=lambda s: [x for x in s.split(",") if x],
        default=None,
    )
    parser.add_argument("--kv-buffer-size-bytes", type=int, default=0)
    parser.add_argument("--kv-buffer-plan-file", type=Path, default=None)
    parser.add_argument("--fault-start-time", type=float, default=None)
    parser.add_argument("--parent-start-wall-time", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-launch-index", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def normalize_run_paths(args: argparse.Namespace) -> argparse.Namespace:
    label = slugify_label(model_label_from_arg(args.model, args.model_label))
    if args.state_file is None:
        args.state_file = DEFAULT_REPORT_ROOT / f"{label}_state.json"
    if args.report_file is None:
        args.report_file = DEFAULT_REPORT_ROOT / f"{label}.json"
    if args.fault_file is None:
        args.fault_file = DEFAULT_REPORT_ROOT / f"{label}_faults.jsonl"
    if args.resume_state_file is None:
        args.resume_state_file = DEFAULT_REPORT_ROOT / f"{label}_resume_state.json"
    if args.recovery_profile_file is None:
        state_stem = Path(args.state_file).stem
        if state_stem.endswith("_state"):
            state_stem = state_stem[: -len("_state")]
        args.recovery_profile_file = Path(args.state_file).with_name(
            f"{state_stem}_first_recovery.json"
        )
    if args.recovery_profiles_file is None:
        state_stem = Path(args.state_file).stem
        if state_stem.endswith("_state"):
            state_stem = state_stem[: -len("_state")]
        args.recovery_profiles_file = Path(args.state_file).with_name(
            f"{state_stem}_recovery_profiles.jsonl"
        )
    if args.kv_checkpoint_file is None:
        args.kv_checkpoint_file = Path("/dev/shm") / f"e2e_vllm_kv_memory_{label}.pt"
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.total_requests <= 0:
        raise SystemExit("--total-requests must be positive.")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive.")
    if args.max_num_batched_tokens <= 0:
        raise SystemExit("--max-num-batched-tokens must be positive.")
    if args.max_num_seqs <= 0:
        raise SystemExit("--max-num-seqs must be positive.")
    if args.max_model_len <= 0:
        raise SystemExit("--max-model-len must be positive.")
    if args.fault_frequency < 0.0 or args.fault_frequency > 100.0:
        raise SystemExit("--fault-frequency must be within [0, 100].")
    if args.fault_check_interval_seconds <= 0.0:
        raise SystemExit("--fault-check-interval-seconds must be positive.")
    if args.max_unexpected_worker_exits < 0:
        raise SystemExit("--max-unexpected-worker-exits must be non-negative.")
    if args.dataset_path is None:
        raise SystemExit("--dataset-path is required.")


def derived_paths(args: argparse.Namespace) -> dict[str, Path]:
    state_file = args.state_file.resolve()
    report_file = args.report_file.resolve()
    fault_file = args.fault_file.resolve()
    resume_state_file = args.resume_state_file.resolve()
    recovery_profile_file = args.recovery_profile_file.resolve()
    recovery_profiles_file = args.recovery_profiles_file.resolve()
    return {
        "state_file": state_file,
        "report_file": report_file,
        "fault_file": fault_file,
        "resume_state_file": resume_state_file,
        "recovery_profile_file": recovery_profile_file,
        "recovery_profiles_file": recovery_profiles_file,
        "runtime_file": state_file.with_name(state_file.stem + "_runtime.json"),
        "metadata_file": state_file.with_name(state_file.stem + "_meta.json"),
        "first_inference_file": state_file.with_name(
            state_file.stem + "_first_inference.json"
        ),
        "token_log_file": resume_state_file.with_name(
            resume_state_file.stem + "_tokenlog.jsonl"
        ),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def atomic_write_pretty_json(path: Path, payload: dict[str, Any]) -> None:
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


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_jsonl_record_to_path(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        append_jsonl_record(handle, payload)


def maybe_record_first_inference_start(
    paths: dict[str, Path],
    args: argparse.Namespace,
    *,
    phase: str,
    parent_start_wall_time: float | None,
) -> float | None:
    if parent_start_wall_time is None:
        return None

    first_inference_path = paths["first_inference_file"]
    existing_payload = load_json_if_exists(first_inference_path)
    if existing_payload is not None:
        recorded_time = existing_payload.get("inference_start_wall_time")
        return None if recorded_time is None else float(recorded_time)

    now = time.time()
    payload = {
        "phase": str(phase),
        "worker_launch_index": int(args.worker_launch_index),
        "inference_start_wall_time": float(now),
        "inference_start_relative_time_seconds": float(now - parent_start_wall_time),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    first_inference_path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    try:
        fd = os.open(
            first_inference_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        existing_payload = load_json_if_exists(first_inference_path)
        if existing_payload is None:
            return None
        recorded_time = existing_payload.get("inference_start_wall_time")
        return None if recorded_time is None else float(recorded_time)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        return float(now)
    finally:
        if fd is not None:
            os.close(fd)


def append_jsonl_record(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def build_engine(
    args: argparse.Namespace,
    *,
    kv_checkpoint_path: Path | None = None,
    kv_buffer_metadata: dict[str, Any] | None = None,
) -> LLMEngine:
    if kv_checkpoint_path is not None and kv_buffer_metadata is not None:
        raise ValueError(
            "kv_checkpoint_path and kv_buffer_metadata are mutually exclusive."
        )

    previous_kv_path = os.environ.get("VLLM_KV_CPU_CHECKPOINT_PATH")
    previous_kv_buffer_metadata = os.environ.get("VLLM_KV_CPU_BUFFER_METADATA")
    if kv_checkpoint_path is None:
        os.environ.pop("VLLM_KV_CPU_CHECKPOINT_PATH", None)
    else:
        os.environ["VLLM_KV_CPU_CHECKPOINT_PATH"] = str(kv_checkpoint_path)

    if kv_buffer_metadata is None:
        os.environ.pop("VLLM_KV_CPU_BUFFER_METADATA", None)
    else:
        os.environ["VLLM_KV_CPU_BUFFER_METADATA"] = json.dumps(
            kv_buffer_metadata, separators=(",", ":")
        )

    try:
        engine_args = EngineArgs(
            model=args.model,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            seed=args.seed,
            disable_log_stats=True,
            enable_prefix_caching=args.enable_prefix_caching,
            enforce_eager=args.enforce_eager,
            max_model_len=args.max_model_len,
        )
        return LLMEngine.from_engine_args(engine_args)
    finally:
        if previous_kv_path is None:
            os.environ.pop("VLLM_KV_CPU_CHECKPOINT_PATH", None)
        else:
            os.environ["VLLM_KV_CPU_CHECKPOINT_PATH"] = previous_kv_path

        if previous_kv_buffer_metadata is None:
            os.environ.pop("VLLM_KV_CPU_BUFFER_METADATA", None)
        else:
            os.environ["VLLM_KV_CPU_BUFFER_METADATA"] = previous_kv_buffer_metadata


def build_sampling_params(max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )


def get_engine_core(engine: LLMEngine):
    client = engine.engine_core
    core = getattr(client, "engine_core", None)
    if core is None:
        raise RuntimeError(
            "Internal EngineCore is not directly accessible. "
            "This script requires in-process engine mode."
        )
    return core


def get_scheduler(engine: LLMEngine):
    core = get_engine_core(engine)
    scheduler = getattr(core, "scheduler", None)
    if scheduler is None:
        raise RuntimeError("Scheduler not found on internal EngineCore.")
    return scheduler


def get_output_processor(engine: LLMEngine):
    output_processor = getattr(engine, "output_processor", None)
    if output_processor is None:
        raise RuntimeError("Output processor not found on LLMEngine.")
    return output_processor


def get_model_runner(engine: LLMEngine):
    model_executor = getattr(engine, "model_executor", None)
    if model_executor is None:
        raise RuntimeError("Model executor not found on LLMEngine.")
    driver_worker = getattr(model_executor, "driver_worker", None)
    if driver_worker is None:
        raise RuntimeError("Driver worker not found on model executor.")
    worker = getattr(driver_worker, "worker", None)
    if worker is None:
        raise RuntimeError("Worker instance not found on driver worker.")
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        raise RuntimeError("Model runner not found on worker.")
    return model_runner


def resolve_internal_request_id(engine: LLMEngine, external_request_id: str) -> str | None:
    output_processor = get_output_processor(engine)
    internal_request_ids = output_processor.external_req_ids.get(
        external_request_id, []
    )
    if not internal_request_ids:
        return None
    if len(internal_request_ids) != 1:
        raise RuntimeError(
            "Expected exactly one internal request id for "
            f"external request {external_request_id!r}, got {internal_request_ids!r}"
        )
    return internal_request_ids[0]


def has_request_entered_decode_stage(
    engine: LLMEngine,
    external_request_id: str,
    *,
    active_requests: dict[str, dict[str, Any]] | None = None,
) -> bool:
    if active_requests is not None and external_request_id not in active_requests:
        return True

    scheduler = get_scheduler(engine)
    internal_request_id = resolve_internal_request_id(engine, external_request_id)
    if internal_request_id is None:
        return False

    scheduler_request = scheduler.requests.get(internal_request_id)
    if scheduler_request is None:
        return False

    return int(scheduler_request.num_computed_tokens) >= int(
        scheduler_request.num_prompt_tokens
    )


def load_request_templates(
    engine: LLMEngine, args: argparse.Namespace
) -> list[dict[str, Any]]:
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"ShareGPT dataset not found: {args.dataset_path}")

    tokenizer = engine.get_tokenizer()
    with open(args.dataset_path, encoding="utf-8") as handle:
        dataset = json.load(handle)

    dataset = [
        entry
        for entry in dataset
        if "conversations" in entry and len(entry["conversations"]) >= 2
    ]
    random.seed(args.seed)
    if not args.disable_shuffle:
        random.shuffle(dataset)

    templates: list[dict[str, Any]] = []
    for entry in dataset:
        prompt = entry["conversations"][0].get("value", "")
        completion = entry["conversations"][1].get("value", "")
        if not prompt or not completion:
            continue

        prompt_token_ids = tokenizer(prompt).input_ids
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        target_output_len = len(completion_token_ids)

        if prompt_len <= 0 or target_output_len <= 0:
            continue
        if prompt_len + target_output_len > args.max_model_len:
            continue

        templates.append(
            {
                "prompt_token_ids": list(prompt_token_ids),
                "prompt_tokens": int(prompt_len),
                "target_output_len": int(target_output_len),
            }
        )
        if len(templates) >= args.total_requests:
            break

    if len(templates) < args.total_requests:
        raise RuntimeError(
            f"Only sampled {len(templates)} valid ShareGPT requests from "
            f"{args.dataset_path}; requested total_requests={args.total_requests}. "
            "Increase --max-model-len or lower --total-requests."
        )

    return templates


def instantiate_request_state(template: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_token_ids": list(template["prompt_token_ids"]),
        "prompt_tokens": int(template["prompt_tokens"]),
        "target_output_len": int(template["target_output_len"]),
        "generated_token_ids": [],
        "finished": False,
        "prefill_accounted": False,
    }


def restore_scheduler_metadata(
    engine: LLMEngine,
    external_request_id: str,
    request_state: dict[str, Any],
) -> tuple[int, int]:
    scheduler = get_scheduler(engine)
    kv_cache_manager = scheduler.kv_cache_manager
    coordinator = kv_cache_manager.coordinator
    managers = coordinator.single_type_managers
    block_pool = kv_cache_manager.block_pool

    internal_request_id = resolve_internal_request_id(engine, external_request_id)
    if internal_request_id is None:
        return 0, 0

    request = scheduler.requests.get(internal_request_id)
    if request is None:
        return 0, 0

    touched_block_ids: set[int] = set()
    target_num_computed = int(request_state.get("num_computed_tokens", 0) or 0)
    max_num_computed = max(0, request.num_tokens - 1)
    target_num_computed = max(0, min(target_num_computed, max_num_computed))

    request.num_computed_tokens = target_num_computed
    request.num_cached_tokens = target_num_computed

    kv_block_ids = request_state.get("kv_block_ids", [])
    restored_blocks = 0
    for group_index, manager in enumerate(managers):
        group_block_ids = (
            kv_block_ids[group_index] if group_index < len(kv_block_ids) else []
        )
        blocks = []
        for block_id in group_block_ids:
            block_id = int(block_id)
            if block_id < 0 or block_id >= len(block_pool.blocks):
                raise ValueError(
                    f"Invalid block id {block_id} for request {external_request_id}."
                )
            blocks.append(block_pool.blocks[block_id])

        manager.req_to_blocks[internal_request_id] = blocks
        num_cached_blocks = min(
            len(blocks),
            target_num_computed // manager.block_size if manager.block_size > 0 else 0,
        )
        manager.num_cached_block[internal_request_id] = num_cached_blocks

        non_null_blocks = []
        for block in blocks:
            if block.is_null:
                continue
            if block.block_id in touched_block_ids:
                continue
            touched_block_ids.add(block.block_id)
            non_null_blocks.append(block)
        if non_null_blocks:
            block_pool.touch(non_null_blocks)

        restored_blocks += len(group_block_ids)

    return 1, restored_blocks


def add_request_to_engine(
    engine: LLMEngine,
    request_id: str,
    request_state: dict[str, Any],
) -> bool:
    remaining_tokens = int(request_state["target_output_len"]) - len(
        request_state["generated_token_ids"]
    )
    if remaining_tokens <= 0:
        return False

    prompt_token_ids = list(request_state["prompt_token_ids"])
    if request_state["generated_token_ids"]:
        prompt_token_ids.extend(request_state["generated_token_ids"])

    engine.add_request(
        request_id,
        {"prompt_token_ids": prompt_token_ids},
        build_sampling_params(remaining_tokens),
    )
    return True


def initialize_manager_files(paths: dict[str, Path]) -> None:
    for path in (
        paths["state_file"],
        paths["report_file"],
        paths["fault_file"],
        paths["resume_state_file"],
        paths["recovery_profile_file"],
        paths["recovery_profiles_file"],
        paths["runtime_file"],
        paths["metadata_file"],
        paths["first_inference_file"],
        paths["token_log_file"],
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()


def write_worker_metadata(
    paths: dict[str, Path],
    args: argparse.Namespace,
    template_count: int,
) -> None:
    payload = {
        "model": args.model,
        "model_label": model_label_from_arg(args.model, args.model_label),
        "dtype": args.dtype,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "dataset_path": str(args.dataset_path.resolve()),
        "total_requests_target": int(args.total_requests),
        "concurrency": int(args.concurrency),
        "template_count": int(template_count),
        "disable_shuffle": bool(args.disable_shuffle),
        "seed": int(args.seed),
        "enable_prefix_caching": bool(args.enable_prefix_caching),
        "max_model_len": int(args.max_model_len),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "max_num_seqs": int(args.max_num_seqs),
        "kv_checkpoint_file": str(args.kv_checkpoint_file),
        "kv_buffer_names": args.kv_buffer_names,
        "kv_buffer_size_bytes": int(args.kv_buffer_size_bytes),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_pretty_json(paths["metadata_file"], payload)


def ensure_worker_metadata(
    paths: dict[str, Path],
    args: argparse.Namespace,
    template_count: int,
) -> None:
    if paths["metadata_file"].exists():
        return
    write_worker_metadata(paths, args, template_count)


_INCREMENTAL_TOKEN_LOG: Any = None


def open_incremental_token_log(paths: dict[str, Path]) -> None:
    global _INCREMENTAL_TOKEN_LOG
    path = paths["token_log_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    _INCREMENTAL_TOKEN_LOG = open(path, "a", encoding="utf-8")


def close_incremental_token_log() -> None:
    global _INCREMENTAL_TOKEN_LOG
    if _INCREMENTAL_TOKEN_LOG is not None:
        try:
            _INCREMENTAL_TOKEN_LOG.flush()
            _INCREMENTAL_TOKEN_LOG.close()
        finally:
            _INCREMENTAL_TOKEN_LOG = None


def _log_incremental_step(new_by_request: dict[str, list[int]], finished_ids: list[str]) -> None:
    log = _INCREMENTAL_TOKEN_LOG
    if log is None or (not new_by_request and not finished_ids):
        return
    log.write(json.dumps({"t": new_by_request, "f": finished_ids}, ensure_ascii=False) + "\n")
    log.flush()


def log_incremental_add(request_id: str, request_state: dict[str, Any]) -> None:
    """Record an add (with prompt) when a new request is enqueued, so the log alone can rebuild recoverable state."""
    log = _INCREMENTAL_TOKEN_LOG
    if log is None:
        return
    log.write(json.dumps({
        "a": request_id,
        "p": list(request_state["prompt_token_ids"]),
        "pt": int(request_state["prompt_tokens"]),
        "tol": int(request_state["target_output_len"]),
    }, ensure_ascii=False) + "\n")
    log.flush()


def log_incremental_blocks(request_id: str, group_block_ids: list[list[int]]) -> None:
    """Record a block manifest (b): the request's full ordered physical block ids per group."""
    log = _INCREMENTAL_TOKEN_LOG
    if log is None:
        return
    log.write(json.dumps({
        "b": request_id,
        "g": [[int(block_id) for block_id in group] for group in group_block_ids],
    }, ensure_ascii=False) + "\n")
    log.flush()


def issue_request_from_template(
    engine: LLMEngine,
    templates: list[dict[str, Any]],
    request_index: int,
    active_requests: dict[str, dict[str, Any]],
) -> None:
    request_id = f"{REQUEST_ID_PREFIX}{request_index}"
    request_state = instantiate_request_state(templates[request_index])
    added = add_request_to_engine(engine, request_id, request_state)
    if not added:
        raise RuntimeError(
            f"Fresh request {request_id} unexpectedly had no remaining tokens to generate."
        )
    active_requests[request_id] = request_state
    log_incremental_add(request_id, request_state)


def top_up_concurrency(
    engine: LLMEngine,
    templates: list[dict[str, Any]],
    args: argparse.Namespace,
    active_requests: dict[str, dict[str, Any]],
    counters: dict[str, int],
) -> None:
    while (
        len(active_requests) < args.concurrency
        and counters["issued_request_count"] < args.total_requests
    ):
        request_index = counters["issued_request_count"]
        issue_request_from_template(engine, templates, request_index, active_requests)
        counters["issued_request_count"] += 1


def process_request_outputs(
    request_outputs: list[Any],
    active_requests: dict[str, dict[str, Any]],
    counters: dict[str, int],
    resumed_request_ids: set[str] | None = None,
) -> None:
    finished_request_ids: list[str] = []
    step_new_tokens: dict[str, list[int]] = {}
    for request_output in request_outputs:
        request_state = active_requests.get(request_output.request_id)
        if request_state is None:
            continue

        new_token_ids = request_output.outputs[0].token_ids
        if new_token_ids and not request_state["prefill_accounted"]:
            request_state["prefill_accounted"] = True
            counters["total_prefilled_prompt_tokens"] += int(
                request_state["prompt_tokens"]
            )
            counters["total_processed_tokens"] += int(request_state["prompt_tokens"])

        if new_token_ids:
            request_state["generated_token_ids"].extend(new_token_ids)
            counters["total_generated_tokens"] += len(new_token_ids)
            counters["total_processed_tokens"] += len(new_token_ids)
            step_new_tokens[request_output.request_id] = list(new_token_ids)
            if resumed_request_ids is not None:
                resumed_request_ids.add(request_output.request_id)

        if request_output.finished:
            request_state["finished"] = True
            finished_request_ids.append(request_output.request_id)

    _log_incremental_step(step_new_tokens, finished_request_ids)

    for request_id in finished_request_ids:
        if request_id in active_requests:
            active_requests.pop(request_id)
            counters["completed_request_count"] += 1


def record_runtime_state(
    paths: dict[str, Path],
    args: argparse.Namespace,
    *,
    counters: dict[str, int],
    phase: str,
    status: str,
    active_requests: dict[str, dict[str, Any]],
    carried_pending_requests: dict[str, dict[str, Any]],
    parent_start_wall_time: float | None,
    fault_requested: bool,
    fault_requested_wall_time: float | None,
    worker_has_started_inference: bool,
) -> None:
    now = time.time()
    payload = {
        "phase": str(phase),
        "status": str(status),
        "issued_request_count": int(counters["issued_request_count"]),
        "completed_request_count": int(counters["completed_request_count"]),
        "remaining_request_count": int(
            args.total_requests - counters["completed_request_count"]
        ),
        "in_engine_request_count": int(len(active_requests)),
        "carried_pending_request_count": int(len(carried_pending_requests)),
        "active_request_count": int(len(active_requests) + len(carried_pending_requests)),
        "total_prefilled_prompt_tokens": int(
            counters["total_prefilled_prompt_tokens"]
        ),
        "total_generated_tokens": int(counters["total_generated_tokens"]),
        "total_processed_tokens": int(counters["total_processed_tokens"]),
        "fault_requested": bool(fault_requested),
        "fault_requested_wall_time": fault_requested_wall_time,
        "worker_launch_index": int(args.worker_launch_index),
        "worker_has_started_inference": bool(worker_has_started_inference),
        "fault_requested_relative_time_seconds": (
            None
            if fault_requested_wall_time is None or parent_start_wall_time is None
            else float(fault_requested_wall_time - parent_start_wall_time)
        ),
        "last_updated_wall_time": float(now),
        "last_updated_relative_time_seconds": (
            None if parent_start_wall_time is None else float(now - parent_start_wall_time)
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_pretty_json(paths["runtime_file"], payload)


def maybe_record_runtime_state(
    paths: dict[str, Path],
    args: argparse.Namespace,
    *,
    counters: dict[str, int],
    phase: str,
    status: str,
    active_requests: dict[str, dict[str, Any]],
    carried_pending_requests: dict[str, dict[str, Any]],
    parent_start_wall_time: float | None,
    fault_requested: bool,
    fault_requested_wall_time: float | None,
    worker_has_started_inference: bool,
    last_runtime_update_time: float,
    force: bool = False,
) -> float:
    now = time.time()
    if not force and (now - last_runtime_update_time) < RUNTIME_UPDATE_INTERVAL_SECONDS:
        return last_runtime_update_time
    record_runtime_state(
        paths,
        args,
        counters=counters,
        phase=phase,
        status=status,
        active_requests=active_requests,
        carried_pending_requests=carried_pending_requests,
        parent_start_wall_time=parent_start_wall_time,
        fault_requested=fault_requested,
        fault_requested_wall_time=fault_requested_wall_time,
        worker_has_started_inference=worker_has_started_inference,
    )
    return now


def load_resume_state_from_log(
    paths: dict[str, Path],
    buffer_plan: dict[str, Any],
) -> tuple[
    dict[str, int],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[list[int]],
]:
    """Fold the normal-path incremental log (add/t/f/b) into counters / recoverable / pending / resume KV buffer metadata."""
    log_path = paths["token_log_file"]
    if not log_path.exists():
        raise FileNotFoundError(f"incremental token log not found: {log_path}")

    group_block_sizes = [int(b) for b in buffer_plan.get("kv_group_block_sizes", [])]
    boundary_block = max(group_block_sizes) if group_block_sizes else 1

    reqs: dict[str, dict[str, Any]] = {}
    blocks_by_req: dict[str, list[list[int]]] = {}
    issued = completed = total_prefilled = total_generated = total_processed = 0
    with open(log_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "a" in rec:
                rid = rec["a"]
                if rid not in reqs:
                    issued += 1
                reqs[rid] = {
                    "prompt_token_ids": list(rec["p"]),
                    "prompt_tokens": int(rec["pt"]),
                    "target_output_len": int(rec["tol"]),
                    "generated_token_ids": [],
                    "finished": False,
                    "prefill_accounted": False,
                }
            if "t" in rec:
                for rid, toks in rec["t"].items():
                    state = reqs.get(rid)
                    if state is None:
                        continue
                    if toks and not state["prefill_accounted"]:
                        state["prefill_accounted"] = True
                        total_prefilled += int(state["prompt_tokens"])
                        total_processed += int(state["prompt_tokens"])
                    state["generated_token_ids"].extend(int(t) for t in toks)
                    total_generated += len(toks)
                    total_processed += len(toks)
            if "f" in rec:
                for rid in rec["f"]:
                    state = reqs.get(rid)
                    if state is not None and not state["finished"]:
                        state["finished"] = True
                        completed += 1
            if "b" in rec:
                blocks_by_req[rec["b"]] = [
                    [int(block_id) for block_id in group] for group in rec["g"]
                ]

    recoverable_requests: dict[str, dict[str, Any]] = {}
    pending_requests: dict[str, dict[str, Any]] = {}
    valid_by_group: dict[int, set[int]] = {}
    for rid, state in reqs.items():
        if state["finished"]:
            continue
        if not state["generated_token_ids"]:
            pending_requests[rid] = state
            continue

        groups = blocks_by_req.get(rid)
        num_computed = int(state["prompt_tokens"]) + len(state["generated_token_ids"])
        manifest_cap: int | None = None
        if groups:
            for group_index, group_blocks in enumerate(groups):
                group_size = (
                    group_block_sizes[group_index]
                    if group_index < len(group_block_sizes)
                    else boundary_block
                )
                supported = len(group_blocks) * max(group_size, 1)
                manifest_cap = (
                    supported if manifest_cap is None else min(manifest_cap, supported)
                )
        capped_num_computed = (
            min(num_computed, manifest_cap) if manifest_cap is not None else 0
        )
        boundary = (
            (capped_num_computed // boundary_block) * boundary_block
            if boundary_block > 0
            else 0
        )
        if not groups or boundary <= 0:
            state["kv_block_ids"] = []
            state["num_computed_tokens"] = 0
            recoverable_requests[rid] = state
            continue

        state["kv_block_ids"] = groups
        state["num_computed_tokens"] = int(boundary)
        recoverable_requests[rid] = state
        for group_index, group_blocks in enumerate(groups):
            group_size = (
                group_block_sizes[group_index]
                if group_index < len(group_block_sizes)
                else boundary_block
            )
            cached_count = boundary // max(group_size, 1)
            valid_by_group.setdefault(group_index, set()).update(
                int(block_id) for block_id in group_blocks[:cached_count]
            )

    max_group = max(valid_by_group.keys(), default=-1)
    valid_block_ids = [
        sorted(valid_by_group.get(group_index, set()))
        for group_index in range(max_group + 1)
    ]

    counters = {
        "issued_request_count": int(issued),
        "completed_request_count": int(completed),
        "total_prefilled_prompt_tokens": int(total_prefilled),
        "total_generated_tokens": int(total_generated),
        "total_processed_tokens": int(total_processed),
    }
    return counters, recoverable_requests, pending_requests, valid_block_ids


def collect_kv_payload_tensors(model_runner: Any) -> tuple[str, dict[str, object]]:
    cross_layers_kv_cache = getattr(model_runner, "cross_layers_kv_cache", None)
    if cross_layers_kv_cache is not None:
        return "cross_layers_kv_cache", {
            "__cross_layers_kv_cache__": cross_layers_kv_cache
        }

    kv_caches = getattr(model_runner, "kv_cache_tensors_by_layer", None)
    if not kv_caches:
        raise RuntimeError("No named KV cache tensors found on model runner.")

    unique_kv_caches = {}
    seen_tensors: set[tuple[int, int, tuple[int, ...], str]] = set()
    for layer_name, tensor in kv_caches.items():
        tensor_key = (
            int(tensor.data_ptr()),
            int(tensor.storage_offset()),
            tuple(int(dim) for dim in tensor.shape),
            str(tensor.dtype),
        )
        if tensor_key in seen_tensors:
            continue
        seen_tensors.add(tensor_key)
        unique_kv_caches[layer_name] = tensor

    return "kv_caches", unique_kv_caches


def _infer_kv_tensor_layout(
    tensor,
    *,
    attn_backend,
    kv_cache_spec: AttentionSpec,
    kernel_block_size: int,
    has_layers_dim: bool = False,
) -> dict[str, object]:
    shape = tuple(int(dim) for dim in tensor.shape)
    test_shape = attn_backend.get_kv_cache_shape(
        num_blocks=1234,
        block_size=kernel_block_size,
        num_kv_heads=kv_cache_spec.num_kv_heads,
        head_size=kv_cache_spec.head_size,
    )

    if has_layers_dim:
        test_shape = (80,) + tuple(test_shape)

    if len(shape) != len(test_shape):
        return {"supports_block_copy": False}

    if test_shape[0] == 1234:
        kv_dim_before_num_blocks = False
    elif len(test_shape) > 1 and test_shape[0] == 2 and test_shape[1] == 1234:
        kv_dim_before_num_blocks = True
    else:
        return {"supports_block_copy": False}

    return {
        "supports_block_copy": True,
        "kv_dim_before_num_blocks": kv_dim_before_num_blocks,
    }


def build_kv_buffer_copy_plan(
    model_runner: Any, payload_tensors: dict[str, object]
) -> tuple[dict[str, dict[str, object]], list[int]]:
    kv_cache_config = getattr(model_runner, "kv_cache_config", None)
    if kv_cache_config is None:
        raise RuntimeError("Model runner has no kv_cache_config.")

    vllm_config = getattr(model_runner, "vllm_config", None)
    config_ctx = (
        set_current_vllm_config(vllm_config)
        if vllm_config is not None
        else nullcontext()
    )
    with config_ctx:
        kernel_block_sizes = model_runner._prepare_kernel_block_sizes(kv_cache_config)
        group_block_sizes: list[int] = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            block_size = getattr(kv_cache_spec, "block_size", None)
            if block_size is not None:
                group_block_sizes.append(int(block_size))

        plan: dict[str, dict[str, object]] = {}
        cross_layers_kv_cache = getattr(model_runner, "cross_layers_kv_cache", None)
        if cross_layers_kv_cache is not None:
            if "__cross_layers_kv_cache__" not in payload_tensors:
                raise RuntimeError("Cross-layer KV cache payload entry is missing.")
            if not model_runner.attn_groups or not model_runner.attn_groups[0]:
                raise RuntimeError(
                    "Could not infer attention backend for cross-layer KV."
                )

            attn_group = model_runner.attn_groups[0][0]
            if not isinstance(attn_group.kv_cache_spec, AttentionSpec):
                plan["__cross_layers_kv_cache__"] = {"supports_block_copy": False}
                return plan, group_block_sizes

            kernel_block_size = int(kernel_block_sizes[0])
            info = _infer_kv_tensor_layout(
                payload_tensors["__cross_layers_kv_cache__"],
                attn_backend=attn_group.backend,
                kv_cache_spec=attn_group.kv_cache_spec,
                kernel_block_size=kernel_block_size,
                has_layers_dim=True,
            )
            info["kv_cache_group_id"] = 0
            info["block_size_factor"] = (
                int(attn_group.kv_cache_spec.block_size) // kernel_block_size
            )
            plan["__cross_layers_kv_cache__"] = info
            return plan, group_block_sizes

        for attn_groups in model_runner.attn_groups:
            for attn_group in attn_groups:
                if not isinstance(attn_group.kv_cache_spec, AttentionSpec):
                    continue
                tensor_name = next(
                    (name for name in attn_group.layer_names if name in payload_tensors),
                    None,
                )
                if tensor_name is None:
                    continue

                kernel_block_size = int(
                    kernel_block_sizes[attn_group.kv_cache_group_id]
                )
                info = _infer_kv_tensor_layout(
                    payload_tensors[tensor_name],
                    attn_backend=attn_group.backend,
                    kv_cache_spec=attn_group.kv_cache_spec,
                    kernel_block_size=kernel_block_size,
                )
                info["kv_cache_group_id"] = int(attn_group.kv_cache_group_id)
                info["block_size_factor"] = (
                    int(attn_group.kv_cache_spec.block_size) // kernel_block_size
                )
                plan[tensor_name] = info

        for tensor_name in payload_tensors:
            plan.setdefault(tensor_name, {"supports_block_copy": False})
        return plan, group_block_sizes


def _expand_block_ids(block_ids: list[int], block_size_factor: int) -> np.ndarray:
    if not block_ids:
        return np.empty((0,), dtype=np.int64)
    if block_size_factor <= 1:
        return np.asarray(block_ids, dtype=np.int64)

    expanded = np.empty((len(block_ids) * block_size_factor,), dtype=np.int64)
    for idx, block_id in enumerate(block_ids):
        start = idx * block_size_factor
        base = int(block_id) * block_size_factor
        expanded[start : start + block_size_factor] = np.arange(
            base, base + block_size_factor, dtype=np.int64
        )
    return expanded


def _swap_blocks_between_tensors(
    source_tensor,
    target_tensor,
    *,
    block_ids: list[int],
    block_size_factor: int,
    kv_dim_before_num_blocks: bool,
    stream=None,
) -> None:
    if not block_ids:
        return

    import torch

    expanded_ids = _expand_block_ids(block_ids, block_size_factor)
    block_pairs = np.empty((expanded_ids.size, 2), dtype=np.int64)
    block_pairs[:, 0] = expanded_ids
    block_pairs[:, 1] = expanded_ids
    block_pairs_tensor = torch.from_numpy(block_pairs)

    def _do_swap() -> None:
        if kv_dim_before_num_blocks:
            ops.swap_blocks(source_tensor[0], target_tensor[0], block_pairs_tensor)
            ops.swap_blocks(source_tensor[1], target_tensor[1], block_pairs_tensor)
        else:
            ops.swap_blocks(source_tensor, target_tensor, block_pairs_tensor)

    if stream is None:
        _do_swap()
        return

    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _do_swap()


class _WorkerKVBufferCopier:
    """Worker-rank local: block-copy between this rank's KV shard and this rank's per-rank SHM
    buffer. Each worker holds its own (built inside the worker by the kv_incremental_* RPCs and
    cached on it). Layout/copy_plan are computed from the worker's own model_runner (ranks are
    symmetric, but copies use local tensors)."""

    def __init__(self, model_runner: Any, buffer_name: str):
        self.buffer_name = buffer_name
        metadata = build_kv_buffer_metadata(model_runner, buffer_name=buffer_name)
        self.kv_kind, self.payload_tensors = collect_kv_payload_tensors(model_runner)
        self.payload_entries = {
            str(entry["name"]): entry for entry in metadata.get("entries", [])
        }
        self.copy_plan, self.group_block_sizes = build_kv_buffer_copy_plan(
            model_runner, self.payload_tensors
        )
        self.shm, _ = prepare_kv_cpu_buffer(buffer_name, pin_memory=True)

        import torch

        self.buffer_tensors: dict[str, torch.Tensor] = {}
        for tensor_name, tensor in self.payload_tensors.items():
            entry = self.payload_entries[tensor_name]
            self.buffer_tensors[tensor_name] = torch.frombuffer(
                self.shm.buf,
                dtype=tensor.dtype,
                count=int(entry["numel"]),
                offset=int(entry["offset"]),
            ).reshape(tensor.shape)

        self.copy_stream = None
        if torch.cuda.is_available():
            first_tensor_name = next(iter(self.payload_tensors))
            self.copy_stream = torch.cuda.Stream(
                device=self.payload_tensors[first_tensor_name].device
            )

    def _copy_group(
        self, group_index: int, block_ids: list[int], *, to_buffer: bool
    ) -> None:
        if not block_ids:
            return
        for tensor_name, gpu_tensor in self.payload_tensors.items():
            plan = self.copy_plan.get(tensor_name, {})
            if not plan.get("supports_block_copy", False):
                continue
            if int(plan.get("kv_cache_group_id", -1)) != group_index:
                continue
            buffer_tensor = self.buffer_tensors[tensor_name]
            source, target = (
                (gpu_tensor, buffer_tensor) if to_buffer else (buffer_tensor, gpu_tensor)
            )
            _swap_blocks_between_tensors(
                source,
                target,
                block_ids=block_ids,
                block_size_factor=int(plan["block_size_factor"]),
                kv_dim_before_num_blocks=bool(plan["kv_dim_before_num_blocks"]),
                stream=self.copy_stream,
            )

    def copy_out(
        self, pending_by_group: list[list[int]], *, do_sync: bool = True
    ) -> None:
        for group_index, block_ids in enumerate(pending_by_group):
            self._copy_group(group_index, [int(b) for b in block_ids], to_buffer=True)
        if do_sync:
            self.sync()

    def copy_in(self, valid_by_group: list[list[int]]) -> None:
        for group_index, block_ids in enumerate(valid_by_group):
            self._copy_group(group_index, [int(b) for b in block_ids], to_buffer=False)
        self.sync()

    def sync(self) -> None:
        if self.copy_stream is not None:
            self.copy_stream.synchronize()


def kv_layout_rpc(worker: Any) -> dict[str, Any]:
    """Compute this rank's KV buffer layout (entries/offsets/total_bytes/group_block_sizes) inside the worker."""
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        raise RuntimeError("model_runner not found on worker")
    return build_kv_buffer_metadata(model_runner)


def _worker_kv_buffer_name(worker: Any, buffer_names: list[str]) -> str:
    rank = int(getattr(worker, "rank", 0))
    if rank < 0 or rank >= len(buffer_names):
        raise IndexError(
            f"rank {rank} out of range for kv buffer names {buffer_names!r}"
        )
    return buffer_names[rank]


def _get_worker_kv_copier(
    worker: Any, buffer_names: list[str]
) -> "_WorkerKVBufferCopier":
    buffer_name = _worker_kv_buffer_name(worker, buffer_names)
    copier = getattr(worker, "_kv_incr_copier", None)
    if copier is None or copier.buffer_name != buffer_name:
        model_runner = getattr(worker, "model_runner", None)
        if model_runner is None:
            raise RuntimeError("model_runner not found on worker")
        copier = _WorkerKVBufferCopier(model_runner, buffer_name)
        worker._kv_incr_copier = copier
    return copier


def kv_incremental_flush_rpc(
    worker: Any,
    pending_by_group: list[list[int]],
    buffer_names: list[str],
    do_sync: bool = True,
) -> dict[str, Any]:
    """Each rank copies pending full blocks from its GPU shard to its SHM buffer (cuda-sync when do_sync)."""
    copier = _get_worker_kv_copier(worker, buffer_names)
    copier.copy_out(pending_by_group, do_sync=bool(do_sync))
    return {"rank": int(getattr(worker, "rank", 0))}


def kv_incremental_load_rpc(
    worker: Any, valid_by_group: list[list[int]], buffer_names: list[str]
) -> dict[str, Any]:
    """At recovery each rank copies valid full blocks from its SHM buffer back to its GPU shard."""
    started_at = time.time()
    copier = _get_worker_kv_copier(worker, buffer_names)
    copier.copy_in(valid_by_group)
    load_seconds = float(time.time() - started_at)
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is not None:
        model_runner.kv_checkpoint_load_seconds = load_seconds
    return {"rank": int(getattr(worker, "rank", 0)), "load_seconds": load_seconds}


class IncrementalKVBufferWriter:
    """Driver-side orchestrator: signatures/manifests/pending are computed on the driver (needs
    the scheduler); the actual GPU<->SHM block copies are pushed to each rank via collective_rpc
    (single-GPU = 1 rank, degenerating to one in-process call). Multi-GPU is the general case."""

    def __init__(
        self,
        engine: LLMEngine,
        buffer_names: list[str],
        group_block_sizes: list[int],
    ):
        self.engine = engine
        self.buffer_names = list(buffer_names)
        self.group_block_sizes = [int(b) for b in group_block_sizes]
        try:
            self.flush_interval = max(1, int(os.environ.get("VLLM_KV_FLUSH_INTERVAL", "1")))
        except (TypeError, ValueError):
            self.flush_interval = 1
        self.step_counter = 0
        self.group_flushed_sig: list[dict[int, tuple[str, int]]] = [
            {} for _ in self.group_block_sizes
        ]
        self.group_window_sig: list[dict[int, tuple[str, int]]] = [
            {} for _ in self.group_block_sizes
        ]
        self.recorded_manifest: dict[str, list[list[int]]] = {}

    def _collect_ready_block_sigs(
        self, active_requests: dict[str, dict[str, Any]]
    ) -> tuple[list[dict[int, tuple[str, int]]], dict[str, list[list[int]]]]:
        """Scan active requests and return per-group ready block signatures and per-request block manifests."""
        scheduler = get_scheduler(self.engine)
        ready_sigs: list[dict[int, tuple[str, int]]] = [
            {} for _ in self.group_block_sizes
        ]
        manifests: dict[str, list[list[int]]] = {}

        for request_id in active_requests:
            internal_request_id = resolve_internal_request_id(self.engine, request_id)
            if internal_request_id is None:
                continue
            request = scheduler.requests.get(internal_request_id)
            if request is None:
                continue
            kv_block_ids = scheduler.kv_cache_manager.get_block_ids(internal_request_id)
            num_computed_tokens = int(request.num_computed_tokens)
            groups_full: list[list[int]] = []
            for group_index, group_block_size in enumerate(self.group_block_sizes):
                if group_index >= len(kv_block_ids):
                    groups_full.append([])
                    continue
                group_ids = [int(block_id) for block_id in kv_block_ids[group_index]]
                full_block_count = min(
                    len(group_ids),
                    num_computed_tokens // max(group_block_size, 1),
                )
                for position in range(full_block_count):
                    ready_sigs[group_index][group_ids[position]] = (
                        internal_request_id,
                        position,
                    )
                groups_full.append(group_ids)
            manifests[request_id] = groups_full

        return ready_sigs, manifests

    def flush_ready_blocks(self, active_requests: dict[str, dict[str, Any]]) -> None:
        """On each commit step (every K steps), flush newly-ready full blocks to SHM and write the block manifest."""
        self.step_counter += 1
        do_commit = (self.step_counter % self.flush_interval) == 0

        if not do_commit:
            return

        ready_sigs, manifests = self._collect_ready_block_sigs(active_requests)

        pending_by_group: list[list[int]] = []
        any_pending = False
        for group_index in range(len(self.group_block_sizes)):
            sig_map = ready_sigs[group_index] if group_index < len(ready_sigs) else {}
            committed = self.group_flushed_sig[group_index]
            window = self.group_window_sig[group_index]
            pending_ids = sorted(
                block_id
                for block_id, sig in sig_map.items()
                if committed.get(block_id) != sig and window.get(block_id) != sig
            )
            pending_by_group.append(pending_ids)
            if pending_ids:
                any_pending = True

        if any_pending or do_commit:
            self.engine.collective_rpc(
                kv_incremental_flush_rpc,
                args=(pending_by_group, self.buffer_names, do_commit),
            )
            for group_index, pending_ids in enumerate(pending_by_group):
                sig_map = (
                    ready_sigs[group_index] if group_index < len(ready_sigs) else {}
                )
                for block_id in pending_ids:
                    self.group_window_sig[group_index][block_id] = sig_map[block_id]

        if do_commit:
            for group_index in range(len(self.group_block_sizes)):
                self.group_flushed_sig[group_index].update(
                    self.group_window_sig[group_index]
                )
                self.group_window_sig[group_index].clear()
            for request_id, groups_full in manifests.items():
                if self.recorded_manifest.get(request_id) != groups_full:
                    log_incremental_blocks(request_id, groups_full)
                    self.recorded_manifest[request_id] = [list(g) for g in groups_full]


def build_kv_buffer_metadata(
    model_runner: Any, *, buffer_name: str | None = None
) -> dict[str, Any]:
    kv_kind, payload_tensors = collect_kv_payload_tensors(model_runner)
    copy_plan, group_block_sizes = build_kv_buffer_copy_plan(
        model_runner, payload_tensors
    )
    entries = []
    total_bytes = 0

    for tensor_name, tensor in payload_tensors.items():
        element_size = int(tensor.element_size())
        tensor_numel = int(tensor.numel())
        tensor_nbytes = tensor_numel * element_size
        alignment = max(KV_BUFFER_ALIGNMENT_BYTES, element_size)
        total_bytes = align_up(total_bytes, alignment)
        entries.append(
            {
                "name": tensor_name,
                "dtype": str(tensor.dtype),
                "shape": [int(dim) for dim in tensor.shape],
                "numel": tensor_numel,
                "nbytes": tensor_nbytes,
                "offset": total_bytes,
            }
        )
        entry_plan = copy_plan.get(tensor_name, {})
        if entry_plan:
            entries[-1].update(entry_plan)
        total_bytes += tensor_nbytes

    metadata = {
        "format": "vllm_kv_cpu_buffer_v1",
        "kind": kv_kind,
        "entries": entries,
        "total_bytes": total_bytes,
        "kv_group_block_sizes": group_block_sizes,
        "supports_incremental_block_copy": all(
            bool(entry.get("supports_block_copy", False)) for entry in entries
        ),
    }
    if buffer_name is not None:
        metadata["buffer_name"] = buffer_name
    return metadata


def checkpoint_backend_from_metadata(
    kv_buffer_metadata: dict[str, Any] | None,
) -> str:
    if kv_buffer_metadata is None:
        return "vllm_kv_cpu_checkpoint"
    if kv_buffer_metadata.get("supports_incremental_block_copy"):
        return "vllm_kv_cpu_buffer_incremental"
    return "vllm_kv_cpu_buffer"


def record_recovery_profile(
    paths: dict[str, Path],
    payload: dict[str, Any],
) -> tuple[int, bool]:
    profile_index = len(load_jsonl_records(paths["recovery_profiles_file"])) + 1
    payload_to_store = dict(payload)
    payload_to_store["fault_profile_index"] = int(profile_index)
    append_jsonl_record_to_path(paths["recovery_profiles_file"], payload_to_store)

    is_first_profile = not paths["recovery_profile_file"].exists()
    if is_first_profile:
        atomic_write_pretty_json(paths["recovery_profile_file"], payload_to_store)
    return profile_index, is_first_profile


def run_probe(args: argparse.Namespace) -> int:
    """Probe phase: build the engine and write out the KV buffer layout plan."""
    if args.kv_buffer_plan_file is None:
        raise ValueError("probe phase requires --kv-buffer-plan-file.")

    engine = build_engine(args)
    layouts = engine.collective_rpc(kv_layout_rpc)
    metadata = layouts[0]
    atomic_write_json(args.kv_buffer_plan_file, metadata)
    print(
        f"[probe] kind={metadata['kind']} entries={len(metadata['entries'])} "
        f"required_bytes={metadata['total_bytes']} "
        f"supports_incremental_block_copy={metadata['supports_incremental_block_copy']} "
        f"plan_file={args.kv_buffer_plan_file}",
        flush=True,
    )
    return 0


def run_generation_phase(args: argparse.Namespace) -> int:
    """Worker phase: run vLLM generation with incremental KV buffer checkpointing to SHM."""
    paths = derived_paths(args)
    parent_start_wall_time = args.parent_start_wall_time
    phase_entry_time = time.time()

    try:
        open_incremental_token_log(paths)
        worker_has_started_inference = False
        kv_buffer_writer: IncrementalKVBufferWriter | None = None

        if args.phase == "creator":
            engine = build_engine(args)
            templates = load_request_templates(engine, args)
            ensure_worker_metadata(paths, args, len(templates))
            counters = {
                "issued_request_count": 0,
                "completed_request_count": 0,
                "total_prefilled_prompt_tokens": 0,
                "total_generated_tokens": 0,
                "total_processed_tokens": 0,
            }
            active_requests: dict[str, dict[str, Any]] = {}
            carried_pending_requests: dict[str, dict[str, Any]] = {}
            top_up_concurrency(engine, templates, args, active_requests, counters)
        else:
            connector_entry_time = phase_entry_time
            state_load_start = time.time()
            if args.kv_buffer_plan_file is None or not args.kv_buffer_plan_file.exists():
                raise RuntimeError(
                    "kv_memory fail-stop recovery requires --kv-buffer-plan-file."
                )
            if not args.kv_buffer_names:
                raise RuntimeError(
                    "kv_memory fail-stop recovery requires --kv-buffer-names."
                )
            buffer_plan = json.loads(
                args.kv_buffer_plan_file.read_text(encoding="utf-8")
            )
            (
                counters,
                recoverable_requests,
                carried_pending_requests,
                valid_block_ids,
            ) = load_resume_state_from_log(paths, buffer_plan)
            state_ready_time = time.time()
            recovery_start_wall_time = float(args.fault_start_time)
            recovery_start_source = "fault_start_time"

            engine_init_start = time.time()
            engine = build_engine(args)
            engine_ready_time = time.time()
            load_results = engine.collective_rpc(
                kv_incremental_load_rpc,
                args=(valid_block_ids, args.kv_buffer_names),
            )
            connector_kv_cache_reload_seconds = float(
                max((float(item.get("load_seconds", 0.0)) for item in load_results), default=0.0)
            )
            request_rebuild_start_time = time.time()

            active_requests = {}
            for request_id, request_state in recoverable_requests.items():
                added = add_request_to_engine(engine, request_id, request_state)
                if not added:
                    counters["completed_request_count"] += 1
                    continue
                active_requests[request_id] = request_state

            replayed_request_ids = set(active_requests)
            resumed_request_ids: set[str] = set()
            decode_ready_request_ids: set[str] = set()
            requests_ready_time = time.time()
            restored_requests = 0
            restored_blocks = 0
            block_claims: dict[int, int] = {}
            for request_id in replayed_request_ids:
                for group in active_requests[request_id].get("kv_block_ids", []):
                    for block_id in group:
                        bid = int(block_id)
                        block_claims[bid] = block_claims.get(bid, 0) + 1
            collided_requests = 0
            for request_id in replayed_request_ids:
                groups = active_requests[request_id].get("kv_block_ids", [])
                if any(block_claims[int(b)] > 1 for group in groups for b in group):
                    active_requests[request_id]["kv_block_ids"] = []
                    active_requests[request_id]["num_computed_tokens"] = 0
                    collided_requests += 1
            if collided_requests:
                print(
                    f"[connector] launch={args.worker_launch_index} block-collision guard: "
                    f"{collided_requests} recoverable request(s) → re-prefill (shared physical block)",
                    flush=True,
                )
            for request_id in replayed_request_ids:
                if not active_requests[request_id].get("kv_block_ids"):
                    continue
                request_count, block_count = restore_scheduler_metadata(
                    engine,
                    request_id,
                    active_requests[request_id],
                )
                restored_requests += request_count
                restored_blocks += block_count
            external_state_ready_time = time.time()
            ready_time = external_state_ready_time if not replayed_request_ids else None

        if args.kv_buffer_names:
            if args.kv_buffer_plan_file is None or not args.kv_buffer_plan_file.exists():
                raise RuntimeError("KV buffer requires --kv-buffer-plan-file.")
            writer_plan = json.loads(
                args.kv_buffer_plan_file.read_text(encoding="utf-8")
            )
            writer_group_block_sizes = [
                int(b) for b in writer_plan.get("kv_group_block_sizes", [])
            ]
            if (
                args.kv_buffer_size_bytes > 0
                and int(writer_plan.get("total_bytes", 0)) > args.kv_buffer_size_bytes
            ):
                raise RuntimeError(
                    "Parent KV buffer is too small: "
                    f"required={writer_plan.get('total_bytes')} "
                    f"available={args.kv_buffer_size_bytes}"
                )
            kv_buffer_writer = IncrementalKVBufferWriter(
                engine, args.kv_buffer_names, writer_group_block_sizes
            )

        if args.phase == "connector":
            last_runtime_update_time = maybe_record_runtime_state(
                paths,
                args,
                counters=counters,
                phase=args.phase,
                status="recovering",
                active_requests=active_requests,
                carried_pending_requests=carried_pending_requests,
                parent_start_wall_time=parent_start_wall_time,
                fault_requested=False,
                fault_requested_wall_time=None,
                worker_has_started_inference=worker_has_started_inference,
                last_runtime_update_time=0.0,
                force=True,
            )

            while ready_time is None:
                if not engine.has_unfinished_requests():
                    raise RuntimeError(
                        "Connector did not emit resumed tokens for all recoverable requests."
                    )

                request_outputs = engine.step()
                force_runtime_update = False
                if not worker_has_started_inference:
                    worker_has_started_inference = True
                    maybe_record_first_inference_start(
                        paths,
                        args,
                        phase=args.phase,
                        parent_start_wall_time=parent_start_wall_time,
                    )
                    force_runtime_update = True
                process_request_outputs(
                    request_outputs,
                    active_requests,
                    counters,
                    resumed_request_ids=resumed_request_ids,
                )
                if kv_buffer_writer is not None:
                    kv_buffer_writer.flush_ready_blocks(active_requests)
                decode_ready_request_ids = {
                    request_id
                    for request_id in replayed_request_ids
                    if has_request_entered_decode_stage(
                        engine,
                        request_id,
                        active_requests=active_requests,
                    )
                }
                if replayed_request_ids.issubset(decode_ready_request_ids):
                    ready_time = time.time()
                last_runtime_update_time = maybe_record_runtime_state(
                    paths,
                    args,
                    counters=counters,
                    phase=args.phase,
                    status="recovering",
                    active_requests=active_requests,
                    carried_pending_requests=carried_pending_requests,
                    parent_start_wall_time=parent_start_wall_time,
                    fault_requested=False,
                    fault_requested_wall_time=None,
                    worker_has_started_inference=worker_has_started_inference,
                    last_runtime_update_time=last_runtime_update_time,
                    force=force_runtime_update,
                )

            if args.fault_start_time is not None:
                recovery_profile = {
                    "profile_type": "fault_recovery",
                    "fault_start_time": float(args.fault_start_time),
                    "recovery_start_wall_time": float(recovery_start_wall_time),
                    "recovery_start_source": recovery_start_source,
                    "recovery_excludes_fault_save": True,
                    "excluded_fault_save_window_seconds": float(
                        recovery_start_wall_time - args.fault_start_time
                    ),
                    "fault_to_recovery_ready_including_save_seconds": float(
                        ready_time - args.fault_start_time
                    ),
                    "fault_to_recovery_ready_seconds": float(
                        ready_time - recovery_start_wall_time
                    ),
                    "connector_python_startup_seconds": float(
                        connector_entry_time - recovery_start_wall_time
                    ),
                    "connector_state_load_seconds": float(
                        state_ready_time - state_load_start
                    ),
                    "connector_vllm_engine_startup_seconds": float(
                        engine_ready_time - connector_entry_time
                    ),
                    "connector_vllm_engine_init_seconds": float(
                        engine_ready_time - engine_init_start
                    ),
                    "connector_kv_cache_reload_seconds": float(
                        connector_kv_cache_reload_seconds
                    ),
                    "connector_request_rebuild_seconds": float(
                        requests_ready_time - request_rebuild_start_time
                    ),
                    "connector_external_state_restore_seconds": float(
                        external_state_ready_time - requests_ready_time
                    ),
                    "connector_kv_metadata_restore_seconds": float(
                        external_state_ready_time - requests_ready_time
                    ),
                    "connector_resume_to_ready_seconds": float(
                        ready_time - external_state_ready_time
                    ),
                    "replayed_request_count": int(len(replayed_request_ids)),
                    "requests_resumed_to_ready": int(len(resumed_request_ids)),
                    "requests_decode_ready_to_ready": int(len(decode_ready_request_ids)),
                    "restored_requests": int(restored_requests),
                    "restored_blocks": int(restored_blocks),
                    "pending_request_count_before_ready": int(
                        len(carried_pending_requests)
                    ),
                    "recoverable_request_count_before_ready": int(
                        len(replayed_request_ids)
                    ),
                    "checkpoint_backend": checkpoint_backend_from_metadata(
                        buffer_plan
                    ),
                    "ready_marker": (
                        "all recoverable requests reached decode stage before pending requests were activated"
                    ),
                    "worker_launch_index": int(args.worker_launch_index),
                    "phase": str(args.phase),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                profile_index, is_first_profile = record_recovery_profile(
                    paths, recovery_profile
                )
                if is_first_profile:
                    print(
                        f"[connector] first recovery profiled "
                        f"index={profile_index} "
                        f"fault_to_ready={recovery_profile['fault_to_recovery_ready_seconds']:.6f}s "
                        f"python={recovery_profile['connector_python_startup_seconds']:.6f}s "
                        f"engine={recovery_profile['connector_vllm_engine_startup_seconds']:.6f}s "
                        f"resume_to_ready={recovery_profile['connector_resume_to_ready_seconds']:.6f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"[connector] recovery profiled "
                        f"index={profile_index} "
                        f"fault_to_ready={recovery_profile['fault_to_recovery_ready_seconds']:.6f}s "
                        f"python={recovery_profile['connector_python_startup_seconds']:.6f}s "
                        f"engine={recovery_profile['connector_vllm_engine_startup_seconds']:.6f}s "
                        f"resume_to_ready={recovery_profile['connector_resume_to_ready_seconds']:.6f}s",
                        flush=True,
                    )

            templates = load_request_templates(engine, args)
            ensure_worker_metadata(paths, args, len(templates))
            for request_id, request_state in carried_pending_requests.items():
                added = add_request_to_engine(engine, request_id, request_state)
                if not added:
                    counters["completed_request_count"] += 1
                    continue
                active_requests[request_id] = request_state
            carried_pending_requests = {}
            top_up_concurrency(engine, templates, args, active_requests, counters)

        last_runtime_update_time = maybe_record_runtime_state(
            paths,
            args,
            counters=counters,
            phase=args.phase,
            status="running",
            active_requests=active_requests,
            carried_pending_requests=carried_pending_requests,
            parent_start_wall_time=parent_start_wall_time,
            fault_requested=False,
            fault_requested_wall_time=None,
            worker_has_started_inference=worker_has_started_inference,
            last_runtime_update_time=0.0,
            force=True,
        )

        last_logged_completed = int(counters["completed_request_count"])

        while counters["completed_request_count"] < args.total_requests:
            if not engine.has_unfinished_requests():
                top_up_concurrency(engine, templates, args, active_requests, counters)
                if not engine.has_unfinished_requests():
                    break

            request_outputs = engine.step()
            force_runtime_update = False
            if not worker_has_started_inference:
                worker_has_started_inference = True
                maybe_record_first_inference_start(
                    paths,
                    args,
                    phase=args.phase,
                    parent_start_wall_time=parent_start_wall_time,
                )
                force_runtime_update = True
            process_request_outputs(request_outputs, active_requests, counters)
            top_up_concurrency(engine, templates, args, active_requests, counters)
            if kv_buffer_writer is not None:
                kv_buffer_writer.flush_ready_blocks(active_requests)

            if (
                args.log_interval_requests > 0
                and counters["completed_request_count"]
                >= last_logged_completed + args.log_interval_requests
            ):
                print(
                    f"[{args.phase}] launch={args.worker_launch_index} "
                    f"completed={counters['completed_request_count']}/{args.total_requests} "
                    f"issued={counters['issued_request_count']} active={len(active_requests)}",
                    flush=True,
                )
                last_logged_completed = counters["completed_request_count"]

            last_runtime_update_time = maybe_record_runtime_state(
                paths,
                args,
                counters=counters,
                phase=args.phase,
                status="running",
                active_requests=active_requests,
                carried_pending_requests=carried_pending_requests,
                parent_start_wall_time=parent_start_wall_time,
                fault_requested=False,
                fault_requested_wall_time=None,
                worker_has_started_inference=worker_has_started_inference,
                last_runtime_update_time=last_runtime_update_time,
                force=force_runtime_update,
            )

        if counters["completed_request_count"] != args.total_requests:
            raise RuntimeError(
                "Worker exited before all requests completed. "
                f"completed={counters['completed_request_count']} target={args.total_requests}"
            )

        maybe_record_runtime_state(
            paths,
            args,
            counters=counters,
            phase=args.phase,
            status="completed",
            active_requests=active_requests,
            carried_pending_requests=carried_pending_requests,
            parent_start_wall_time=parent_start_wall_time,
            fault_requested=False,
            fault_requested_wall_time=None,
            worker_has_started_inference=worker_has_started_inference,
            last_runtime_update_time=last_runtime_update_time,
            force=True,
        )
        print(
            f"[{args.phase}] launch={args.worker_launch_index} "
            f"completed={counters['completed_request_count']}/{args.total_requests}",
            flush=True,
        )
        return 0
    finally:
        close_incremental_token_log()


def build_phase_cmd(
    args: argparse.Namespace,
    phase: str,
    *,
    worker_launch_index: int | None = None,
    parent_start_wall_time: float | None = None,
    fault_start_time: float | None = None,
    kv_buffer_names: list[str] | None = None,
    kv_buffer_size_bytes: int | None = None,
    kv_buffer_plan_file: Path | None = None,
) -> list[str]:
    script_path = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(script_path),
        "--phase",
        phase,
        "--model",
        str(args.model),
        "--dtype",
        str(args.dtype),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(args.max_model_len),
        "--trust-remote-code",
        str(args.trust_remote_code).lower(),
        "--seed",
        str(args.seed),
        "--enable-prefix-caching",
        str(args.enable_prefix_caching).lower(),
        "--dataset-path",
        str(args.dataset_path),
        "--total-requests",
        str(args.total_requests),
        "--concurrency",
        str(args.concurrency),
        "--disable-shuffle",
        str(args.disable_shuffle).lower(),
        "--fault-frequency",
        str(args.fault_frequency),
        "--fault-check-interval-seconds",
        str(args.fault_check_interval_seconds),
        "--max-unexpected-worker-exits",
        str(args.max_unexpected_worker_exits),
        "--log-interval-requests",
        str(args.log_interval_requests),
        "--state-file",
        str(args.state_file),
        "--report-file",
        str(args.report_file),
        "--fault-file",
        str(args.fault_file),
        "--resume-state-file",
        str(args.resume_state_file),
        "--recovery-profile-file",
        str(args.recovery_profile_file),
        "--recovery-profiles-file",
        str(args.recovery_profiles_file),
        "--kv-checkpoint-file",
        str(args.kv_checkpoint_file),
    ]
    if args.model_label:
        cmd.extend(["--model-label", str(args.model_label)])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if worker_launch_index is not None:
        cmd.extend(["--worker-launch-index", str(worker_launch_index)])
    if parent_start_wall_time is not None:
        cmd.extend(["--parent-start-wall-time", f"{parent_start_wall_time:.9f}"])
    if fault_start_time is not None:
        cmd.extend(["--fault-start-time", f"{fault_start_time:.9f}"])
    if kv_buffer_names:
        cmd.extend(["--kv-buffer-names", ",".join(kv_buffer_names)])
    if kv_buffer_size_bytes is not None and kv_buffer_size_bytes > 0:
        cmd.extend(["--kv-buffer-size-bytes", str(kv_buffer_size_bytes)])
    if kv_buffer_plan_file is not None:
        cmd.extend(["--kv-buffer-plan-file", str(kv_buffer_plan_file)])
    return cmd


def kill_worker_group(process: subprocess.Popen) -> None:
    """SIGKILL the whole worker session group (kills MultiprocExecutor subprocesses too)."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            pass
    try:
        process.wait(timeout=30)
    except Exception:
        pass


def run_manager(args: argparse.Namespace) -> int:
    """Manager phase: inject random faults and relaunch the worker until the run completes."""
    paths = derived_paths(args)
    initialize_manager_files(paths)
    cleanup_kv_checkpoint_files(args.kv_checkpoint_file, args.tensor_parallel_size)

    rng = random.Random(args.seed)
    parent_start_wall_time = time.time()
    fault_injection_count = 0
    worker_launch_count = 0
    worker_restart_count = 0
    unexpected_worker_exit_count = 0
    fault_check_draw_count = 0
    first_generation_worker_launch_wall_time: float | None = None
    current_phase = "creator"
    fault_pending = False
    pending_fault_start_time: float | None = None
    first_recovery_profile_pending = False
    first_recovery_fault_start_time: float | None = None
    next_fault_check_wall_time: float | None = None
    kv_buffers: list[Any] = []
    kv_buffer_names: list[str] = []
    kv_buffer_size_bytes = 0
    kv_buffer_plan: dict[str, Any] | None = None
    kv_buffer_plan_file: Path | None = None

    try:
        with paths["fault_file"].open("a", encoding="utf-8") as fault_handle:
            append_jsonl_record(
                fault_handle,
                {
                    "event": "manager_start",
                    "wall_time": float(parent_start_wall_time),
                    "fault_frequency": float(args.fault_frequency),
                    "fault_check_interval_seconds": float(
                        args.fault_check_interval_seconds
                    ),
                    "total_requests": int(args.total_requests),
                    "concurrency": int(args.concurrency),
                },
            )

            with NamedTemporaryFile(
                prefix="e2e_vllm_kv_buffer_plan_", suffix=".json", delete=False
            ) as plan_file:
                kv_buffer_plan_file = Path(plan_file.name)

            probe_cmd = build_phase_cmd(
                args, "probe", kv_buffer_plan_file=kv_buffer_plan_file
            )
            append_jsonl_record(
                fault_handle,
                {
                    "event": "probe_launch",
                    "wall_time": float(time.time()),
                    "plan_file": str(kv_buffer_plan_file),
                },
            )
            print("[manager] launching probe", flush=True)
            probe_rc = subprocess.call(probe_cmd)
            if probe_rc != 0:
                return probe_rc

            if kv_buffer_plan_file is None or not kv_buffer_plan_file.exists():
                raise FileNotFoundError("KV buffer plan file was not created.")
            kv_buffer_plan = json.loads(
                kv_buffer_plan_file.read_text(encoding="utf-8")
            )
            kv_buffer_size_bytes = int(kv_buffer_plan.get("total_bytes", 0) or 0)
            if kv_buffer_size_bytes <= 0:
                raise RuntimeError(
                    f"Invalid KV buffer size from probe: {kv_buffer_size_bytes}"
                )
            tp_ranks = max(1, int(args.tensor_parallel_size))
            kv_buffers = [
                create_kv_cpu_buffer(kv_buffer_size_bytes) for _ in range(tp_ranks)
            ]
            kv_buffer_names = [str(buf.name) for buf in kv_buffers]

            append_jsonl_record(
                fault_handle,
                {
                    "event": "probe_ready",
                    "wall_time": float(time.time()),
                    "kv_buffer_names": kv_buffer_names,
                    "kv_buffer_size_bytes_per_rank": int(kv_buffer_size_bytes),
                    "tensor_parallel_size": int(tp_ranks),
                    "supports_incremental_block_copy": bool(
                        kv_buffer_plan.get("supports_incremental_block_copy", False)
                    ),
                },
            )

            def launch_worker(
                phase: str, fault_start_time: float | None = None
            ) -> subprocess.Popen[str]:
                nonlocal next_fault_check_wall_time
                nonlocal worker_launch_count
                nonlocal first_generation_worker_launch_wall_time
                launch_wall_time = time.time()
                effective_fault_start_time = launch_wall_time if phase == "connector" else fault_start_time

                cmd = build_phase_cmd(
                    args,
                    phase,
                    worker_launch_index=worker_launch_count,
                    parent_start_wall_time=parent_start_wall_time,
                    fault_start_time=effective_fault_start_time,
                    kv_buffer_names=kv_buffer_names,
                    kv_buffer_size_bytes=kv_buffer_size_bytes,
                    kv_buffer_plan_file=kv_buffer_plan_file,
                )
                if first_generation_worker_launch_wall_time is None:
                    first_generation_worker_launch_wall_time = launch_wall_time
                append_jsonl_record(
                    fault_handle,
                    {
                        "event": "worker_launch",
                        "phase": phase,
                        "worker_launch_index": int(worker_launch_count),
                        "fault_start_time": effective_fault_start_time,
                        "wall_time": float(launch_wall_time),
                    },
                )
                print(
                    f"[manager] launching {phase} index={worker_launch_count}",
                    flush=True,
                )
                process = subprocess.Popen(cmd, start_new_session=True)
                worker_launch_count += 1
                next_fault_check_wall_time = None
                return process

            process = launch_worker(current_phase)

            while True:
                rc = process.poll()
                if rc is not None:
                    runtime_state = load_json_if_exists(paths["runtime_file"]) or {}
                    completed_requests = int(
                        runtime_state.get("completed_request_count", 0) or 0
                    )

                    if fault_pending:
                        append_jsonl_record(
                            fault_handle,
                            {
                                "event": "fault_killed",
                                "phase": current_phase,
                                "return_code": int(rc),
                                "wall_time": float(time.time()),
                                "completed_requests": int(completed_requests),
                            },
                        )
                        current_phase = "connector"
                        fault_pending = False
                        if fault_injection_count == 1:
                            first_recovery_profile_pending = True
                            first_recovery_fault_start_time = pending_fault_start_time
                        worker_restart_count += 1
                        process = launch_worker(current_phase, pending_fault_start_time)
                        pending_fault_start_time = None
                        continue

                    if rc == 0 and completed_requests >= args.total_requests:
                        append_jsonl_record(
                            fault_handle,
                            {
                                "event": "worker_completed",
                                "phase": current_phase,
                                "wall_time": float(time.time()),
                                "completed_requests": int(completed_requests),
                            },
                        )
                        break

                    unexpected_worker_exit_count += 1
                    append_jsonl_record(
                        fault_handle,
                        {
                            "event": "unexpected_worker_exit",
                            "phase": current_phase,
                            "return_code": int(rc),
                            "wall_time": float(time.time()),
                            "relative_time_seconds": float(
                                time.time() - parent_start_wall_time
                            ),
                            "completed_requests": int(completed_requests),
                            "unexpected_worker_exit_count": int(
                                unexpected_worker_exit_count
                            ),
                        },
                    )
                    if unexpected_worker_exit_count > args.max_unexpected_worker_exits:
                        raise RuntimeError(
                            "Worker exited unexpectedly too many times. "
                            f"limit={args.max_unexpected_worker_exits}"
                        )
                    current_phase = (
                        "connector"
                        if paths["token_log_file"].exists()
                        else current_phase
                    )
                    worker_restart_count += 1
                    process = launch_worker(current_phase, pending_fault_start_time)
                    pending_fault_start_time = None
                    continue

                time.sleep(MANAGER_POLL_INTERVAL_SECONDS)
                if fault_pending:
                    continue
                if first_recovery_profile_pending:
                    if paths["recovery_profile_file"].exists():
                        first_recovery_profile_pending = False
                        first_recovery_fault_start_time = None
                        append_jsonl_record(
                            fault_handle,
                            {
                                "event": "first_recovery_profile_captured",
                                "wall_time": float(time.time()),
                            },
                        )
                    else:
                        continue

                runtime_state = load_json_if_exists(paths["runtime_file"]) or {}
                current_worker_launch_index = worker_launch_count - 1
                runtime_worker_launch_index_raw = runtime_state.get("worker_launch_index")
                runtime_worker_launch_index = (
                    -1
                    if runtime_worker_launch_index_raw is None
                    else int(runtime_worker_launch_index_raw)
                )
                runtime_status = str(runtime_state.get("status", ""))
                runtime_worker_has_started_inference = bool(
                    runtime_state.get("worker_has_started_inference", False)
                )
                if runtime_worker_launch_index != current_worker_launch_index:
                    continue
                if not runtime_worker_has_started_inference:
                    continue
                if runtime_status == "recovering":
                    continue

                now = time.time()
                if next_fault_check_wall_time is None:
                    next_fault_check_wall_time = now + args.fault_check_interval_seconds
                    append_jsonl_record(
                        fault_handle,
                        {
                            "event": "fault_checks_armed",
                            "phase": current_phase,
                            "wall_time": float(now),
                            "worker_launch_index": int(current_worker_launch_index),
                            "first_fault_check_wall_time": float(
                                next_fault_check_wall_time
                            ),
                        },
                    )
                    continue
                if now < next_fault_check_wall_time:
                    continue

                observed_completed = int(
                    runtime_state.get("completed_request_count", 0) or 0
                )
                if observed_completed >= args.total_requests:
                    continue

                fault_check_draw_count += 1
                draw = rng.uniform(0.0, 100.0)
                next_fault_check_wall_time = now + args.fault_check_interval_seconds
                if draw >= args.fault_frequency:
                    continue

                fault_pending = True
                pending_fault_start_time = now
                fault_injection_count += 1
                append_jsonl_record(
                    fault_handle,
                    {
                        "event": "fault_requested",
                        "phase": current_phase,
                        "wall_time": float(pending_fault_start_time),
                        "relative_time_seconds": float(
                            pending_fault_start_time - parent_start_wall_time
                        ),
                        "random_draw": float(draw),
                        "threshold": float(args.fault_frequency),
                        "observed_completed_before_kill": int(observed_completed),
                        "worker_launch_index": int(current_worker_launch_index),
                        "pid": int(process.pid),
                    },
                )
                kill_worker_group(process)

        end_wall_time = time.time()
        metadata = load_json_if_exists(paths["metadata_file"]) or {}
        runtime_state = load_json_if_exists(paths["runtime_file"]) or {}
        recovery_profiles = load_jsonl_records(paths["recovery_profiles_file"])
        recovery_profile = load_json_if_exists(paths["recovery_profile_file"])
        if recovery_profile is None and recovery_profiles:
            recovery_profile = recovery_profiles[0]
        recovery_profile = recovery_profile or {}
        last_resume_state = load_json_if_exists(paths["resume_state_file"]) or {}
        first_inference_state = load_json_if_exists(paths["first_inference_file"]) or {}

        total_elapsed_start_wall_time = (
            first_generation_worker_launch_wall_time
            if first_generation_worker_launch_wall_time is not None
            else parent_start_wall_time
        )
        total_elapsed_start_marker = (
            "first_generation_worker_launch_excluding_probe"
            if first_generation_worker_launch_wall_time is not None
            else "manager_start"
        )
        first_inference_start_wall_time = first_inference_state.get(
            "inference_start_wall_time"
        )
        if first_inference_start_wall_time is None:
            first_inference_start_relative_time_seconds = None
        else:
            first_inference_start_wall_time = float(first_inference_start_wall_time)
            first_inference_start_relative_time_seconds = float(
                first_inference_start_wall_time - parent_start_wall_time
            )

        total_elapsed_seconds = end_wall_time - total_elapsed_start_wall_time

        completed_requests = int(runtime_state.get("completed_request_count", 0) or 0)
        requests_per_second = (
            float(completed_requests / total_elapsed_seconds)
            if total_elapsed_seconds > 0.0
            else 0.0
        )
        average_ms_per_request = (
            float(total_elapsed_seconds * 1000.0 / completed_requests)
            if completed_requests > 0
            else 0.0
        )
        checkpoint_backend = checkpoint_backend_from_metadata(kv_buffer_plan)
        _complete_profiles = [
            profile
            for profile in recovery_profiles
            if profile.get("fault_to_recovery_ready_seconds") is not None
            and profile.get("connector_python_startup_seconds") is not None
            and profile.get("connector_vllm_engine_startup_seconds") is not None
        ]

        def _avg(values: list[float]) -> float | None:
            return float(sum(values) / len(values)) if values else None

        _avg_total = _avg(
            [float(p["fault_to_recovery_ready_seconds"]) for p in _complete_profiles]
        )
        _avg_py = _avg(
            [float(p["connector_python_startup_seconds"]) for p in _complete_profiles]
        )
        _avg_eng = _avg(
            [float(p["connector_vllm_engine_startup_seconds"]) for p in _complete_profiles]
        )
        _avg_other = _avg(
            [
                max(
                    0.0,
                    float(p["fault_to_recovery_ready_seconds"])
                    - float(p["connector_python_startup_seconds"])
                    - float(p["connector_vllm_engine_startup_seconds"]),
                )
                for p in _complete_profiles
            ]
        )

        payload = {
            "experiment": "e2e_vllm_kv_memory_faults",
            "checkpoint_backend": checkpoint_backend,
            "model": args.model,
            "model_label": metadata.get(
                "model_label", model_label_from_arg(args.model, args.model_label)
            ),
            "dtype": args.dtype,
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "total_requests_target": int(args.total_requests),
            "completed_requests": int(completed_requests),
            "concurrency": int(args.concurrency),
            "seed": int(args.seed),
            "enable_prefix_caching": bool(args.enable_prefix_caching),
            "max_model_len": int(args.max_model_len),
            "max_num_batched_tokens": int(args.max_num_batched_tokens),
            "max_num_seqs": int(args.max_num_seqs),
            "fault_frequency": float(args.fault_frequency),
            "fault_injection_count": int(fault_injection_count),
            "worker_restart_count": int(worker_restart_count),
            "average_ms_per_request": float(average_ms_per_request),
            "recovery_sample_count": int(len(_complete_profiles)),
            "fault_to_recovery_ready_seconds": _avg_total,
            "connector_python_startup_seconds": _avg_py,
            "connector_vllm_engine_startup_seconds": _avg_eng,
            "other_seconds": _avg_other,
        }
        atomic_write_pretty_json(paths["state_file"], payload)
        atomic_write_pretty_json(paths["report_file"], payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    finally:
        for buf in kv_buffers:
            try:
                buf.unlink()
            except FileNotFoundError:
                pass
        if kv_buffer_plan_file is not None and kv_buffer_plan_file.exists():
            kv_buffer_plan_file.unlink()
        cleanup_kv_checkpoint_files(args.kv_checkpoint_file, args.tensor_parallel_size)


def main() -> int:
    """Entry point: dispatch to the manager, probe, or generation phase based on parsed args."""
    args = parse_args()
    args = normalize_run_paths(args)
    validate_args(args)
    print(
        f"[main] phase={args.phase} pid={os.getpid()} model={args.model} "
        f"enable_prefix_caching={args.enable_prefix_caching} "
        f"kv_checkpoint_file={args.kv_checkpoint_file} "
        f"kv_buffer_names={args.kv_buffer_names}",
        flush=True,
    )


    try:
        if args.phase == "manager":
            return run_manager(args)
        if args.phase == "probe":
            return run_probe(args)
        return run_generation_phase(args)
    finally:
        release_all_kv_cpu_buffers()


if __name__ == "__main__":
    raise SystemExit(main())
