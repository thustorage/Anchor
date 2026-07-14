#!/usr/bin/env python3
"""End-to-end vLLM replay-based recovery benchmark with random fault injection (single + multi GPU)."""
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_IPC_TOOL"] = "0"
os.environ.pop("VLLM_DEBUG_SCHEDULE_PREFILL_TOKENS", None)
os.environ.pop("VLLM_KV_CPU_CHECKPOINT_PATH", None)
os.environ.pop("VLLM_KV_CPU_BUFFER_METADATA", None)

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.sampling_params import RequestOutputKind


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_ROOT = SCRIPT_DIR / "report1" / "e2e_vllm_replay"
DEFAULT_SHAREGPT_DATASET_PATH = Path(
    "/root/mhy/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json"
)
REQUEST_ID_PREFIX = "sharegpt_request_"
RUNTIME_UPDATE_INTERVAL_SECONDS = 1.0
MANAGER_POLL_INTERVAL_SECONDS = 0.2


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end vLLM replay benchmark with a manager process that injects "
            "random faults and a lab1-style creator/connector handoff."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("manager", "creator", "connector"),
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
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_SHAREGPT_DATASET_PATH)
    parser.add_argument("--total-requests", type=int, default=10000)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--disable-shuffle", type=str2bool, default=False)
    parser.add_argument(
        "--fault-frequency",
        type=float,
        default=10.0,
        help="Every fault check, inject a fault when a random draw in [0, 100) is below this value.",
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
        help="Creator/connector handoff JSON written only when a fault is requested.",
    )
    parser.add_argument(
        "--recovery-profile-file",
        type=Path,
        default=None,
        help="Optional JSON file storing the first fault's recovery timing breakdown.",
    )
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
        args.recovery_profile_file = DEFAULT_REPORT_ROOT / f"{label}_first_recovery.json"
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
    return {
        "state_file": state_file,
        "report_file": report_file,
        "fault_file": fault_file,
        "resume_state_file": resume_state_file,
        "token_log_file": resume_state_file.with_name(resume_state_file.stem + "_tokenlog.jsonl"),
        "recovery_profile_file": recovery_profile_file,
        "recovery_profiles_file": recovery_profile_file.with_name(
            recovery_profile_file.stem + "_all.jsonl"
        ),
        "runtime_file": state_file.with_name(state_file.stem + "_runtime.json"),
        "metadata_file": state_file.with_name(state_file.stem + "_meta.json"),
        "first_inference_file": state_file.with_name(
            state_file.stem + "_first_inference.json"
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


def build_engine(args: argparse.Namespace) -> LLMEngine:
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
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
    )
    return LLMEngine.from_engine_args(engine_args)


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


def resolve_internal_request_id(engine: LLMEngine, external_request_id: str) -> str | None:
    output_processor = get_output_processor(engine)
    internal_request_ids = output_processor.external_req_ids.get(external_request_id, [])
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


def load_request_templates(engine: LLMEngine, args: argparse.Namespace) -> list[dict[str, Any]]:
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


def add_request_to_engine(
    engine: LLMEngine,
    request_id: str,
    request_state: dict[str, Any],
) -> bool:
    remaining_tokens = int(request_state["target_output_len"]) - len(request_state["generated_token_ids"])
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
        paths["token_log_file"],
        paths["recovery_profile_file"],
        paths["recovery_profiles_file"],
        paths["runtime_file"],
        paths["metadata_file"],
        paths["first_inference_file"],
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
        "max_model_len": int(args.max_model_len),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "max_num_seqs": int(args.max_num_seqs),
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


def load_resume_state_from_log(
    paths: dict[str, Path],
) -> tuple[dict[str, int], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Fold the normal-path incremental log into (counters, recoverable, pending, state)."""
    log_path = paths["token_log_file"]
    if not log_path.exists():
        raise FileNotFoundError(f"incremental token log not found: {log_path}")

    reqs: dict[str, dict[str, Any]] = {}
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

    recoverable_requests = {
        rid: s for rid, s in reqs.items()
        if not s["finished"] and s["generated_token_ids"]
    }
    pending_requests = {
        rid: s for rid, s in reqs.items()
        if not s["finished"] and not s["generated_token_ids"]
    }
    counters = {
        "issued_request_count": int(issued),
        "completed_request_count": int(completed),
        "total_prefilled_prompt_tokens": int(total_prefilled),
        "total_generated_tokens": int(total_generated),
        "total_processed_tokens": int(total_processed),
    }
    state = {"recoverable_requests": recoverable_requests, "pending_requests": pending_requests}
    return counters, recoverable_requests, pending_requests, state


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
            counters["total_prefilled_prompt_tokens"] += int(request_state["prompt_tokens"])
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
        "remaining_request_count": int(args.total_requests - counters["completed_request_count"]),
        "in_engine_request_count": int(len(active_requests)),
        "carried_pending_request_count": int(len(carried_pending_requests)),
        "active_request_count": int(len(active_requests) + len(carried_pending_requests)),
        "total_prefilled_prompt_tokens": int(counters["total_prefilled_prompt_tokens"]),
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


def maybe_write_first_recovery_profile(
    paths: dict[str, Path],
    payload: dict[str, Any],
) -> bool:
    if paths["recovery_profile_file"].exists():
        return False
    atomic_write_pretty_json(paths["recovery_profile_file"], payload)
    return True


def run_generation_phase(args: argparse.Namespace) -> int:
    """Worker phase: run vLLM generation and log incremental state for replay recovery."""
    paths = derived_paths(args)
    parent_start_wall_time = args.parent_start_wall_time
    phase_entry_time = time.time()

    open_incremental_token_log(paths)
    try:
        worker_has_started_inference = False
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
            engine_init_start = time.time()
            engine = build_engine(args)
            engine_ready_time = time.time()
            counters, recoverable_requests, carried_pending_requests, _ = load_resume_state_from_log(paths)
            state_ready_time = time.time()
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
            ready_time = requests_ready_time if not replayed_request_ids else None
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
                    "fault_to_recovery_ready_seconds": float(ready_time - args.fault_start_time),
                    "connector_python_startup_seconds": float(
                        connector_entry_time - args.fault_start_time
                    ),
                    "connector_vllm_engine_startup_seconds": float(
                        engine_ready_time - connector_entry_time
                    ),
                    "connector_resume_to_ready_seconds": float(
                        ready_time - engine_ready_time
                    ),
                    "connector_resume_state_load_seconds": float(
                        state_ready_time - engine_ready_time
                    ),
                    "connector_vllm_engine_init_seconds": float(
                        engine_ready_time - engine_init_start
                    ),
                    "connector_request_rebuild_seconds": float(
                        requests_ready_time - state_ready_time
                    ),
                    "replayed_request_count": int(len(replayed_request_ids)),
                    "requests_resumed_to_ready": int(len(resumed_request_ids)),
                    "requests_decode_ready_to_ready": int(len(decode_ready_request_ids)),
                    "pending_request_count_before_ready": int(len(carried_pending_requests)),
                    "recoverable_request_count_before_ready": int(len(replayed_request_ids)),
                    "ready_marker": (
                        "all recoverable requests reached decode stage before pending requests were activated"
                    ),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                with paths["recovery_profiles_file"].open("a", encoding="utf-8") as _prof_handle:
                    _prof_handle.write(json.dumps(recovery_profile, ensure_ascii=False) + "\n")
                if maybe_write_first_recovery_profile(paths, recovery_profile):
                    print(
                        f"[connector] first recovery profiled "
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


def build_worker_cmd(
    args: argparse.Namespace,
    phase: str,
    worker_launch_index: int,
    parent_start_wall_time: float,
    fault_start_time: float | None,
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
        "--parent-start-wall-time",
        f"{parent_start_wall_time:.9f}",
        "--worker-launch-index",
        str(worker_launch_index),
    ]
    if args.model_label:
        cmd.extend(["--model-label", str(args.model_label)])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if fault_start_time is not None:
        cmd.extend(["--fault-start-time", f"{fault_start_time:.9f}"])
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

    rng = random.Random(args.seed)
    parent_start_wall_time = time.time()
    fault_injection_count = 0
    worker_launch_count = 0
    worker_restart_count = 0
    unexpected_worker_exit_count = 0
    fault_check_draw_count = 0
    first_worker_launch_wall_time: float | None = None
    current_phase = "creator"
    fault_pending = False
    pending_fault_start_time: float | None = None
    first_recovery_profile_pending = False
    first_recovery_fault_start_time: float | None = None
    next_fault_check_wall_time: float | None = None

    with paths["fault_file"].open("a", encoding="utf-8") as fault_handle:
        append_jsonl_record(
            fault_handle,
            {
                "event": "manager_start",
                "wall_time": float(parent_start_wall_time),
                "fault_frequency": float(args.fault_frequency),
                "fault_check_interval_seconds": float(args.fault_check_interval_seconds),
                "total_requests": int(args.total_requests),
                "concurrency": int(args.concurrency),
            },
        )

        def launch_worker(phase: str, fault_start_time: float | None = None) -> subprocess.Popen[str]:
            nonlocal next_fault_check_wall_time
            nonlocal worker_launch_count
            nonlocal first_worker_launch_wall_time
            launch_wall_time = time.time()
            effective_fault_start_time = launch_wall_time if phase == "connector" else fault_start_time
            cmd = build_worker_cmd(
                args,
                phase,
                worker_launch_index=worker_launch_count,
                parent_start_wall_time=parent_start_wall_time,
                fault_start_time=effective_fault_start_time,
            )
            if first_worker_launch_wall_time is None:
                first_worker_launch_wall_time = launch_wall_time
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
                completed_requests = int(runtime_state.get("completed_request_count", 0) or 0)

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
                        "relative_time_seconds": float(time.time() - parent_start_wall_time),
                        "unexpected_worker_exit_count": int(unexpected_worker_exit_count),
                    },
                )
                if unexpected_worker_exit_count > args.max_unexpected_worker_exits:
                    raise RuntimeError(
                        "Worker exited unexpectedly too many times. "
                        f"limit={args.max_unexpected_worker_exits}"
                    )
                current_phase = "connector" if paths["token_log_file"].exists() else current_phase
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
                -1 if runtime_worker_launch_index_raw is None else int(runtime_worker_launch_index_raw)
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
                        "first_fault_check_wall_time": float(next_fault_check_wall_time),
                    },
                )
                continue
            if now < next_fault_check_wall_time:
                continue

            observed_completed = int(runtime_state.get("completed_request_count", 0) or 0)
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
                    "relative_time_seconds": float(pending_fault_start_time - parent_start_wall_time),
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
    recovery_profile = load_json_if_exists(paths["recovery_profile_file"]) or {}
    first_inference_state = load_json_if_exists(paths["first_inference_file"]) or {}

    total_elapsed_start_wall_time = (
        first_worker_launch_wall_time
        if first_worker_launch_wall_time is not None
        else parent_start_wall_time
    )
    total_elapsed_start_marker = (
        "first_worker_launch"
        if first_worker_launch_wall_time is not None
        else "manager_start"
    )
    first_inference_start_wall_time = first_inference_state.get("inference_start_wall_time")
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
        float(completed_requests / total_elapsed_seconds) if total_elapsed_seconds > 0.0 else 0.0
    )
    average_ms_per_request = (
        float(total_elapsed_seconds * 1000.0 / completed_requests) if completed_requests > 0 else 0.0
    )

    _recovery_profiles: list[dict[str, Any]] = []
    if paths["recovery_profiles_file"].exists():
        with paths["recovery_profiles_file"].open(encoding="utf-8") as _prof_handle:
            for _line in _prof_handle:
                _line = _line.strip()
                if _line:
                    _recovery_profiles.append(json.loads(_line))
    _complete_profiles = [
        p
        for p in _recovery_profiles
        if p.get("fault_to_recovery_ready_seconds") is not None
        and p.get("connector_python_startup_seconds") is not None
        and p.get("connector_vllm_engine_startup_seconds") is not None
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
        "experiment": "e2e_vllm_replay_faults",
        "checkpoint_backend": "vllm_replay_incremental_token_log",
        "model": args.model,
        "model_label": metadata.get("model_label", model_label_from_arg(args.model, args.model_label)),
        "dtype": args.dtype,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "total_requests_target": int(args.total_requests),
        "completed_requests": int(completed_requests),
        "concurrency": int(args.concurrency),
        "seed": int(args.seed),
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


def main() -> int:
    """Entry point: dispatch to the manager or generation phase based on parsed args."""
    args = parse_args()
    args = normalize_run_paths(args)
    validate_args(args)
    if args.phase == "manager":
        return run_manager(args)
    return run_generation_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
