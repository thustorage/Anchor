#!/usr/bin/env python3
"""Measure vLLM serving overhead on the ShareGPT workload with and without IPC."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_ROOT = SCRIPT_DIR / "report1"
REQUEST_ID_PREFIX = "sharegpt_request_"
PROGRESS_LOG_INTERVAL_TOKENS = 10_000
DEFAULT_SHAREGPT_DATASET_PATH = Path(
    "/root/data/mhy/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json"
)


def str2bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_model_label(model: str, provided_label: str | None) -> str:
    if provided_label:
        return provided_label
    return Path(model.rstrip("/")).name


def label_to_stem(label: str) -> str:
    stem = re.sub(r"[^0-9A-Za-z]+", "_", label.strip().lower()).strip("_")
    return stem or "model"


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.model_label = resolve_model_label(args.model, args.model_label)
    default_stem = label_to_stem(args.model_label)
    method_dir = DEFAULT_REPORT_ROOT / args.method
    if args.state_file is None:
        args.state_file = method_dir / f"{default_stem}_state.json"
    if args.report_file is None:
        args.report_file = method_dir / f"{default_stem}.json"
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure steady-state vLLM ShareGPT inference overhead for baseline and IPC. "
            "The request driver uses fixed concurrency with continuous top-up. "
            "Single-GPU runs are just --tensor-parallel-size 1."
        )
    )
    parser.add_argument("--method", required=True, choices=("baseline", "ipc"))
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--report-file", type=Path, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--enable-prefix-caching", type=str2bool, default=False)
    parser.add_argument("--send-exit-on-finish", type=str2bool, default=True)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_SHAREGPT_DATASET_PATH)
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--disable-shuffle", type=str2bool, default=False)
    parser.add_argument("--warmup-processed-tokens", type=int, default=50_000)
    parser.add_argument("--measure-processed-tokens", type=int, default=100_000)
    parser.add_argument("--log-interval-tokens", type=int, default=PROGRESS_LOG_INTERVAL_TOKENS)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=16384)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.dataset_path is None:
        raise SystemExit("--dataset-path is required.")
    if not args.dataset_path.exists():
        raise SystemExit(f"ShareGPT dataset not found: {args.dataset_path}")
    if args.num_requests <= 0:
        raise SystemExit("--num-requests must be positive.")
    if args.max_model_len <= 0:
        raise SystemExit("--max-model-len must be positive.")
    if args.warmup_processed_tokens < 0:
        raise SystemExit("--warmup-processed-tokens must be non-negative.")
    if args.measure_processed_tokens <= 0:
        raise SystemExit("--measure-processed-tokens must be positive.")
    if args.log_interval_tokens < 0:
        raise SystemExit("--log-interval-tokens must be non-negative.")


def validate_gpu_args(args: argparse.Namespace, torch: Any) -> None:
    if args.tensor_parallel_size <= 0:
        raise SystemExit("--tensor-parallel-size must be positive.")

    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_gpu_count > 0 and args.tensor_parallel_size > visible_gpu_count:
        raise SystemExit(
            f"--tensor-parallel-size={args.tensor_parallel_size} exceeds the visible GPU "
            f"count {visible_gpu_count}. Check CUDA_VISIBLE_DEVICES or lower the TP size."
        )


def configure_environment(ipc_enabled: bool) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_IPC_TOOL"] = "1" if ipc_enabled else "0"
    os.environ.pop("VLLM_DEBUG_SCHEDULE_PREFILL_TOKENS", None)
    os.environ.pop("VLLM_KV_CPU_CHECKPOINT_PATH", None)
    os.environ.pop("VLLM_KV_CPU_BUFFER_METADATA", None)


def import_runtime_modules() -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    return torch, EngineArgs, LLMEngine, SamplingParams, RequestOutputKind


def try_send_ipc_exit_signal() -> bool:
    ipc_tools_dir = Path(__file__).resolve().parents[2] / "ipc_tools"
    ipc_tools_path = str(ipc_tools_dir)
    if ipc_tools_path not in sys.path:
        sys.path.insert(0, ipc_tools_path)

    try:
        from ipc_socket import IPCSocket
    except Exception as exc:
        print(f"[cleanup] IPC EXIT skipped (import failed): {exc}")
        return False

    sock = None
    try:
        ipc = IPCSocket()
        sock = ipc.connect()
        ipc.send(sock, {"cmd": "EXIT"})
        print("[cleanup] IPC EXIT sent.")
        return True
    except Exception as exc:
        print(f"[cleanup] IPC EXIT skipped: {exc}")
        return False
    finally:
        if sock is not None:
            with contextlib.suppress(Exception):
                sock.close()


def build_engine(
    args: argparse.Namespace,
    engine_args_cls: Any,
    llm_engine_cls: Any,
) -> Any:
    engine_args = engine_args_cls(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        disable_log_stats=True,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
    )
    return llm_engine_cls.from_engine_args(engine_args)


def build_sampling_params(
    sampling_params_cls: Any,
    request_output_kind: Any,
    max_tokens: int,
) -> Any:
    return sampling_params_cls(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
        output_kind=request_output_kind.DELTA,
    )


def load_request_templates(engine: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    tokenizer = engine.get_tokenizer()
    with open(args.dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

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
                "prompt_token_ids": prompt_token_ids,
                "prompt_tokens": int(prompt_len),
                "target_output_len": int(target_output_len),
            }
        )

    if len(templates) < args.num_requests:
        raise RuntimeError(
            f"Only sampled {len(templates)} valid ShareGPT requests from "
            f"{args.dataset_path}; requested concurrency {args.num_requests}. "
            "Increase --max-model-len or lower --num-requests."
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
    engine: Any,
    request_id: str,
    request_state: dict[str, Any],
    sampling_params_cls: Any,
    request_output_kind: Any,
) -> None:
    engine.add_request(
        request_id,
        {"prompt_token_ids": request_state["prompt_token_ids"]},
        build_sampling_params(
            sampling_params_cls,
            request_output_kind,
            int(request_state["target_output_len"]),
        ),
    )


def sample_gpu_used_bytes_avg(torch: Any) -> float:
    """Return the average used GPU memory (bytes) across all visible GPUs."""
    if not torch.cuda.is_available():
        return 0.0
    used_bytes_per_gpu: list[float] = []
    for device_idx in range(int(torch.cuda.device_count())):
        torch.cuda.synchronize(device_idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
        used_bytes_per_gpu.append(float(total_bytes - free_bytes))
    if not used_bytes_per_gpu:
        return 0.0
    return sum(used_bytes_per_gpu) / len(used_bytes_per_gpu)


def get_engine_core(engine: Any):
    client = engine.engine_core
    core = getattr(client, "engine_core", None)
    if core is None:
        raise RuntimeError(
            "Internal EngineCore is not directly accessible. "
            "This script requires in-process engine mode."
        )
    return core


def get_scheduler(engine: Any):
    core = get_engine_core(engine)
    scheduler = getattr(core, "scheduler", None)
    if scheduler is None:
        raise RuntimeError("Scheduler not found on internal EngineCore.")
    return scheduler


def get_output_processor(engine: Any):
    output_processor = getattr(engine, "output_processor", None)
    if output_processor is None:
        raise RuntimeError("Output processor not found on LLMEngine.")
    return output_processor


def resolve_internal_request_id(engine: Any, external_request_id: str) -> str | None:
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
    log = _INCREMENTAL_TOKEN_LOG
    if log is None:
        return
    log.write(json.dumps({
        "b": request_id,
        "g": [[int(block_id) for block_id in group] for group in group_block_ids],
    }, ensure_ascii=False) + "\n")
    log.flush()


class IpcMetadataRecorder:
    """Record each request's physical block table (b record) per step, writing only on change. Same as e2e ipc."""

    def __init__(self, engine: Any):
        self.engine = engine
        self.recorded: dict[str, list[list[int]]] = {}

    def record(self, active_requests: dict[str, dict[str, Any]]) -> None:
        scheduler = get_scheduler(self.engine)
        for request_id in active_requests:
            internal_request_id = resolve_internal_request_id(self.engine, request_id)
            if internal_request_id is None:
                continue
            if scheduler.requests.get(internal_request_id) is None:
                continue
            kv_block_ids = scheduler.kv_cache_manager.get_block_ids(internal_request_id)
            groups = [[int(block_id) for block_id in group] for group in kv_block_ids]
            if self.recorded.get(request_id) != groups:
                log_incremental_blocks(request_id, groups)
                self.recorded[request_id] = groups


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Drive the ShareGPT overhead measurement and return the result payload."""
    ipc_enabled = args.method == "ipc"
    configure_environment(ipc_enabled=ipc_enabled)
    torch, engine_args_cls, llm_engine_cls, sampling_params_cls, request_output_kind = (
        import_runtime_modules()
    )
    validate_gpu_args(args, torch)

    engine = build_engine(args, engine_args_cls, llm_engine_cls)
    visible_gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0

    metadata_recorder: IpcMetadataRecorder | None = None
    if ipc_enabled:
        token_log_file = args.report_file.resolve().with_name(
            args.report_file.stem + "_tokenlog.jsonl"
        )
        if token_log_file.exists():
            token_log_file.unlink()
        open_incremental_token_log({"token_log_file": token_log_file})
        metadata_recorder = IpcMetadataRecorder(engine)

    templates = load_request_templates(engine, args)

    active_requests: dict[str, dict[str, Any]] = {}
    issued_requests = 0
    completed_requests = 0
    total_prompt_tokens_issued = 0
    total_target_output_tokens_issued = 0

    def top_up_concurrency() -> None:
        nonlocal issued_requests
        nonlocal total_prompt_tokens_issued
        nonlocal total_target_output_tokens_issued

        while len(active_requests) < args.num_requests:
            template = templates[issued_requests % len(templates)]
            request_id = f"{REQUEST_ID_PREFIX}{issued_requests}"
            request_state = instantiate_request_state(template)
            add_request_to_engine(
                engine,
                request_id,
                request_state,
                sampling_params_cls,
                request_output_kind,
            )
            active_requests[request_id] = request_state
            if ipc_enabled:
                log_incremental_add(request_id, request_state)
            issued_requests += 1
            total_prompt_tokens_issued += int(request_state["prompt_tokens"])
            total_target_output_tokens_issued += int(request_state["target_output_len"])

    top_up_concurrency()

    warmup_processed_tokens = 0
    warmup_generated_tokens = 0
    measured_processed_tokens = 0
    measured_generated_tokens = 0
    benchmark_seconds = 0.0
    mid_benchmark_used_bytes_avg: float | None = None
    benchmark_started_at: float | None = None
    phase = "warmup" if args.warmup_processed_tokens > 0 else "measure"
    if phase == "measure":
        benchmark_started_at = time.perf_counter()
    next_progress_mark = args.log_interval_tokens if args.log_interval_tokens > 0 else 0
    midpoint_target = max(1, args.measure_processed_tokens // 2)

    print(
        f"[run] method={args.method} model={args.model_label} "
        f"num_requests={args.num_requests} warmup_processed_tokens={args.warmup_processed_tokens} "
        f"measure_processed_tokens={args.measure_processed_tokens} "
        f"tensor_parallel_size={args.tensor_parallel_size}"
    )

    try:
        while engine.has_unfinished_requests():
            request_outputs = engine.step()
            finished_request_ids: list[str] = []
            step_new_tokens: dict[str, list[int]] = {}
            phase_processed_delta = 0
            phase_generated_delta = 0

            for request_output in request_outputs:
                request_state = active_requests.get(request_output.request_id)
                if request_state is None:
                    continue

                new_token_ids = request_output.outputs[0].token_ids
                if new_token_ids and not request_state["prefill_accounted"]:
                    request_state["prefill_accounted"] = True
                    phase_processed_delta += int(request_state["prompt_tokens"])

                if request_output.finished:
                    request_state["finished"] = True
                    finished_request_ids.append(request_output.request_id)

                if not new_token_ids:
                    continue

                request_state["generated_token_ids"].extend(new_token_ids)
                step_new_tokens[request_output.request_id] = list(new_token_ids)
                phase_generated_delta += len(new_token_ids)
                phase_processed_delta += len(new_token_ids)

            if ipc_enabled:
                _log_incremental_step(step_new_tokens, finished_request_ids)

            for request_id in finished_request_ids:
                if request_id in active_requests:
                    active_requests.pop(request_id)
                    completed_requests += 1

            if metadata_recorder is not None:
                metadata_recorder.record(active_requests)

            top_up_concurrency()

            if phase == "warmup":
                warmup_processed_tokens += phase_processed_delta
                warmup_generated_tokens += phase_generated_delta

                while (
                    next_progress_mark > 0
                    and warmup_processed_tokens >= next_progress_mark
                ):
                    print(
                        f"[warmup-progress] processed_tokens={next_progress_mark} "
                        f"current={warmup_processed_tokens} "
                        f"target={args.warmup_processed_tokens}"
                    )
                    next_progress_mark += args.log_interval_tokens

                if warmup_processed_tokens >= args.warmup_processed_tokens:
                    phase = "measure"
                    measured_processed_tokens = 0
                    measured_generated_tokens = 0
                    benchmark_started_at = time.perf_counter()
                    next_progress_mark = (
                        args.log_interval_tokens if args.log_interval_tokens > 0 else 0
                    )
                    print(
                        f"[warmup] completed processed_tokens={warmup_processed_tokens} "
                        f"generated_tokens={warmup_generated_tokens}"
                    )
                    continue

            else:
                measured_processed_tokens += phase_processed_delta
                measured_generated_tokens += phase_generated_delta

                if (
                    mid_benchmark_used_bytes_avg is None
                    and measured_processed_tokens >= midpoint_target
                ):
                    mid_benchmark_used_bytes_avg = sample_gpu_used_bytes_avg(torch)
                    print(
                        f"[memory] midpoint processed_tokens={measured_processed_tokens} "
                        f"used_gib_avg={mid_benchmark_used_bytes_avg / (1024**3):.4f}"
                    )

                while (
                    next_progress_mark > 0
                    and measured_processed_tokens >= next_progress_mark
                ):
                    print(
                        f"[measure-progress] processed_tokens={next_progress_mark} "
                        f"current={measured_processed_tokens} "
                        f"target={args.measure_processed_tokens}"
                    )
                    next_progress_mark += args.log_interval_tokens

                if measured_processed_tokens >= args.measure_processed_tokens:
                    break

        if phase == "warmup":
            raise RuntimeError(
                "Warmup did not reach the requested processed-token target before the "
                "engine stopped producing work."
            )
        if benchmark_started_at is None:
            raise RuntimeError("Benchmark measurement never started.")
        if measured_processed_tokens < args.measure_processed_tokens:
            raise RuntimeError(
                "Benchmark stopped before reaching the requested processed-token target. "
                f"current={measured_processed_tokens}, "
                f"target={args.measure_processed_tokens}."
            )

        benchmark_seconds = time.perf_counter() - benchmark_started_at
        if mid_benchmark_used_bytes_avg is None:
            mid_benchmark_used_bytes_avg = sample_gpu_used_bytes_avg(torch)

        average_token_throughput_tps = (
            float(measured_processed_tokens) / benchmark_seconds
            if benchmark_seconds > 0
            else 0.0
        )

        payload = {
            "experiment": "sharegpt_inference_overhead",
            "method": args.method,
            "ipc_enabled": bool(ipc_enabled),
            "runtime_backend": "vllm_ipc_live" if ipc_enabled else "vllm_baseline",
            "model": args.model,
            "model_label": args.model_label,
            "dtype": args.dtype,
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "visible_gpu_count": int(visible_gpu_count),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "enable_prefix_caching": bool(args.enable_prefix_caching),
            "seed": int(args.seed),
            "num_requests": int(args.num_requests),
            "max_model_len": int(args.max_model_len),
            "warmup_processed_tokens_target": int(args.warmup_processed_tokens),
            "measure_processed_tokens_target": int(args.measure_processed_tokens),
            "mid_benchmark_gpu_used_gib_avg": float(mid_benchmark_used_bytes_avg / (1024**3)),
            "average_token_throughput_tps": float(average_token_throughput_tps),
        }
        return payload
    finally:
        close_incremental_token_log()
        if ipc_enabled and args.send_exit_on_finish:
            try_send_ipc_exit_signal()


def main() -> int:
    """Parse CLI arguments and run the ShareGPT overhead benchmark."""
    args = normalize_paths(parse_args())
    validate_args(args)
    print(
        f"[main] method={args.method} pid={os.getpid()} model={args.model} "
        f"label={args.model_label} tensor_parallel_size={args.tensor_parallel_size}"
    )
    payload = run_benchmark(args)
    save_json(args.state_file.resolve(), payload)
    save_json(args.report_file.resolve(), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
