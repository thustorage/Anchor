#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ds_hf_checkpoint_bench as native


CREATOR_FAILURE_EXIT_CODE = native.CREATOR_FAILURE_EXIT_CODE
FAULT_EPOCH = native.FAULT_EPOCH
RESUME_EPOCHS = native.RESUME_EPOCHS
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_DIR = SCRIPT_DIR / "report1"
DEFAULT_IPC_BASELINE = "ipc"
IPC_TOOLS_DIR = WORKSPACE_ROOT / "IPC" / "ipc_tools"


def add_repo_paths() -> None:
    if str(IPC_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(IPC_TOOLS_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one HF causal LM with DeepSpeed while persisting weights and "
            "AdamW state in the IPC daemon, simulate a creator crash, then "
            "reattach live state from a connector and report attach timing."
        )
    )
    parser.add_argument("--phase", choices=["creator", "connector", "all"], default="all")
    parser.add_argument("--model", required=True, help="Local HF causal LM path or model identifier.")
    parser.add_argument("--model-label", default=None, help="Short label used in report filenames.")
    parser.add_argument("--state-file",
                        type=Path,
                        default=DEFAULT_REPORT_DIR / DEFAULT_IPC_BASELINE / "state.json",
                        help="Creator/connector shared state JSON.")
    parser.add_argument("--report-file",
                        type=Path,
                        default=DEFAULT_REPORT_DIR / DEFAULT_IPC_BASELINE / "report.json",
                        help="Final JSON report path.")
    parser.add_argument("--ipc-group",
                        default=None,
                        help="IPC tensor group prefix. Defaults to model-specific lab1 name.")
    parser.add_argument("--master-port",
                        type=int,
                        default=29661,
                        help="Port used for the single-rank DeepSpeed runtime.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--dataset-dir", type=Path, default=native.DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset-split",
                        choices=("train", "validation", "test"),
                        default=native.DATASET_SPLIT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--zero-stage", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--zero3-sub-group-size",
                        type=int,
                        default=None,
                        help="Optional DeepSpeed ZeRO-3 subgroup size override for finer-grained optimizer commits.")
    parser.add_argument("--dtype",
                        choices=("auto", "float16", "bfloat16", "float32"),
                        default="bfloat16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--inject-fault-sub-group",
                        type=int,
                        default=None,
                        help="Optional subgroup id where the creator should simulate a ZeRO-3 step failure.")
    parser.add_argument("--inject-fault-stage",
                        choices=("after_begin", "after_optimizer_step", "after_commit"),
                        default="after_optimizer_step",
                        help="Failure injection point within the target ZeRO-3 subgroup update.")
    parser.add_argument("--fault-start-time", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def normalize_run_paths(args: argparse.Namespace) -> argparse.Namespace:
    label = native.model_label_from_arg(args.model, args.model_label)
    if args.state_file == DEFAULT_REPORT_DIR / DEFAULT_IPC_BASELINE / "state.json":
        args.state_file = DEFAULT_REPORT_DIR / DEFAULT_IPC_BASELINE / f"{label}_state.json"
    if args.report_file == DEFAULT_REPORT_DIR / DEFAULT_IPC_BASELINE / "report.json":
        args.report_file = DEFAULT_REPORT_DIR / DEFAULT_IPC_BASELINE / f"{label}.json"
    return args


def validate_args(args: argparse.Namespace) -> None:
    native.validate_args(args)
    if args.zero_stage != 3:
        native.fail("IPC live-state benchmark currently supports only --zero-stage 3.")
    if args.zero3_sub_group_size is not None and args.zero3_sub_group_size <= 0:
        native.fail("--zero3-sub-group-size must be positive when provided.")
    if args.inject_fault_sub_group is not None and args.inject_fault_sub_group < 0:
        native.fail("--inject-fault-sub-group must be non-negative when provided.")


def load_runtime_modules() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    add_repo_paths()
    torch, deepspeed, auto_model_cls, auto_tokenizer_cls, load_dataset_fn = native.load_runtime_modules()
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        native.fail(f"transformers AutoConfig is required: {exc}")
    try:
        from ds_tool import DSTensorFactoryInterceptor
        from ipc_socket import IPCSocket
    except ImportError as exc:
        native.fail(f"IPC tooling is required: {exc}")
    return torch, deepspeed, auto_model_cls, auto_tokenizer_cls, AutoConfig, load_dataset_fn, DSTensorFactoryInterceptor, IPCSocket


def resolve_ipc_group(args: argparse.Namespace) -> str:
    if args.ipc_group:
        return args.ipc_group
    label = native.model_label_from_arg(args.model, args.model_label).replace(" ", "_")
    return f"lab1_ipc_{label}"


def zero3_optimizer_group_name(group_name: str) -> str:
    return f"{group_name}__optimizer_z3"


def build_tool(torch: Any, tool_cls: Any):
    return tool_cls(target_device=torch.device("cuda"))


def sync_cuda(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def suppress_model_init_resets() -> Any:
    try:
        from transformers.modeling_utils import no_init_weights
    except Exception:
        yield
        return
    with no_init_weights():
        yield


def load_model_from_config(args: argparse.Namespace, torch: Any, auto_config_cls: Any, auto_model_cls: Any):
    config = auto_config_cls.from_pretrained(args.model,
                                             trust_remote_code=args.trust_remote_code,
                                             local_files_only=args.local_files_only)
    if args.attn_implementation:
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = args.attn_implementation
        elif hasattr(config, "attn_implementation"):
            config.attn_implementation = args.attn_implementation

    resolved_dtype = native.resolve_dtype(torch, args.dtype)
    load_kwargs: dict[str, Any] = {}
    if resolved_dtype is not None:
        load_kwargs["torch_dtype"] = resolved_dtype
    try:
        with suppress_model_init_resets():
            model = auto_model_cls.from_config(config, trust_remote_code=args.trust_remote_code, **load_kwargs)
    except TypeError:
        with suppress_model_init_resets():
            model = auto_model_cls.from_config(config, **load_kwargs)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    return model


def build_creator_model(args: argparse.Namespace, torch: Any, auto_model_cls: Any):
    return native.load_model(args, torch, auto_model_cls)


def build_connector_model(args: argparse.Namespace, torch: Any, auto_model_cls: Any, auto_config_cls: Any):
    resolved_dtype = native.resolve_dtype(torch, args.dtype)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    if resolved_dtype is not None:
        load_kwargs["torch_dtype"] = resolved_dtype
    try:
        model = auto_model_cls.from_pretrained(args.model, **load_kwargs)
    except TypeError as exc:
        if resolved_dtype is None or "dtype" not in str(exc):
            raise
        load_kwargs.pop("dtype", None)
        load_kwargs["torch_dtype"] = resolved_dtype
        model = auto_model_cls.from_pretrained(args.model, **load_kwargs)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    return model


def optimizer_ipc_group_exists(tool: Any, group_name: str) -> bool:
    return tool.get_size(zero3_optimizer_group_name(group_name)) > 0


def restore_model_partitions_from_zero3_state(torch: Any, engine: Any) -> None:
    zero_optimizer = getattr(engine, "optimizer", None)
    if zero_optimizer is None:
        return

    restore_fn = getattr(zero_optimizer, "_reassign_or_swap_out_partitioned_parameters", None)
    fp32_groups = getattr(zero_optimizer, "fp32_partitioned_groups_flat", None)
    if restore_fn is None or fp32_groups is None:
        return

    with torch.no_grad():
        for sub_group_id in range(len(fp32_groups)):
            restore_fn(sub_group_id)
    native.sync_device(torch, engine.device)


def configure_zero3_ipc_env(group_name: str, tool_module: str, require_existing: bool) -> None:
    os.environ["DEEPSPEED_ZERO3_IPC_GROUP"] = group_name
    os.environ["DEEPSPEED_ZERO3_IPC_TOOL_MODULE"] = tool_module
    os.environ["DEEPSPEED_ZERO3_IPC_REQUIRE_EXISTING"] = "1" if require_existing else "0"


def configure_zero3_ipc_fault_env(args: argparse.Namespace, enabled: bool) -> None:
    keys = (
        "DEEPSPEED_ZERO3_IPC_FAULT_SUBGROUP",
        "DEEPSPEED_ZERO3_IPC_FAULT_STAGE",
        "DEEPSPEED_ZERO3_IPC_FAULT_EXIT_CODE",
    )
    if enabled and args.inject_fault_sub_group is not None:
        os.environ["DEEPSPEED_ZERO3_IPC_FAULT_SUBGROUP"] = str(args.inject_fault_sub_group)
        os.environ["DEEPSPEED_ZERO3_IPC_FAULT_STAGE"] = str(args.inject_fault_stage)
        os.environ["DEEPSPEED_ZERO3_IPC_FAULT_EXIT_CODE"] = str(CREATOR_FAILURE_EXIT_CODE)
        return
    for key in keys:
        os.environ.pop(key, None)


def build_ds_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "train_batch_size": args.batch_size,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 1,
        "wall_clock_breakdown": False,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": args.weight_decay,
            },
        },
    }
    if args.zero_stage > 0:
        config["zero_optimization"] = {
            "stage": args.zero_stage,
        }
        if args.zero_stage == 3 and args.zero3_sub_group_size is not None:
            config["zero_optimization"]["sub_group_size"] = int(args.zero3_sub_group_size)
    if args.dtype == "float16":
        config["fp16"] = {"enabled": True, "loss_scale": 0}
    if args.dtype == "bfloat16":
        config["bf16"] = {"enabled": True}
    return config


def create_engine(args: argparse.Namespace, torch: Any, deepspeed: Any, model: Any):
    engine, _, _, _ = deepspeed.initialize(model=model,
                                           model_parameters=[p for p in model.parameters() if p.requires_grad],
                                           config=build_ds_config(args))
    return engine


def summarize_largest_layer(model: Any, engine: Any) -> dict[str, Any]:
    zero_optimizer = getattr(engine, "optimizer", None)
    largest_layer_name = getattr(zero_optimizer, "largest_layer_name", None)
    largest_layer_params = getattr(zero_optimizer, "largest_layer_params", None)
    largest_layer_param_bytes = getattr(zero_optimizer, "largest_layer_param_bytes", None)
    if largest_layer_name is not None and largest_layer_params is not None and largest_layer_param_bytes is not None:
        return {
            "largest_layer_name": str(largest_layer_name),
            "largest_layer_parameter_count": int(largest_layer_params),
            "largest_layer_param_bytes": int(largest_layer_param_bytes),
        }

    largest_layer_name = ""
    largest_layer_params = 0
    largest_layer_param_bytes = 0
    for module_name, module in model.named_modules():
        layer_params = 0
        layer_param_bytes = 0
        layer_seen = set()
        for param in module.parameters(recurse=False):
            storage_key = param.data_ptr() if param.data_ptr() != 0 else id(param)
            if storage_key in layer_seen:
                continue
            layer_seen.add(storage_key)
            layer_params += param.numel()
            layer_param_bytes += param.numel() * param.element_size()
        if layer_params > largest_layer_params:
            largest_layer_name = module_name or "<root>"
            largest_layer_params = layer_params
            largest_layer_param_bytes = layer_param_bytes

    return {
        "largest_layer_name": str(largest_layer_name),
        "largest_layer_parameter_count": int(largest_layer_params),
        "largest_layer_param_bytes": int(largest_layer_param_bytes),
    }


def summarize_zero3_layout(engine: Any) -> dict[str, Any]:
    zero_optimizer = getattr(engine, "optimizer", None)
    fp16_groups = getattr(zero_optimizer, "fp16_groups", None)
    subgroup_count = len(fp16_groups) if fp16_groups is not None else 0
    sub_group_size = getattr(zero_optimizer, "sub_group_size", None)
    return {
        "zero3_optimizer_subgroup_count": int(subgroup_count),
        "zero3_sub_group_size": None if sub_group_size is None else int(sub_group_size),
    }


def summarize_ipc_undo_log(engine: Any) -> dict[str, Any]:
    zero_optimizer = getattr(engine, "optimizer", None)
    total_seconds = float(getattr(zero_optimizer, "_ipc_undo_log_total_seconds", 0.0)) if zero_optimizer is not None else 0.0
    last_step_seconds = float(getattr(zero_optimizer, "_ipc_undo_log_last_step_seconds", 0.0)) if zero_optimizer is not None else 0.0
    step_samples = int(getattr(zero_optimizer, "_ipc_undo_log_step_samples", 0)) if zero_optimizer is not None else 0
    avg_seconds = total_seconds / step_samples if step_samples > 0 else 0.0
    return {
        "ipc_undo_log_total_seconds": float(total_seconds),
        "ipc_undo_log_last_step_seconds": float(last_step_seconds),
        "ipc_undo_log_step_samples": int(step_samples),
        "ipc_undo_log_seconds_avg": float(avg_seconds),
    }


def run_one_step(engine: Any, batch: dict[str, Any]) -> float:
    outputs = engine(**batch)
    loss = native.extract_loss(outputs)
    engine.backward(loss)
    engine.step()
    return float(loss.detach().float().item())


def run_epoch(torch: Any,
              engine: Any,
              blocks: list[dict[str, list[int]]],
              batch_size: int,
              seq_len: int,
              pad_token_id: int,
              steps_per_epoch: int,
              epoch_idx: int,
              log_interval: int) -> tuple[float, float]:
    native.distributed_barrier(torch)
    native.sync_device(torch, engine.device)
    epoch_start = time.perf_counter()
    last_loss = 0.0
    for step_idx in range(steps_per_epoch):
        batch = native.build_batch_from_blocks(torch,
                                               blocks,
                                               step_idx,
                                               batch_size,
                                               seq_len,
                                               pad_token_id,
                                               engine.device)
        last_loss = run_one_step(engine, batch)
        if log_interval > 0 and ((step_idx + 1) % log_interval == 0 or step_idx + 1 == steps_per_epoch):
            native.log(f"[train] epoch={epoch_idx} step={step_idx + 1}/{steps_per_epoch} loss={last_loss:.6f}")
    native.sync_device(torch, engine.device)
    native.distributed_barrier(torch)
    epoch_seconds = time.perf_counter() - epoch_start
    native.log(f"[train] epoch={epoch_idx} completed epoch_seconds={epoch_seconds:.6f} last_loss={last_loss:.6f}")
    return native.max_across_ranks(torch, epoch_seconds, engine.device), last_loss


def destroy_training_state(torch: Any, engine: Any) -> None:
    optimizer = getattr(engine, "optimizer", None)
    if optimizer is not None and hasattr(optimizer, "destroy"):
        try:
            setattr(optimizer, "destroy", lambda: None)
        except Exception:
            pass
    native.destroy_engine(torch, engine)
    native.destroy_process_group(torch)


def shutdown_ipc_server(ipc_socket_cls: Any) -> None:
    try:
        ipc = ipc_socket_cls()
        sock = ipc.connect()
        ipc.send(sock, {"cmd": "EXIT"})
        sock.close()
        native.log("[all] requested IPC daemon shutdown")
    except Exception as exc:
        native.log(f"[all] IPC daemon shutdown skipped: {exc}")


def report_common_metadata(args: argparse.Namespace, stats: dict[str, Any], group_name: str) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "model_label": native.model_label_from_arg(args.model, args.model_label),
        "parameter_count": int(stats["parameter_count"]),
        "model_weight_bytes": int(stats["model_weight_bytes"]),
        "model_weight_gib": float(stats["model_weight_bytes"]) / (1024**3),
        "largest_layer_name": str(stats["largest_layer_name"]),
        "largest_layer_parameter_count": int(stats["largest_layer_parameter_count"]),
        "largest_layer_param_bytes": int(stats["largest_layer_param_bytes"]),
        "largest_layer_param_gib": float(stats["largest_layer_param_bytes"]) / (1024**3),
        "dtype": args.dtype,
        "zero_stage": args.zero_stage,
        "zero3_sub_group_size": stats.get("zero3_sub_group_size"),
        "zero3_optimizer_subgroup_count": int(stats.get("zero3_optimizer_subgroup_count", 0)),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "configured_steps_per_epoch": int(args.steps_per_epoch),
        "dataset_name": native.DATASET_NAME,
        "dataset_subset": native.DATASET_SUBSET,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_split": args.dataset_split,
        "fault_epoch": FAULT_EPOCH,
        "resume_epochs_after_recovery": RESUME_EPOCHS,
        "log_interval": args.log_interval,
        "state_file": str(args.state_file.resolve()),
        "report_file": str(args.report_file.resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_backend": "ipc_live_daemon",
        "checkpoint_save_seconds": 0.0,
        "checkpoint_size_bytes": 0,
        "checkpoint_size_gib": 0.0,
        "ipc_group": group_name,
        "ipc_optimizer_group": zero3_optimizer_group_name(group_name),
        "inject_fault_sub_group": args.inject_fault_sub_group,
        "inject_fault_stage": args.inject_fault_stage if args.inject_fault_sub_group is not None else None,
    }
    return payload


def build_creator_state(args: argparse.Namespace,
                        stats: dict[str, Any],
                        ipc_undo_log_stats: dict[str, Any],
                        group_name: str,
                        data_stats: dict[str, int],
                        available_step_count: int,
                        epoch_step_count: int,
                        epoch_seconds: float,
                        last_loss: float,
                        tool_init_seconds: float,
                        model_attach_seconds: float,
                        pretrained_weight_load_seconds: float,
                        optimizer_attach_seconds: float,
                        engine_init_seconds: float) -> dict[str, Any]:
    payload = report_common_metadata(args, stats, group_name)
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
        "epoch_train_seconds": float(epoch_seconds),
        "last_loss": float(last_loss),
        "checkpoint_save_seconds": float(ipc_undo_log_stats["ipc_undo_log_seconds_avg"]),
        "ipc_undo_log_seconds": float(ipc_undo_log_stats["ipc_undo_log_seconds_avg"]),
        "ipc_undo_log_total_seconds": float(ipc_undo_log_stats["ipc_undo_log_total_seconds"]),
        "ipc_undo_log_last_step_seconds": float(ipc_undo_log_stats["ipc_undo_log_last_step_seconds"]),
        "ipc_undo_log_step_samples": int(ipc_undo_log_stats["ipc_undo_log_step_samples"]),
        "creator_ipc_handshake_seconds": float(tool_init_seconds),
        "creator_model_attach_seconds": float(model_attach_seconds),
        "creator_pretrained_weight_load_seconds": float(pretrained_weight_load_seconds),
        "creator_optimizer_attach_seconds": float(optimizer_attach_seconds),
        "creator_engine_init_seconds": float(engine_init_seconds),
        "ipc_live_ready_seconds": float(
            tool_init_seconds + model_attach_seconds + pretrained_weight_load_seconds +
            optimizer_attach_seconds + engine_init_seconds
        ),
    })
    return payload


def run_creator(args: argparse.Namespace) -> int:
    """Run the initial DeepSpeed training that populates IPC live state and injects the fault."""
    state_file = args.state_file.resolve()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    if state_file.exists():
        state_file.unlink()

    torch, deepspeed, auto_model_cls, auto_tokenizer_cls, auto_config_cls, load_dataset_fn, tool_cls, _ = load_runtime_modules()
    native.ensure_single_rank_env(args)
    device = native.init_distributed(args, torch, deepspeed)
    native.seed_everything(args.seed, torch)
    group_name = resolve_ipc_group(args)

    tool_init_start = time.perf_counter()
    tool = build_tool(torch, tool_cls)
    tool_init_seconds = native.max_across_ranks(torch, time.perf_counter() - tool_init_start, device)

    group_exists = optimizer_ipc_group_exists(tool, group_name)
    if group_exists:
        native.fail(f"IPC group '{group_name}' already exists. Use a fresh group name.")

    pretrained_weight_load_start = time.perf_counter()
    model = build_creator_model(args, torch, auto_model_cls)
    pretrained_weight_load_seconds = native.max_across_ranks(
        torch,
        time.perf_counter() - pretrained_weight_load_start,
        device,
    )
    model_attach_seconds = 0.0

    configure_zero3_ipc_env(group_name, "ds_tool", require_existing=False)
    configure_zero3_ipc_fault_env(args, enabled=True)
    engine_init_start = time.perf_counter()
    engine = create_engine(args, torch, deepspeed, model)
    engine_init_seconds = native.max_across_ranks(torch, time.perf_counter() - engine_init_start, engine.device)
    largest_layer_stats = summarize_largest_layer(model, engine)
    zero3_layout_stats = summarize_zero3_layout(engine)
    stats = {
        "parameter_count": native.parameter_count(model),
        "model_weight_bytes": native.model_weight_bytes(model),
        **largest_layer_stats,
        **zero3_layout_stats,
    }
    optimizer_attach_seconds = 0.0

    tokenizer = native.load_tokenizer(args, auto_tokenizer_cls)
    blocks, data_stats = native.build_token_blocks(args, tokenizer, load_dataset_fn)
    available_step_count = native.batches_per_epoch(blocks, args.batch_size)
    epoch_step_count = native.resolve_epoch_steps(available_step_count, args.steps_per_epoch)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        native.fail("Tokenizer must provide either pad_token_id or eos_token_id.")

    native.log(
        f"[creator] model={native.model_label_from_arg(args.model, args.model_label)} "
        f"steps_per_epoch={epoch_step_count} available_steps_per_epoch={available_step_count} "
        f"dataset_rows={data_stats['dataset_rows']} token_blocks={data_stats['token_blocks']}"
    )

    initial_payload = build_creator_state(args,
                                          stats,
                                          summarize_ipc_undo_log(engine),
                                          group_name,
                                          data_stats,
                                          available_step_count,
                                          epoch_step_count,
                                          0.0,
                                          0.0,
                                          tool_init_seconds,
                                          model_attach_seconds,
                                          pretrained_weight_load_seconds,
                                          optimizer_attach_seconds,
                                          engine_init_seconds)
    initial_payload["train_epoch"] = 0
    native.maybe_write_json(state_file, initial_payload)

    try:
        epoch_seconds, last_loss = run_epoch(torch,
                                             engine,
                                             blocks,
                                             args.batch_size,
                                             args.seq_len,
                                             int(pad_token_id),
                                             epoch_step_count,
                                             FAULT_EPOCH,
                                             args.log_interval)
        payload = build_creator_state(args,
                                      stats,
                                      summarize_ipc_undo_log(engine),
                                      group_name,
                                      data_stats,
                                      available_step_count,
                                      epoch_step_count,
                                      epoch_seconds,
                                      last_loss,
                                      tool_init_seconds,
                                      model_attach_seconds,
                                      pretrained_weight_load_seconds,
                                      optimizer_attach_seconds,
                                      engine_init_seconds)
        payload["train_epoch"] = FAULT_EPOCH
        native.maybe_write_json(state_file, payload)
        return CREATOR_FAILURE_EXIT_CODE
    finally:
        destroy_training_state(torch, engine)


def run_connector(args: argparse.Namespace) -> int:
    """Recover by reattaching live IPC state, resuming training, and reporting attach timing."""
    connector_entry_time = time.perf_counter()
    recovery_start_time = args.fault_start_time if args.fault_start_time is not None else connector_entry_time
    state = native.load_json(args.state_file.resolve())
    group_name = str(state["ipc_group"])

    engine_init_start = time.perf_counter()
    torch, deepspeed, auto_model_cls, auto_tokenizer_cls, auto_config_cls, load_dataset_fn, tool_cls, _ = load_runtime_modules()
    native.ensure_single_rank_env(args)
    device = native.init_distributed(args, torch, deepspeed)
    native.seed_everything(args.seed, torch)
    engine_bootstrap_seconds = native.max_across_ranks(torch, time.perf_counter() - engine_init_start, device)

    tool_init_start = time.perf_counter()
    tool = build_tool(torch, tool_cls)
    tool_init_seconds = native.max_across_ranks(torch, time.perf_counter() - tool_init_start, device)

    group_exists = optimizer_ipc_group_exists(tool, group_name)
    if not group_exists:
        native.fail(f"IPC group '{group_name}' does not exist.")

    model_attach_start = time.perf_counter()
    model = build_connector_model(args, torch, auto_model_cls, auto_config_cls)
    model_attach_seconds = native.max_across_ranks(torch, time.perf_counter() - model_attach_start, device)

    configure_zero3_ipc_env(group_name, "ds_tool", require_existing=True)
    configure_zero3_ipc_fault_env(args, enabled=False)
    deepspeed_init_start = time.perf_counter()
    engine = create_engine(args, torch, deepspeed, model)
    deepspeed_init_seconds = native.max_across_ranks(torch, time.perf_counter() - deepspeed_init_start, engine.device)
    resume_inflight_step = False
    resume_fn = getattr(engine.optimizer, "resume_ipc_inflight_step_if_needed", None)
    if callable(resume_fn):
        resume_inflight_step = bool(resume_fn())
    resume_debug = getattr(engine.optimizer, "_ipc_resume_state", {"resumed": False})
    if not resume_inflight_step:
        restore_model_partitions_from_zero3_state(torch, engine)
    engine_init_seconds = engine_bootstrap_seconds + deepspeed_init_seconds

    optimizer_attach_seconds = 0.0
    attach_total_seconds = tool_init_seconds + model_attach_seconds + optimizer_attach_seconds
    recovery_total_seconds = time.perf_counter() - recovery_start_time
    recovery_total_seconds = native.max_across_ranks(torch, recovery_total_seconds, engine.device)
    connector_other_seconds = max(0.0, recovery_total_seconds - engine_init_seconds - attach_total_seconds)

    native.log(
        f"[resume] ipc_group={group_name} recovery_ready_seconds={recovery_total_seconds:.6f} "
        f"attach_seconds={attach_total_seconds:.6f}"
    )

    tokenizer = native.load_tokenizer(args, auto_tokenizer_cls)
    blocks, data_stats = native.build_token_blocks(args, tokenizer, load_dataset_fn)
    available_step_count = native.batches_per_epoch(blocks, args.batch_size)
    epoch_step_count = native.resolve_epoch_steps(available_step_count, args.steps_per_epoch)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        native.fail("Tokenizer must provide either pad_token_id or eos_token_id.")

    try:
        resumed_epoch = int(state.get("train_epoch", FAULT_EPOCH))
        resume_epoch_seconds = 0.0
        resume_last_loss = 0.0
        for resume_epoch_idx in range(1, RESUME_EPOCHS + 1):
            current_epoch = resumed_epoch + resume_epoch_idx
            resume_epoch_seconds, resume_last_loss = run_epoch(torch,
                                                               engine,
                                                               blocks,
                                                               args.batch_size,
                                                               args.seq_len,
                                                               int(pad_token_id),
                                                               epoch_step_count,
                                                               current_epoch,
                                                               args.log_interval)
    finally:
        destroy_training_state(torch, engine)

    payload = dict(state)
    payload.update({
        "fault_to_recovery_ready_seconds": float(recovery_total_seconds),
        "connector_engine_init_seconds": float(engine_init_seconds),
        "connector_engine_bootstrap_seconds": float(engine_bootstrap_seconds),
        "connector_deepspeed_init_seconds": float(deepspeed_init_seconds),
        "connector_checkpoint_load_seconds": float(attach_total_seconds),
        "connector_ipc_handshake_seconds": float(tool_init_seconds),
        "connector_model_attach_seconds": float(model_attach_seconds),
        "connector_optimizer_attach_seconds": float(optimizer_attach_seconds),
        "connector_other_seconds": float(connector_other_seconds),
        "connector_resumed_inflight_step": bool(resume_inflight_step),
        "connector_resume_detected_active_sub_group": resume_debug.get("detected_active_sub_group"),
        "connector_resume_rolled_back_sub_group": resume_debug.get("rolled_back_sub_group"),
        "connector_resume_start_sub_group": resume_debug.get("resume_start_sub_group"),
        "connector_resume_total_subgroups": resume_debug.get("resume_total_subgroups"),
        "connector_resume_replayed_sub_groups": list(resume_debug.get("replayed_sub_groups", [])),
        "loaded_train_epoch": int(state.get("train_epoch", 0)),
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
    if args.zero3_sub_group_size is not None:
        cmd.extend(["--zero3-sub-group-size", str(args.zero3_sub_group_size)])
    if args.model_label:
        cmd.extend(["--model-label", str(args.model_label)])
    if args.ipc_group:
        cmd.extend(["--ipc-group", str(args.ipc_group)])
    if args.gradient_checkpointing:
        cmd.append("--gradient-checkpointing")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.attn_implementation:
        cmd.extend(["--attn-implementation", str(args.attn_implementation)])
    if phase == "creator" and args.inject_fault_sub_group is not None:
        cmd.extend([
            "--inject-fault-sub-group",
            str(args.inject_fault_sub_group),
        ])
        cmd.extend([
            "--inject-fault-stage",
            str(args.inject_fault_stage),
        ])
    return cmd


def phase_all(args: argparse.Namespace) -> int:
    _, _, _, _, _, _, _, ipc_socket_cls = load_runtime_modules()
    creator_cmd = build_subprocess_args(args, "creator")
    native.log("[all] starting creator process")
    try:
        rc = subprocess.call(creator_cmd)
        if rc != CREATOR_FAILURE_EXIT_CODE:
            return rc

        fault_start_time = time.perf_counter()
        connector_cmd = build_subprocess_args(args, "connector")
        connector_cmd.extend(["--fault-start-time", f"{fault_start_time:.9f}"])
        native.log("[all] starting connector process")
        return subprocess.call(connector_cmd)
    finally:
        shutdown_ipc_server(ipc_socket_cls)


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
