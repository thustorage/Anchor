#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CREATOR_FAILURE_EXIT_CODE = 86
FAULT_EPOCH = 1
RESUME_EPOCHS = 1
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_ROOT = SCRIPT_DIR / "ckpt"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "report1"
DEFAULT_DATASET_DIR = WORKSPACE_ROOT / "datasets" / "wikitext" / "wikitext-2-raw-v1"
DATASET_NAME = "local_wikitext"
DATASET_SUBSET = "wikitext-2-raw-v1"
DATASET_SPLIT = "train"


def add_repo_paths() -> Path:
    repo_root = WORKSPACE_ROOT
    deepspeed_root = repo_root / "DeepSpeed"
    if str(deepspeed_root) not in sys.path:
        sys.path.insert(0, str(deepspeed_root))
    return repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one HF causal LM with native DeepSpeed, save a checkpoint at the "
            "failure boundary, relaunch a connector, and report save/load timing."
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
                        help="Directory used to save and reload DeepSpeed checkpoints.")
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
                        default=29631,
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
                        default=DEFAULT_DATASET_DIR,
                        help="Local directory containing WikiText-2 parquet shards.")
    parser.add_argument("--dataset-split",
                        choices=("train", "validation", "test"),
                        default=DATASET_SPLIT,
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


def fail(message: str) -> None:
    raise SystemExit(message)


def rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(message: str) -> None:
    if rank0():
        print(message, flush=True)


def ensure_single_rank_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))


def resolve_dtype(torch: Any, dtype_name: str) -> Any | None:
    if dtype_name == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync_device(torch: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def distributed_barrier(torch: Any) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def destroy_process_group(torch: Any) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def max_across_ranks(torch: Any, value: float, device: Any) -> float:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return float(value)
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor.item())


def load_runtime_modules() -> tuple[Any, Any, Any]:
    add_repo_paths()
    try:
        import torch
    except ImportError as exc:
        fail(f"PyTorch is required: {exc}")
    try:
        import deepspeed
    except ImportError as exc:
        fail(f"DeepSpeed is required: {exc}")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        fail(f"transformers is required: {exc}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        fail(f"datasets is required for local WikiText-2 parquet loading: {exc}")
    return torch, deepspeed, AutoModelForCausalLM, AutoTokenizer, load_dataset


def init_distributed(args: argparse.Namespace, torch: Any, deepspeed: Any) -> Any:
    ensure_single_rank_env(args)
    if not torch.cuda.is_available():
        fail("CUDA/ROCm GPU is required for this benchmark.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        deepspeed.init_distributed(dist_backend="nccl")
    return torch.device(f"cuda:{local_rank}")


def model_label_from_arg(model_arg: str, explicit_label: str | None) -> str:
    if explicit_label:
        return explicit_label
    model_path = Path(model_arg)
    if model_path.name:
        return model_path.name
    return model_arg.rstrip("/").split("/")[-1]


def normalize_run_paths(args: argparse.Namespace) -> argparse.Namespace:
    label = model_label_from_arg(args.model, args.model_label)

    if args.checkpoint_dir == DEFAULT_CHECKPOINT_ROOT:
        args.checkpoint_dir = DEFAULT_CHECKPOINT_ROOT / label
    if args.state_file == DEFAULT_REPORT_DIR / "state.json":
        args.state_file = DEFAULT_REPORT_DIR / f"{label}_state.json"
    if args.report_file == DEFAULT_REPORT_DIR / "report.json":
        args.report_file = DEFAULT_REPORT_DIR / f"{label}.json"
    return args


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
        "checkpoint": {
            "tag_validation": "WARN",
        },
    }
    if args.zero_stage > 0:
        config["zero_optimization"] = {
            "stage": args.zero_stage,
        }
    if args.dtype == "float16":
        config["fp16"] = {"enabled": True, "loss_scale": 0}
    if args.dtype == "bfloat16":
        config["bf16"] = {"enabled": True}
    return config


def load_model(args: argparse.Namespace, torch: Any, auto_model_cls: Any):
    resolved_dtype = resolve_dtype(torch, args.dtype)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    if resolved_dtype is not None:
        load_kwargs["dtype"] = resolved_dtype
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


def load_model_fast(args: argparse.Namespace, torch: Any, auto_model_cls: Any):
    """Load model with low_cpu_mem_usage for near-zero-copy safetensor mapping."""
    resolved_dtype = resolve_dtype(torch, args.dtype)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    if resolved_dtype is not None:
        load_kwargs["dtype"] = resolved_dtype
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


def parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def model_weight_bytes(model: Any) -> int:
    return int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()))


def extract_loss(outputs: Any) -> Any:
    if hasattr(outputs, "loss"):
        return outputs.loss
    if isinstance(outputs, dict) and "loss" in outputs:
        return outputs["loss"]
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def load_tokenizer(args: argparse.Namespace, auto_tokenizer_cls: Any):
    tokenizer = auto_tokenizer_cls.from_pretrained(args.model,
                                                   trust_remote_code=args.trust_remote_code,
                                                   local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def extract_row_text(row: Any) -> str:
    if isinstance(row, dict):
        for key in ("text", "content", "sentence"):
            value = row.get(key)
            if isinstance(value, str):
                return value
        for value in row.values():
            if isinstance(value, str):
                return value
    return ""


def make_lm_block(token_ids: list[int], seq_len: int, pad_token_id: int) -> dict[str, list[int]]:
    valid_tokens = len(token_ids)
    if valid_tokens > seq_len:
        fail(f"Internal error: block length {valid_tokens} exceeds seq_len={seq_len}")
    pad_count = seq_len - valid_tokens
    return {
        "input_ids": list(token_ids) + ([pad_token_id] * pad_count),
        "attention_mask": ([1] * valid_tokens) + ([0] * pad_count),
        "labels": list(token_ids) + ([-100] * pad_count),
    }


def build_token_blocks(args: argparse.Namespace,
                       tokenizer: Any,
                       load_dataset_fn: Any) -> tuple[list[dict[str, list[int]]], dict[str, int]]:
    data_dir = args.dataset_dir.resolve()
    split = args.dataset_split
    parquet_files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not parquet_files:
        fail(f"No local parquet files matched {data_dir / f'{split}-*.parquet'}")
    log(
        f"[data] loading local parquet split={split} "
        f"files={len(parquet_files)} dir={data_dir}"
    )
    dataset = load_dataset_fn(
        "parquet",
        data_files={split: [str(path) for path in parquet_files]},
        split=split,
    )

    dataset_rows = 0
    nonempty_rows = 0
    total_tokens = 0
    padding_tokens_added = 0
    blocks: list[dict[str, list[int]]] = []
    token_buffer: list[int] = []
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if eos_token_id is None:
            fail("Tokenizer must provide either pad_token_id or eos_token_id.")
        pad_token_id = eos_token_id

    for row in dataset:
        dataset_rows += 1
        text = extract_row_text(row)
        if not isinstance(text, str):
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if eos_token_id is not None:
            token_ids.append(eos_token_id)
        if not token_ids:
            continue
        nonempty_rows += 1
        total_tokens += len(token_ids)
        token_buffer.extend(token_ids)

        while len(token_buffer) >= args.seq_len:
            blocks.append(make_lm_block(token_buffer[:args.seq_len], args.seq_len, int(pad_token_id)))
            del token_buffer[:args.seq_len]

    if token_buffer:
        padding_tokens_added += args.seq_len - len(token_buffer)
        blocks.append(make_lm_block(token_buffer, args.seq_len, int(pad_token_id)))
        token_buffer = []

    if len(blocks) < args.batch_size:
        fail(
            f"WikiText-2 does not provide enough token blocks for one batch. "
            f"required>={args.batch_size}, available={len(blocks)}"
        )
    stats = {
        "dataset_rows": dataset_rows,
        "nonempty_rows": nonempty_rows,
        "parquet_files": len(parquet_files),
        "total_tokens_with_eos": total_tokens,
        "token_blocks": len(blocks),
        "tokens_used_for_blocks": total_tokens,
        "padding_tokens_added": padding_tokens_added,
        "remainder_tokens": 0,
    }
    log(
        f"[data] prepared rows={dataset_rows} nonempty_rows={nonempty_rows} "
        f"blocks={len(blocks)} seq_len={args.seq_len} padding_tokens_added={padding_tokens_added}"
    )
    return blocks, stats


def batches_per_epoch(blocks: list[dict[str, list[int]]], batch_size: int) -> int:
    count = (len(blocks) + batch_size - 1) // batch_size
    if count <= 0:
        fail(f"Not enough token blocks for one epoch: blocks={len(blocks)} batch_size={batch_size}")
    return count


def resolve_epoch_steps(available_steps: int, requested_steps: int) -> int:
    if requested_steps <= 0:
        fail(f"--steps-per-epoch must be positive, got {requested_steps}")
    return min(available_steps, requested_steps)


def build_batch_from_blocks(torch: Any,
                            blocks: list[dict[str, list[int]]],
                            batch_index: int,
                            batch_size: int,
                            seq_len: int,
                            pad_token_id: int,
                            device: Any) -> dict[str, Any]:
    start = batch_index * batch_size
    end = start + batch_size
    chunk = blocks[start:end]
    if not chunk:
        fail(f"Insufficient data blocks for batch_index={batch_index}, batch_size={batch_size}")
    if len(chunk) < batch_size:
        chunk = chunk + [make_lm_block([], seq_len, pad_token_id) for _ in range(batch_size - len(chunk))]
    input_ids = torch.tensor([item["input_ids"] for item in chunk], dtype=torch.long, device=device)
    attention_mask = torch.tensor([item["attention_mask"] for item in chunk], dtype=torch.long, device=device)
    labels = torch.tensor([item["labels"] for item in chunk], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def create_engine(args: argparse.Namespace, connector_mode: bool = False):
    torch, deepspeed, auto_model_cls, auto_tokenizer_cls, load_dataset_fn = load_runtime_modules()
    device = init_distributed(args, torch, deepspeed)
    seed_everything(args.seed, torch)
    if connector_mode:
        model = load_model_fast(args, torch, auto_model_cls)
    else:
        model = load_model(args, torch, auto_model_cls)
    tokenizer = load_tokenizer(args, auto_tokenizer_cls)
    ds_config = build_ds_config(args)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[parameter for parameter in model.parameters() if parameter.requires_grad],
        config=ds_config,
    )
    stats = {
        "parameter_count": parameter_count(model),
        "model_weight_bytes": model_weight_bytes(model),
    }
    return torch, engine, optimizer, device, tokenizer, load_dataset_fn, stats


def run_one_step(engine: Any, batch: dict[str, Any]) -> float:
    outputs = engine(**batch)
    loss = extract_loss(outputs)
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
    distributed_barrier(torch)
    sync_device(torch, engine.device)
    epoch_start = time.perf_counter()
    last_loss = 0.0
    for step_idx in range(steps_per_epoch):
        batch_index = step_idx
        batch = build_batch_from_blocks(
            torch,
            blocks,
            batch_index,
            batch_size,
            seq_len,
            pad_token_id,
            engine.device,
        )
        last_loss = run_one_step(engine, batch)
        if log_interval > 0 and ((step_idx + 1) % log_interval == 0 or step_idx + 1 == steps_per_epoch):
            log(
                f"[train] epoch={epoch_idx} "
                f"step={step_idx + 1}/{steps_per_epoch} "
                f"loss={last_loss:.6f}"
            )
    sync_device(torch, engine.device)
    distributed_barrier(torch)
    epoch_seconds = time.perf_counter() - epoch_start
    log(
        f"[train] epoch={epoch_idx} completed "
        f"epoch_seconds={epoch_seconds:.6f} "
        f"last_loss={last_loss:.6f}"
    )
    return max_across_ranks(torch, epoch_seconds, engine.device), last_loss


def file_size_bytes(path: Path, apparent: bool = False) -> int:
    stat_result = path.stat()
    if apparent:
        return int(stat_result.st_size)
    block_count = getattr(stat_result, "st_blocks", None)
    if block_count is None:
        return int(stat_result.st_size)
    return int(block_count * 512)


def path_size_bytes(path: Path, apparent: bool = False) -> int:
    if path.is_file():
        return file_size_bytes(path, apparent=apparent)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += file_size_bytes(child, apparent=apparent)
    return int(total)


def checkpoint_size_bytes(checkpoint_dir: Path, tag: str, apparent: bool = False) -> int:
    tag_dir = checkpoint_dir / tag
    target = tag_dir if tag_dir.exists() else checkpoint_dir
    return path_size_bytes(target, apparent=apparent)


def save_checkpoint(torch: Any,
                    engine: Any,
                    checkpoint_dir: Path,
                    tag: str,
                    client_state: dict[str, Any]) -> tuple[float, int, int]:
    distributed_barrier(torch)
    sync_device(torch, engine.device)
    save_start = time.perf_counter()
    engine.save_checkpoint(str(checkpoint_dir), tag=tag, client_state=client_state)
    sync_device(torch, engine.device)
    distributed_barrier(torch)
    save_seconds = time.perf_counter() - save_start
    save_seconds = max_across_ranks(torch, save_seconds, engine.device)
    size_bytes = checkpoint_size_bytes(checkpoint_dir, tag, apparent=False)
    apparent_size_bytes = checkpoint_size_bytes(checkpoint_dir, tag, apparent=True)
    return save_seconds, size_bytes, apparent_size_bytes


def maybe_write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")


def enrich_training_throughput_metrics(payload: dict[str, Any]) -> None:
    """Augment final JSON reports with steps/s and tokens/s (in-place)."""
    steps = payload.get("steps_per_epoch")
    secs = payload.get("epoch_train_seconds")
    if steps is not None and secs is not None:
        try:
            fs = float(secs)
            if fs > 0:
                payload["train_steps_per_second"] = float(steps) / fs
        except (TypeError, ValueError):
            pass
    rsteps = payload.get("resume_steps_per_epoch")
    rsecs = payload.get("resume_epoch_seconds")
    if rsteps is not None and rsecs is not None:
        try:
            fr = float(rsecs)
            if fr > 0:
                payload["resume_train_steps_per_second"] = float(rsteps) / fr
        except (TypeError, ValueError):
            pass
    bs = payload.get("batch_size")
    sl = payload.get("seq_len")
    if bs is not None and sl is not None:
        try:
            mult = int(bs) * int(sl)
            if "train_steps_per_second" in payload:
                payload["train_tokens_per_second"] = (
                    float(payload["train_steps_per_second"]) * mult
                )
            if "resume_train_steps_per_second" in payload:
                payload["resume_train_tokens_per_second"] = (
                    float(payload["resume_train_steps_per_second"]) * mult
                )
        except (TypeError, ValueError):
            pass


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def destroy_engine(torch: Any, engine: Any) -> None:
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def report_common_metadata(args: argparse.Namespace, stats: dict[str, Any]) -> dict[str, Any]:
    label = model_label_from_arg(args.model, args.model_label)
    return {
        "model": args.model,
        "model_label": label,
        "parameter_count": int(stats["parameter_count"]),
        "model_weight_bytes": int(stats["model_weight_bytes"]),
        "model_weight_gib": float(stats["model_weight_bytes"]) / (1024**3),
        "dtype": args.dtype,
        "zero_stage": args.zero_stage,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "configured_steps_per_epoch": int(args.steps_per_epoch),
        "dataset_name": DATASET_NAME,
        "dataset_subset": DATASET_SUBSET,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_split": args.dataset_split,
        "fault_epoch": FAULT_EPOCH,
        "resume_epochs_after_recovery": RESUME_EPOCHS,
        "log_interval": args.log_interval,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "state_file": str(args.state_file.resolve()),
        "report_file": str(args.report_file.resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_creator(args: argparse.Namespace) -> int:
    """Run initial training, save a native DeepSpeed checkpoint at the failure boundary, then inject the fault."""
    checkpoint_dir = args.checkpoint_dir.resolve()
    state_file = args.state_file.resolve()
    clear_directory(checkpoint_dir)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    if state_file.exists():
        state_file.unlink()

    torch, engine, _, _, tokenizer, load_dataset_fn, stats = create_engine(args)
    blocks, data_stats = build_token_blocks(args, tokenizer, load_dataset_fn)
    available_step_count = batches_per_epoch(blocks, args.batch_size)
    epoch_step_count = resolve_epoch_steps(available_step_count, args.steps_per_epoch)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        fail("Tokenizer must provide either pad_token_id or eos_token_id.")
    log(
        f"[creator] model={model_label_from_arg(args.model, args.model_label)} "
        f"steps_per_epoch={epoch_step_count} "
        f"available_steps_per_epoch={available_step_count} "
        f"dataset_rows={data_stats['dataset_rows']} "
        f"token_blocks={data_stats['token_blocks']}"
    )

    epoch_seconds = 0.0
    last_loss = 0.0
    try:
        epoch_seconds, last_loss = run_epoch(
            torch,
            engine,
            blocks,
            args.batch_size,
            args.seq_len,
            int(pad_token_id),
            epoch_step_count,
            FAULT_EPOCH,
            args.log_interval,
        )

        tag = f"epoch_{FAULT_EPOCH:06d}"
        client_state = {
            "train_epoch": FAULT_EPOCH,
            "last_loss": last_loss,
            "model_label": model_label_from_arg(args.model, args.model_label),
        }
        save_seconds, size_bytes, apparent_size_bytes = save_checkpoint(torch, engine, checkpoint_dir, tag, client_state)
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
            "checkpoint_tag": tag,
            "epoch_train_seconds": float(epoch_seconds),
            "last_loss": float(last_loss),
            "checkpoint_save_seconds": float(save_seconds),
            "checkpoint_size_bytes": int(size_bytes),
            "checkpoint_size_gib": float(size_bytes) / (1024**3),
            "checkpoint_apparent_size_bytes": int(apparent_size_bytes),
            "checkpoint_apparent_size_gib": float(apparent_size_bytes) / (1024**3),
        })
        maybe_write_json(state_file, payload)
        log(
            f"[creator] model={payload['model_label']} "
            f"save_seconds={payload['checkpoint_save_seconds']:.6f} "
            f"checkpoint_size_gib={payload['checkpoint_size_gib']:.6f}"
        )
        return CREATOR_FAILURE_EXIT_CODE
    finally:
        destroy_engine(torch, engine)
        destroy_process_group(torch)


def run_connector(args: argparse.Namespace) -> int:
    """Reload the DeepSpeed checkpoint, restore engine state, and resume training."""
    connector_entry_time = time.perf_counter()
    recovery_start_time = args.fault_start_time if args.fault_start_time is not None else connector_entry_time
    state = load_json(args.state_file.resolve())

    engine_init_start = time.perf_counter()
    torch, engine, _, _, tokenizer, load_dataset_fn, _ = create_engine(args, connector_mode=True)
    checkpoint_dir = Path(state["checkpoint_dir"]).resolve()
    tag = str(state["checkpoint_tag"])
    checkpoint_load_seconds = 0.0
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        fail("Tokenizer must provide either pad_token_id or eos_token_id.")

    try:
        engine_init_seconds = time.perf_counter() - engine_init_start
        engine_init_seconds = max_across_ranks(torch, engine_init_seconds, engine.device)

        distributed_barrier(torch)
        sync_device(torch, engine.device)
        load_start = time.perf_counter()
        load_path, client_state = engine.load_checkpoint(str(checkpoint_dir), tag=tag)
        sync_device(torch, engine.device)
        distributed_barrier(torch)
        checkpoint_load_seconds = time.perf_counter() - load_start
        checkpoint_load_seconds = max_across_ranks(torch, checkpoint_load_seconds, engine.device)
        if load_path is None or client_state is None:
            fail(f"Failed to load checkpoint {tag} from {checkpoint_dir}")
        torch.cuda.empty_cache()

        recovery_total_seconds = time.perf_counter() - recovery_start_time
        recovery_total_seconds = max_across_ranks(torch, recovery_total_seconds, engine.device)
        connector_other_seconds = max(0.0, recovery_total_seconds - engine_init_seconds - checkpoint_load_seconds)
        resumed_epoch = int(client_state.get("train_epoch", 0))

        blocks, data_stats = build_token_blocks(args, tokenizer, load_dataset_fn)
        available_step_count = batches_per_epoch(blocks, args.batch_size)
        epoch_step_count = resolve_epoch_steps(available_step_count, args.steps_per_epoch)
        resume_epoch_seconds = 0.0
        resume_last_loss = 0.0
        for resume_epoch_idx in range(1, RESUME_EPOCHS + 1):
            current_epoch = resumed_epoch + resume_epoch_idx
            resume_epoch_seconds, resume_last_loss = run_epoch(
                torch,
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
        destroy_engine(torch, engine)
        destroy_process_group(torch)

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
    enrich_training_throughput_metrics(payload)
    maybe_write_json(args.report_file.resolve(), payload)
    if rank0():
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
    log("[all] starting creator process")
    rc = subprocess.call(creator_cmd)
    if rc != CREATOR_FAILURE_EXIT_CODE:
        return rc

    fault_start_time = time.perf_counter()
    connector_cmd = build_subprocess_args(args, "connector")
    connector_cmd.extend(["--fault-start-time", f"{fault_start_time:.9f}"])
    log("[all] starting connector process")
    return subprocess.call(connector_cmd)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        fail("--batch-size must be positive.")
    if args.seq_len <= 0:
        fail("--seq-len must be positive.")
    if args.steps_per_epoch <= 0:
        fail("--steps-per-epoch must be positive.")
    if not args.dataset_dir.exists():
        fail(f"--dataset-dir does not exist: {args.dataset_dir}")


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
