#!/usr/bin/env python3
"""Shared flat-buffer packing / structure (de)serialization helpers for the in-memory checkpoint benchmark."""
from __future__ import annotations

from typing import Any

import torch

import ds_hf_checkpoint_bench as native


FLOAT32_SLOT_BYTES = 4
RAW_TENSOR_ALIGNMENT_BYTES = 8
FLAT_INDEX_KEY = "__pccheck_flat_index__"
TUPLE_KEY = "__pccheck_tuple__"


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


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if dtype is None:
        native.fail(f"Unsupported dtype in checkpoint metadata: {name}")
    return dtype


def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def tensor_byte_view(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "untyped_storage"):
        byte_tensor = torch.empty(0, dtype=torch.uint8, device=tensor.device)
        return byte_tensor.set_(
            tensor.untyped_storage(),
            tensor.storage_offset() * tensor.element_size(),
            (tensor.numel() * tensor.element_size(),),
            (1,),
        )
    try:
        return tensor.view(torch.uint8)
    except (AttributeError, RuntimeError, TypeError) as exc:
        native.fail(f"Unable to reinterpret tensor storage as raw bytes: {exc}")


def shape_numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def tensor_from_byte_view(byte_tensor: torch.Tensor,
                          dtype: torch.dtype,
                          shape: list[int]) -> torch.Tensor:
    element_size = torch.empty((), dtype=dtype).element_size()
    numel = shape_numel(shape)
    expected_bytes = numel * element_size
    if byte_tensor.numel() != expected_bytes:
        native.fail(
            "Raw checkpoint slice size does not match tensor metadata: "
            f"expected {expected_bytes} bytes, got {byte_tensor.numel()}."
        )

    if hasattr(byte_tensor, "untyped_storage"):
        storage_offset_bytes = int(byte_tensor.storage_offset())
        if storage_offset_bytes % element_size != 0:
            native.fail(
                "Raw checkpoint slice is not aligned for dtype reconstruction: "
                f"offset_bytes={storage_offset_bytes}, element_size={element_size}."
            )
        typed_tensor = torch.empty(0, dtype=dtype, device=byte_tensor.device)
        typed_tensor = typed_tensor.set_(
            byte_tensor.untyped_storage(),
            storage_offset_bytes // element_size,
            (numel,),
            (1,),
        )
        return typed_tensor.view(shape)

    try:
        return byte_tensor.view(dtype).view(shape)
    except (AttributeError, RuntimeError, TypeError) as exc:
        native.fail(f"Unable to rebuild tensor from raw bytes: {exc}")


def resolve_flat_tensor_size(aux_payload: dict[str, Any], flat_entries: list[dict[str, Any]]) -> int:
    stored_size = int(aux_payload.get("flat_total_size", 0) or 0)
    if stored_size > 0:
        return stored_size

    stored_bytes = int(aux_payload.get("flat_total_bytes", 0) or 0)
    if stored_bytes > 0:
        return max(1, (stored_bytes + FLOAT32_SLOT_BYTES - 1) // FLOAT32_SLOT_BYTES)

    total_slots = 0
    total_bytes = 0
    for entry in flat_entries:
        if "byte_offset" in entry and "num_bytes" in entry:
            total_bytes = max(total_bytes, int(entry["byte_offset"]) + int(entry["num_bytes"]))
            continue
        total_slots = max(total_slots, int(entry["offset"]) + int(entry["numel"]))

    if total_bytes > 0:
        return max(1, (total_bytes + FLOAT32_SLOT_BYTES - 1) // FLOAT32_SLOT_BYTES)
    return total_slots


def encode_structure(obj: Any, flat_sources: list[dict[str, Any]]) -> Any:
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            flat_sources.append(
                {
                    "tensor": obj.detach(),
                    "shape": list(obj.shape),
                    "numel": int(obj.numel()),
                    "element_size": int(obj.element_size()),
                    "dtype": dtype_name(obj.dtype),
                }
            )
            return {FLAT_INDEX_KEY: len(flat_sources) - 1}
        return clone_to_cpu(obj)
    if isinstance(obj, dict):
        return {key: encode_structure(value, flat_sources) for key, value in obj.items()}
    if isinstance(obj, list):
        return [encode_structure(value, flat_sources) for value in obj]
    if isinstance(obj, tuple):
        return {TUPLE_KEY: [encode_structure(value, flat_sources) for value in obj]}
    return obj


def decode_structure(obj: Any,
                     flat_tensor: torch.Tensor,
                     flat_entries: list[dict[str, Any]],
                     flat_tensor_bytes: torch.Tensor | None = None,
                     copy: bool = True) -> Any:
    """Rebuild the original tensor structure from the flat buffer (copy=True clones leaves; copy=False returns zero-copy views)."""
    if flat_tensor_bytes is None:
        flat_tensor_bytes = tensor_byte_view(flat_tensor)
    if isinstance(obj, dict):
        if FLAT_INDEX_KEY in obj:
            meta = flat_entries[int(obj[FLAT_INDEX_KEY])]
            if "byte_offset" in meta and "num_bytes" in meta:
                start = int(meta["byte_offset"])
                end = start + int(meta["num_bytes"])
                tensor = tensor_from_byte_view(
                    flat_tensor_bytes[start:end],
                    dtype_from_name(str(meta["dtype"])),
                    list(meta["shape"]),
                )
                return tensor.clone() if copy else tensor
            start = int(meta["offset"])
            end = start + int(meta["numel"])
            tensor = flat_tensor[start:end].view(meta["shape"]).to(dtype=dtype_from_name(str(meta["dtype"])))
            return tensor.clone() if copy else tensor
        if TUPLE_KEY in obj:
            return tuple(
                decode_structure(value, flat_tensor, flat_entries, flat_tensor_bytes, copy)
                for value in obj[TUPLE_KEY]
            )
        return {
            key: decode_structure(value, flat_tensor, flat_entries, flat_tensor_bytes, copy)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [
            decode_structure(value, flat_tensor, flat_entries, flat_tensor_bytes, copy)
            for value in obj
        ]
    return obj
