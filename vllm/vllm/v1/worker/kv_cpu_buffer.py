# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from multiprocessing import shared_memory

import torch

try:
    from multiprocessing import resource_tracker
except ImportError:
    resource_tracker = None


_KV_CPU_BUFFER_CACHE: dict[str, tuple[shared_memory.SharedMemory, torch.Tensor | None]] = {}


def _maybe_unregister_shared_memory(shm: shared_memory.SharedMemory) -> None:
    if resource_tracker is None:
        return
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass


def _maybe_pin_shared_memory(
    shm: shared_memory.SharedMemory, *, pin_memory: bool
) -> torch.Tensor | None:
    if not pin_memory or not torch.cuda.is_available():
        return None

    pinned_view = torch.frombuffer(shm.buf, dtype=torch.uint8, count=len(shm.buf))
    if pinned_view.numel() == 0 or pinned_view.is_pinned():
        return pinned_view

    try:
        torch.cuda.init()
    except Exception:
        pass

    result = int(
        torch.cuda.cudart().cudaHostRegister(
            pinned_view.data_ptr(),
            pinned_view.numel() * pinned_view.element_size(),
            0,
        )
    )
    if result != 0:
        print(
            f"[kv_cpu_buffer] cudaHostRegister failed (err={result}, "
            f"bytes={pinned_view.numel()}); using unpinned host memory.",
            flush=True,
        )
        return None
    return pinned_view


def _maybe_unpin_shared_memory(pinned_view: torch.Tensor | None) -> None:
    if pinned_view is None:
        return

    try:
        if pinned_view.is_pinned():
            torch.cuda.check_error(
                torch.cuda.cudart().cudaHostUnregister(pinned_view.data_ptr())
            )
    finally:
        del pinned_view


def prepare_kv_cpu_buffer(
    buffer_name: str, *, pin_memory: bool
) -> tuple[shared_memory.SharedMemory, torch.Tensor | None]:
    cached = _KV_CPU_BUFFER_CACHE.get(buffer_name)
    if cached is not None:
        shm, pinned_view = cached
        if pinned_view is None and pin_memory:
            pinned_view = _maybe_pin_shared_memory(shm, pin_memory=True)
            _KV_CPU_BUFFER_CACHE[buffer_name] = (shm, pinned_view)
        return _KV_CPU_BUFFER_CACHE[buffer_name]

    shm = shared_memory.SharedMemory(name=buffer_name, create=False)
    _maybe_unregister_shared_memory(shm)
    pinned_view = _maybe_pin_shared_memory(shm, pin_memory=pin_memory)
    _KV_CPU_BUFFER_CACHE[buffer_name] = (shm, pinned_view)
    return shm, pinned_view


def create_kv_cpu_buffer(size: int) -> shared_memory.SharedMemory:
    shm = shared_memory.SharedMemory(create=True, size=size)
    _KV_CPU_BUFFER_CACHE[shm.name] = (shm, None)
    return shm


def release_all_kv_cpu_buffers() -> None:
    while _KV_CPU_BUFFER_CACHE:
        _, (shm, pinned_view) = _KV_CPU_BUFFER_CACHE.popitem()
        _maybe_unpin_shared_memory(pinned_view)
        try:
            shm.close()
        except FileNotFoundError:
            pass
