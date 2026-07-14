import collections
import copy
import inspect
import logging
import os
import subprocess
import sys
import time
import types

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from ipc_socket import IPCSocket
from tool import (ALLOCING, IDLE, INIT, IPCServer, SOCKET_PATH, TensorFactoryInterceptor,
                  get_storage_id, print_colored, tensor_size)


DS_SOCKET_PATH = SOCKET_PATH
TOOL = None


def get_active_tool():
    return TOOL


class DSIPCServer(IPCServer):
    """DeepSpeed IPC daemon: memory owner plus ZeRO-3 consistency state machine
    (committed per-sub-group steps + a single-step rollback journal)."""

    def __init__(self, socket_path=DS_SOCKET_PATH):
        super().__init__(socket_path=socket_path)
        self.zero3_optimizer_steps = {}
        self.zero3_step_journals = {}

    @staticmethod
    def _zero3_tensor_full_name(group_name: str, tensor_name: str) -> str:
        return f"{group_name}:{tensor_name}"

    def _zero3_get_step_store(self, group_name: str):
        return self.zero3_optimizer_steps.setdefault(group_name, {})

    def _zero3_get_journal(self, group_name: str):
        return self.zero3_step_journals.setdefault(
            group_name,
            {
                "inflight": False,
                "next_sub_group": 0,
                "active_sub_group": None,
                "total_subgroups": 0,
                "loss_scale": None,
                "active_backup_tensors": {},
                "active_backup_step": None,
            },
        )

    @staticmethod
    def _zero3_public_journal_state(journal: dict) -> dict:
        return {
            "inflight": bool(journal.get("inflight", False)),
            "next_sub_group": int(journal.get("next_sub_group", 0)),
            "active_sub_group": journal.get("active_sub_group"),
            "total_subgroups": int(journal.get("total_subgroups", 0)),
            "loss_scale": journal.get("loss_scale"),
        }

    def run(self):
        """Daemon main loop: serve memory commands and ZeRO-3 consistency commands one at a time (exceptions are non-fatal)."""
        pid = os.getpid()
        print_colored(f"[DS-Server] server process started, PID: {pid}, socket: {self.socket_path}")
        try:
            self.ipc.listen()
            while True:
                conn, _ = self.ipc.accept()
                try:
                    proto_type, data = self.ipc.recv_auto(conn)
                    if proto_type == "cpp":
                        self._handle_cpp_request(conn, data)
                        continue
                    req = data
                    cmd = req.get("cmd")
                    if cmd == "INIT":
                        self.device = torch.device(req.get("device"))
                        torch.empty(0, device=self.device)
                        print_colored(f"[DS-Server] handshake complete, PID: {pid}, socket: {self.socket_path}, target device: {self.device}")
                        response = {
                            "status": "ok",
                            "server_pid": pid,
                            "kv_memory": self.vllm_kv_memory,
                        }
                        self.ipc.send(conn, response)
                        if self.state == INIT:
                            self.state = IDLE
                        else:
                            self.CHECK(IDLE)

                    elif cmd == "BEGIN_ALLOC":
                        self.active_id = {}
                        self.id_to_empty = {}
                        self.reusing = 0
                        self.CHECK(IDLE)
                        self.state = ALLOCING
                        self.curr_alloc_name = req["name"]
                        torch.cuda.memory.caching_allocator_enable(False)

                    elif cmd == "LAZY_BEGIN":
                        torch.cuda.memory.caching_allocator_enable(True)
                        self.CHECK(ALLOCING)
                        self.ipc.send(conn, {"status": "ok"})
                        self.lazying = True

                    elif cmd == "LAZY_END":
                        self.CHECK(ALLOCING)
                        self.state = IDLE
                        free_cnt = 0
                        alloc_cnt = 0
                        for alloc_id, state in self.active_id.items():
                            if state == 1:
                                free_cnt += 1
                            elif state == 2:
                                alloc_cnt += 1
                            else:
                                raise AssertionError(f"Unexpected active state {state} for {alloc_id}")
                        print_colored(
                            f"[DS-Server]: Group {self.curr_alloc_name} OK! "
                            f"ALLOC:{alloc_cnt}, FREE:{free_cnt}, Reusing:{self.reusing}")
                        self.ipc.send(conn, {"status": "ok"})
                        self.lazying = False

                    elif cmd == "EXIT":
                        exit()

                    elif cmd == "KV_MEMORY":
                        if self.vllm_kv_memory is None:
                            self.vllm_kv_memory = req["memory"]
                        self.ipc.send(conn, {"kv_memory": self.vllm_kv_memory})

                    elif cmd == "GET_SIZE":
                        if req["name"] not in self.named_groups_to_tensor:
                            self.ipc.send(conn, {"size": 0})
                        else:
                            total = 0
                            for tensor_name in self.named_groups_to_tensor[req["name"]]:
                                total += tensor_size(self.tensor_named_store[tensor_name])
                            self.ipc.send(conn, {"size": total})

                    elif cmd == "ZERO3_GET_STEP":
                        group_name = req["group"]
                        sub_group_id = int(req["sub_group_id"])
                        step_store = self._zero3_get_step_store(group_name)
                        if sub_group_id in step_store:
                            self.ipc.send(conn, {"found": True, "step": copy.deepcopy(step_store[sub_group_id])})
                        else:
                            self.ipc.send(conn, {"found": False})

                    elif cmd == "ZERO3_SET_STEP":
                        group_name = req["group"]
                        sub_group_id = int(req["sub_group_id"])
                        self._zero3_get_step_store(group_name)[sub_group_id] = copy.deepcopy(req.get("step"))
                        self.ipc.send(conn, {"status": "ok"})

                    elif cmd == "ZERO3_BEGIN_STEP":
                        group_name = req["group"]
                        journal = self._zero3_get_journal(group_name)
                        journal["inflight"] = True
                        journal["total_subgroups"] = int(req.get("total_subgroups", journal.get("total_subgroups", 0)))
                        journal["loss_scale"] = copy.deepcopy(req.get("loss_scale"))
                        if journal.get("active_sub_group") is None and journal.get("next_sub_group") is None:
                            journal["next_sub_group"] = 0
                        self.ipc.send(conn, {"status": "ok", **self._zero3_public_journal_state(journal)})

                    elif cmd == "ZERO3_BEGIN_SUBGROUP":
                        group_name = req["group"]
                        sub_group_id = int(req["sub_group_id"])
                        tensor_names = list(req.get("tensor_names", []))
                        journal = self._zero3_get_journal(group_name)
                        journal["inflight"] = True
                        journal["active_sub_group"] = sub_group_id
                        journal["next_sub_group"] = sub_group_id
                        backup_tensors = {}
                        for tensor_name in tensor_names:
                            full_name = self._zero3_tensor_full_name(group_name, tensor_name)
                            if full_name not in self.tensor_named_store:
                                raise KeyError(f"Missing tensor for ZeRO-3 subgroup backup: {full_name}")
                            backup_tensors[tensor_name] = self.tensor_named_store[full_name].clone()
                        if getattr(self.device, "type", None) == "cuda":
                            torch.cuda.synchronize(self.device)
                        journal["active_backup_tensors"] = backup_tensors
                        journal["active_backup_step"] = copy.deepcopy(req.get("step"))
                        self.ipc.send(conn, {"status": "ok", **self._zero3_public_journal_state(journal)})

                    elif cmd == "ZERO3_COMMIT_SUBGROUP":
                        group_name = req["group"]
                        sub_group_id = int(req["sub_group_id"])
                        next_sub_group = int(req.get("next_sub_group", sub_group_id + 1))
                        journal = self._zero3_get_journal(group_name)
                        self._zero3_get_step_store(group_name)[sub_group_id] = copy.deepcopy(req.get("step"))
                        journal["active_sub_group"] = None
                        journal["active_backup_tensors"] = {}
                        journal["active_backup_step"] = None
                        journal["next_sub_group"] = next_sub_group
                        self.ipc.send(conn, {"status": "ok", **self._zero3_public_journal_state(journal)})

                    elif cmd == "ZERO3_FINISH_STEP":
                        group_name = req["group"]
                        journal = self._zero3_get_journal(group_name)
                        journal["inflight"] = False
                        journal["active_sub_group"] = None
                        journal["active_backup_tensors"] = {}
                        journal["active_backup_step"] = None
                        journal["next_sub_group"] = 0
                        journal["loss_scale"] = None
                        self.ipc.send(conn, {"status": "ok", **self._zero3_public_journal_state(journal)})

                    elif cmd == "ZERO3_GET_JOURNAL":
                        group_name = req["group"]
                        journal = self._zero3_get_journal(group_name)
                        self.ipc.send(conn, {"status": "ok", **self._zero3_public_journal_state(journal)})

                    elif cmd == "ZERO3_ROLLBACK_ACTIVE_SUBGROUP":
                        group_name = req["group"]
                        journal = self._zero3_get_journal(group_name)
                        active_sub_group = journal.get("active_sub_group")
                        if active_sub_group is not None:
                            for tensor_name, backup_tensor in journal.get("active_backup_tensors", {}).items():
                                full_name = self._zero3_tensor_full_name(group_name, tensor_name)
                                if full_name not in self.tensor_named_store:
                                    raise KeyError(f"Missing tensor for ZeRO-3 subgroup restore: {full_name}")
                                self.tensor_named_store[full_name].copy_(backup_tensor)
                            if getattr(self.device, "type", None) == "cuda":
                                torch.cuda.synchronize(self.device)
                            if journal.get("active_backup_step", None) is not None:
                                self._zero3_get_step_store(group_name)[active_sub_group] = copy.deepcopy(
                                    journal["active_backup_step"]
                                )
                            journal["next_sub_group"] = active_sub_group
                            journal["active_sub_group"] = None
                            journal["active_backup_tensors"] = {}
                            journal["active_backup_step"] = None
                        self.ipc.send(conn, {"status": "ok", **self._zero3_public_journal_state(journal)})

                    elif cmd == "ALLOC":
                        self.CHECK(ALLOCING)

                        func_name = req["func"]
                        name = self.curr_alloc_name + ":" + req["name"]
                        args = req["args"]
                        kwargs = req["kwargs"]
                        alloc_id = req["id"]

                        assert kwargs["device"] == self.device
                        assert hasattr(torch, func_name)

                        if name in self.tensor_named_store:
                            real_tensor = self.tensor_named_store[name]
                            self.reusing += 1
                            self.ipc.send(conn, {"status": "ok", "handle": real_tensor})
                        else:
                            func = getattr(torch, func_name)
                            real_tensor = func(*args, **kwargs)
                            self.tensor_named_store[name] = real_tensor
                            if self.curr_alloc_name not in self.named_groups_to_tensor:
                                self.named_groups_to_tensor[self.curr_alloc_name] = []
                            self.named_groups_to_tensor[self.curr_alloc_name].append(name)
                            self.ipc.send(conn, {"status": "ok", "handle": real_tensor})
                        self.active_id[alloc_id] = 2

                except Exception as exc:
                    print(f"[DS-Server] Error: {exc}")
                    try:
                        self.ipc.send(conn, {"status": "error", "message": str(exc)})
                    except Exception:
                        pass
                finally:
                    conn.close()
        except KeyboardInterrupt:
            pass
        finally:
            self.ipc.close_server()


class DSTensorFactoryInterceptor(TensorFactoryInterceptor):
    """DeepSpeed interceptor (client) in the training worker: adds zero3_* RPC helpers
    and named materialization on top of the base intercept/materialize; single-GPU (default socket)."""

    def __init__(self, target_device, socket_path=DS_SOCKET_PATH, af=False):
        global TOOL
        self.materialized_tensor_names = {}
        self._scope_declared_name = None
        super().__init__(target_device=target_device, socket_path=socket_path, af=af)
        TOOL = self

    def _lazy_begin(self):
        try:
            ipc = IPCSocket(self.socket_path)
            sock = ipc.connect()
            ipc.send(sock, {"cmd": "LAZY_BEGIN"})
            ipc.recv(sock)
            sock.close()
        except Exception as exc:
            print(f"Exit failed: {exc}")

    def _lazy_end(self):
        try:
            ipc = IPCSocket(self.socket_path)
            sock = ipc.connect()
            ipc.send(sock, {"cmd": "LAZY_END"})
            ipc.recv(sock)
            sock.close()
        except Exception as exc:
            print(f"Exit failed: {exc}")

    def _rpc(self, payload: dict):
        ipc = IPCSocket(self.socket_path)
        sock = ipc.connect()
        try:
            ipc.send(sock, payload)
            return ipc.recv(sock)
        finally:
            sock.close()

    def zero3_get_optimizer_step(self, group_name: str, sub_group_id: int):
        """Read a sub-group's committed optimizer step metadata; None if never committed."""
        response = self._rpc({"cmd": "ZERO3_GET_STEP", "group": group_name, "sub_group_id": int(sub_group_id)})
        if isinstance(response, dict) and response.get("found"):
            return response.get("step")
        return None

    def zero3_set_optimizer_step(self, group_name: str, sub_group_id: int, step_value):
        """Directly write a sub-group's step metadata (commit is the usual path; this is for explicit correction/backfill)."""
        return self._rpc({
            "cmd": "ZERO3_SET_STEP",
            "group": group_name,
            "sub_group_id": int(sub_group_id),
            "step": step_value,
        })

    def zero3_begin_step(self, group_name: str, total_subgroups: int, loss_scale):
        """Open an optimizer step: daemon sets inflight=True and records sub-group count / loss_scale; returns the public journal view."""
        return self._rpc({
            "cmd": "ZERO3_BEGIN_STEP",
            "group": group_name,
            "total_subgroups": int(total_subgroups),
            "loss_scale": loss_scale,
        })

    def zero3_begin_subgroup(self, group_name: str, sub_group_id: int, tensor_names: list[str], step_value):
        """Register an undo backup before updating a sub-group: daemon clones its state tensors (by tensor_names) and sets active_sub_group."""
        return self._rpc({
            "cmd": "ZERO3_BEGIN_SUBGROUP",
            "group": group_name,
            "sub_group_id": int(sub_group_id),
            "tensor_names": list(tensor_names),
            "step": step_value,
        })

    def zero3_commit_subgroup(self, group_name: str, sub_group_id: int, next_sub_group: int, step_value):
        """Commit a sub-group: daemon persists step metadata, drops the undo backup, and advances the cursor to next_sub_group."""
        return self._rpc({
            "cmd": "ZERO3_COMMIT_SUBGROUP",
            "group": group_name,
            "sub_group_id": int(sub_group_id),
            "next_sub_group": int(next_sub_group),
            "step": step_value,
        })

    def zero3_finish_step(self, group_name: str):
        """Finish the whole optimizer step: daemon clears inflight and the cursor, marking the completed-step boundary."""
        return self._rpc({"cmd": "ZERO3_FINISH_STEP", "group": group_name})

    def zero3_get_step_journal(self, group_name: str):
        """Read the current public journal view; recovery uses it to tell which step/sub-group the last run crashed in."""
        return self._rpc({"cmd": "ZERO3_GET_JOURNAL", "group": group_name})

    def zero3_rollback_active_subgroup(self, group_name: str):
        """Roll back the currently-updating sub-group in place from its undo backup and rewind the cursor to it (recovery)."""
        return self._rpc({"cmd": "ZERO3_ROLLBACK_ACTIVE_SUBGROUP", "group": group_name})

    def enter(self, group_name: str, tensor_name: str | None = None):
        """Open a managed scope; optional tensor_name gives the scope's single tensor a deterministic daemon name (else BFS auto-naming)."""
        self._scope_declared_name = tensor_name
        super().enter(group_name)

    def _finalize_group(self, caller_frame_depth: int = 1,
                        override_name: str | None = None):
        self.__exit__(None, None, None)
        self._lazy_begin()

        materialized_count = 0
        frame = getattr(self, "_scope_root_frame", None)
        if frame is None:
            frame = inspect.currentframe()
            for _ in range(caller_frame_depth):
                if frame is None:
                    break
                frame = frame.f_back
        if frame is None:
            raise RuntimeError("Could not resolve caller frame for IPC tensor materialization")
        roots = {**frame.f_globals, **frame.f_locals}
        visited_ids = {id(self)}

        queue = collections.deque()
        for name, root_obj in roots.items():
            if name.startswith("__"):
                continue
            queue.append((root_obj, name))

        while queue:
            obj, path = queue.popleft()
            obj_id = id(obj)
            if obj_id in visited_ids:
                continue
            visited_ids.add(obj_id)

            if isinstance(obj, torch.Tensor):
                sid = get_storage_id(obj)
                if sid in self.captured_tensors:
                    materialized_count += 1
                    if override_name is not None and materialized_count > 1:
                        raise RuntimeError(
                            "enter(tensor_name=...) expects exactly one captured tensor in the "
                            f"scope, but found more; use name() per tensor instead. name={override_name}")
                    op_name, args, kwargs, alloc_id = self.captured_tensors[sid]
                    full_name = override_name if override_name is not None else (path if path else f"auto_found_{sid}")

                    ipc_tensor = self._ipc_allocate(op_name, args, kwargs, full_name, alloc_id)

                    req_grad = obj.requires_grad
                    with torch.no_grad():
                        obj.set_(ipc_tensor)
                    obj.requires_grad = req_grad
                    self.materialized_tensor_names[id(obj)] = f"{self.current_group}:{full_name}"

                    if sid in self.cpu_cached_tensors:
                        obj.copy_(self.cpu_cached_tensors[sid], non_blocking=True)
                continue

            if isinstance(
                obj,
                (str, int, float, bool, bytes, type(None), types.ModuleType, types.FunctionType, type),
            ):
                continue

            if isinstance(obj, collections.abc.Mapping):
                for key, value in obj.items():
                    if isinstance(key, str):
                        queue.append((value, f"{path}['{key}']"))
            elif isinstance(obj, collections.abc.Iterable):
                try:
                    for index, value in enumerate(obj):
                        queue.append((value, f"{path}[{index}]"))
                except Exception:
                    pass
            else:
                if hasattr(obj, "__dict__"):
                    for key, value in obj.__dict__.items():
                        if not key.startswith("__"):
                            queue.append((value, f"{path}.{key}"))
                if hasattr(obj, "__slots__"):
                    slots = obj.__slots__
                    if slots is not None:
                        if isinstance(slots, str):
                            slots = [slots]
                        for slot in slots:
                            if hasattr(obj, slot):
                                queue.append((getattr(obj, slot), f"{path}.{slot}"))

        print_colored(f">>> [DS-IPC] Auto-scan finished. Materialized {materialized_count} tensors.")
        del frame
        self.captured_tensors.clear()
        self.cpu_cached_tensors.clear()
        self._lazy_end()

    def exit(self):
        """Close the scope and materialize: under the declared name if enter gave one, else BFS auto access-path naming."""
        if self._scope_declared_name is not None:
            self._finalize_group(caller_frame_depth=2,
                                 override_name=self._scope_declared_name)
        else:
            self._finalize_group(caller_frame_depth=2)

    def resolve_tensor_name(self, tensor):
        """Look up a tensor's full name "group:access_path" in the daemon; None if never materialized."""
        return self.materialized_tensor_names.get(id(tensor))

    def _ensure_connection(self):
        ipc = IPCSocket(self.socket_path)
        try:
            sock = ipc.connect()
            self.is_server_creator = False
            self._handshake(ipc, sock)
            sock.close()
            return
        except (ConnectionRefusedError, FileNotFoundError):
            pass

        current_script = os.path.abspath(__file__)
        proc = subprocess.Popen([sys.executable, current_script, "--run-server"], start_new_session=True)
        print_colored(f"[DS-Tool] starting child process (PID: {proc.pid})...")
        self.is_server_creator = True

        start_time = time.time()
        while time.time() - start_time < 10:
            try:
                sock = ipc.connect()
                self._handshake(ipc, sock)
                sock.close()
                return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("DS server failed to start")


DS_MP_SOCKET_PATH = f"{SOCKET_PATH}_dsmp"


def resolve_physical_gpu_id(device) -> str:
    device = torch.device(device)
    if device.type != "cuda":
        return "cpu"

    logical_index = device.index if device.index is not None else torch.cuda.current_device()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_parts = [part.strip() for part in visible.split(",") if part.strip()]
    if visible_parts and logical_index < len(visible_parts):
        return visible_parts[logical_index]
    return str(logical_index)


def socket_path_for_device(device, base_socket_path: str = DS_MP_SOCKET_PATH) -> str:
    return f"{base_socket_path}.gpu{resolve_physical_gpu_id(device)}"


def socket_path_for_gpu_id(gpu_id: str | int, base_socket_path: str = DS_MP_SOCKET_PATH) -> str:
    return f"{base_socket_path}.gpu{gpu_id}"


class DSMPIPCServer(DSIPCServer):
    """Multi-GPU daemon: identical logic to DSIPCServer, only it listens on a per-GPU socket (one per card)."""

    def __init__(self, socket_path=DS_MP_SOCKET_PATH):
        super().__init__(socket_path=socket_path)


class DSMPTensorFactoryInterceptor(DSTensorFactoryInterceptor):
    """Multi-GPU interceptor: same as the single-GPU version but uses a per-GPU socket so each rank connects to its own card's daemon."""

    def __init__(self, target_device, socket_path=DS_MP_SOCKET_PATH, af=False):
        global TOOL
        self.base_socket_path = socket_path
        resolved_socket_path = socket_path_for_device(target_device, base_socket_path=socket_path)
        os.environ["PYTORCH_IPC_SOCKET_PATH"] = resolved_socket_path
        super().__init__(target_device=target_device, socket_path=resolved_socket_path, af=af)
        TOOL = self

    def _ensure_connection(self):
        ipc = IPCSocket(self.socket_path)
        try:
            sock = ipc.connect()
            self.is_server_creator = False
            self._handshake(ipc, sock)
            sock.close()
            return
        except (ConnectionRefusedError, FileNotFoundError):
            pass

        current_script = os.path.abspath(__file__)
        proc = subprocess.Popen(
            [sys.executable, current_script, "--run-server", "--socket-path", self.socket_path],
            start_new_session=True,
        )
        print_colored(f"[DS-MP-Tool] starting child process (PID: {proc.pid})...")
        self.is_server_creator = True

        start_time = time.time()
        while time.time() - start_time < 10:
            try:
                sock = ipc.connect()
                self._handshake(ipc, sock)
                sock.close()
                return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("DS MP server failed to start")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run-server":
        socket_path = DS_SOCKET_PATH
        if "--socket-path" in sys.argv:
            _idx = sys.argv.index("--socket-path")
            if _idx + 1 < len(sys.argv):
                socket_path = sys.argv[_idx + 1]
        DSIPCServer(socket_path=socket_path).run()
