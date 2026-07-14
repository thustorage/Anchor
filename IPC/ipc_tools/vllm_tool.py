import argparse
import importlib
import os
import subprocess
import sys
import time

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from ipc_socket import IPCSocket, SOCKET_PATH
from tool import (
    ALLOCING,
    IDLE,
    INIT,
    IPCServer,
    TensorFactoryInterceptor as _BaseTensorFactoryInterceptor,
    print_colored,
    tensor_size,
)




def _env_enabled() -> bool:
    value = os.environ.get("VLLM_IPC_TOOL", "")
    return value.lower() in {"1", "true", "yes", "on"}


TOOL = None
ENABLED = _env_enabled()
disguise_switch = ENABLED
disguise_kv = ENABLED

VLLM_MP_SOCKET_PATH = f"{SOCKET_PATH}_vllmmp"


def ipc_enabled() -> bool:
    return ENABLED


def set_enabled(enabled: bool) -> None:
    global ENABLED, disguise_switch, disguise_kv, TOOL
    ENABLED = enabled
    os.environ["VLLM_IPC_TOOL"] = "1" if enabled else "0"
    if enabled:
        disguise_switch = True
        disguise_kv = True
    else:
        disguise_switch = False
        disguise_kv = False
        TOOL = None


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


def socket_path_for_device(device, base_socket_path: str = VLLM_MP_SOCKET_PATH) -> str:
    return f"{base_socket_path}.gpu{resolve_physical_gpu_id(device)}"


def socket_path_for_gpu_id(gpu_id: str | int, base_socket_path: str = VLLM_MP_SOCKET_PATH) -> str:
    return f"{base_socket_path}.gpu{gpu_id}"


class VLLMMPIPCServer(IPCServer):
    """vLLM IPC daemon (memory owner).

    Exists independently of the vLLM worker (spawned by _ensure_connection with
    start_new_session), holding all real GPU tensors in tensor_named_store and the
    KV memory handle in vllm_kv_memory. It is unaffected when the worker crashes;
    the restarted worker remaps surviving memory back via reuse-by-name -- the
    foundation of the whole IPC recovery scheme. Inherits IPCServer and only
    overrides the command loop run().
    """

    def __init__(self, socket_path=VLLM_MP_SOCKET_PATH):
        super().__init__(socket_path=socket_path)

    def run(self):
        """Daemon main loop: listen and handle worker commands one at a time."""
        import traceback

        pid = os.getpid()
        print_colored(f"[VLLM-MP-Server] server process started, PID: {pid}, socket: {self.socket_path}")
        try:
            self.ipc.listen()
            while True:
                conn, _ = self.ipc.accept()
                cmd = None
                proto_type = None
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
                        print_colored(f"[Server] handshake complete, PID: {pid}, socket: {self.socket_path}, target device: {self.device}")
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
                        print_colored(
                            f"[Server]: Group {self.curr_alloc_name} OK! "
                            f"ALLOC:{alloc_cnt}, FREE:{free_cnt}, Reusing:{self.reusing}"
                        )
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
                            res = 0
                            for tensor_name in self.named_groups_to_tensor[req["name"]]:
                                res += tensor_size(self.tensor_named_store[tensor_name])
                            self.ipc.send(conn, {"size": res})

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
                    print(f"[Server] Error: type={type(exc).__name__} cmd={cmd!r} proto={proto_type!r} exc={exc!r}")
                    traceback.print_exc()
                    sys.stdout.flush()
                    sys.stderr.flush()
                finally:
                    conn.close()
        except KeyboardInterrupt:
            pass
        finally:
            self.ipc.close_server()


class TensorFactoryInterceptor(_BaseTensorFactoryInterceptor):
    """vLLM interceptor (client), running inside the worker process.

    On top of the base tool.TensorFactoryInterceptor it does one thing: change the
    socket path to per-GPU by physical GPU id (<base>.gpu<n>), so in multi-GPU runs
    each rank connects to its own card's daemon. Allocation interception, BFS
    identity resolution, set_ remap and other core logic all reuse the base class.
    """

    def __init__(self, target_device, socket_path=VLLM_MP_SOCKET_PATH, af=False):
        self.base_socket_path = socket_path
        resolved_socket_path = socket_path_for_device(target_device, base_socket_path=socket_path)
        os.environ["PYTORCH_IPC_SOCKET_PATH"] = resolved_socket_path
        super().__init__(target_device=target_device, socket_path=resolved_socket_path, af=af)

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
        print_colored(f"[VLLM-MP-Tool] starting child process (PID: {proc.pid})...")
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
        raise RuntimeError("VLLM MP server failed to start")


def initialize_tool(load_device, local_rank: int | None = None, *, af: bool = False):
    """Build and return the global singleton interceptor TOOL (multi-GPU entry point)."""
    global TOOL

    target_device = torch.device(load_device)
    if target_device.type == "cuda":
        resolved_index = target_device.index
        if resolved_index is None:
            resolved_index = local_rank if local_rank is not None else 0
        target_device = torch.device(target_device.type, resolved_index)
        torch.cuda.set_device(target_device)

    if TOOL is None:
        TOOL = TensorFactoryInterceptor(target_device=target_device, af=af)
    return TOOL


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-server", action="store_true")
    parser.add_argument("--socket-path", default=VLLM_MP_SOCKET_PATH)
    parsed_args = parser.parse_args()
    if parsed_args.run_server:
        VLLMMPIPCServer(socket_path=parsed_args.socket_path).run()
