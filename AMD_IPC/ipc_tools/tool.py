import os
import sys
import time
from tokenize import group
import torch
import functools
import subprocess
import gc
import uuid
import inspect
import types
import collections
import struct
import ctypes

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from ipc_socket import IPCSocket, SOCKET_PATH, CPP_MAGIC

INIT = 0
ALLOCING = 1
IDLE = 2

def tensor_size(real_tensor):
    num_elements = real_tensor.numel()
    element_size = real_tensor.element_size()
    total_bytes = num_elements * element_size
    return total_bytes

def dict_size(dict_map):
    total_size = 0
    for t_id, tensor in dict_map.items(): 
        total_size += tensor_size(tensor)
    return total_size

def _ipc_verbose_enabled() -> bool:
    return os.environ.get("IPC_TOOL_VERBOSE", "").lower() in {"1", "true", "yes", "on"}


def print_colored(text, color="magenta"):
    if not _ipc_verbose_enabled():
        return
    colors = {
        "black": "30",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
        "bright_red": "91",
        "bright_green": "92",
        "bright_yellow": "93",
        "bright_blue": "94",
        "bright_magenta": "95",
        "bright_cyan": "96",
    }
    color_code = colors.get(color, "32")
    print(f"\033[{color_code}m{text}\033[0m")

CPP_CMD_GENERIC = 0x01
CPP_CMD_STRIDED = 0x02
CPP_CMD_FREE    = 0x03
CPP_STATUS_OK   = 0x00
CPP_STATUS_ERR  = 0x01

def _get_cuda_ipc_handle(data_ptr):
    handle = ctypes.create_string_buffer(64)
    if getattr(torch.version, "hip", None) is not None:
        hip_rt = ctypes.CDLL("libamdhip64.so")
        hip_rt.hipIpcGetMemHandle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        hip_rt.hipIpcGetMemHandle.restype = ctypes.c_int
        ret = hip_rt.hipIpcGetMemHandle(handle, ctypes.c_void_p(data_ptr))
        assert ret == 0, f"hipIpcGetMemHandle failed with error code {ret}"
    else:
        cuda_rt = ctypes.CDLL("libcudart.so")
        cuda_rt.cudaIpcGetMemHandle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cuda_rt.cudaIpcGetMemHandle.restype = ctypes.c_int
        ret = cuda_rt.cudaIpcGetMemHandle(handle, ctypes.c_void_p(data_ptr))
        assert ret == 0, f"cudaIpcGetMemHandle failed with error code {ret}"
    return handle.raw

class IPCServer:
    def __init__(self, socket_path=SOCKET_PATH):
        self.socket_path = socket_path
        self.ipc = IPCSocket(self.socket_path)

        self.tensor_named_store = {}

        self.named_groups_to_tensor = {}

        self.id_to_empty = {}
        self.active_id = {}

        self.device = None
        assert torch.cuda.is_available(), "IPC allocation only supports GPU tensors"
        self.state = INIT
        self.curr_alloc_name = None
        self.reusing = 0
        self.lazying = False

        self.vllm_kv_memory = None

    def CHECK(self, state):
        if self.state != state:
            print(f"\n[FATAL SERVER CRASH] state check failed!", file=sys.stderr)
            print(f"Current State: {self.state}", file=sys.stderr)
            print(f"Expected State: {state}", file=sys.stderr)
            import traceback
            print("Traceback (most recent call last):", file=sys.stderr)
            traceback.print_stack(file=sys.stderr)
            
            sys.stderr.flush()
            sys.stdout.flush()
            
            os._exit(1) 
    
    def run(self):
        """Run the server loop, dispatching incoming IPC requests."""
        pid = os.getpid()
        print_colored(f"[Server-Process] server process started, PID: {pid}")
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
                        print_colored(f"[Server] handshake complete, target device: {self.device}")
                        response = {
                            "status": "ok",
                            "server_pid": pid,
                            "kv_memory": self.vllm_kv_memory
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
                        for id,state in self.active_id.items():
                            if state == 1:
                                free_cnt += 1
                            elif state == 2:
                                alloc_cnt += 1
                        print_colored(f"[Server]: Group {self.curr_alloc_name} OK! ALLOC:{alloc_cnt}, FREE:{free_cnt}, Reusing:{self.reusing}")
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
                        name = self.curr_alloc_name +":" + req["name"]
                        args = req["args"]
                        kwargs = req["kwargs"]
                        id = req["id"]

                        assert kwargs["device"] ==  self.device
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
                        self.active_id[id] = 2

                except Exception as e:
                    print(f"[Server] Error: {e}")
                finally:
                    conn.close()
        except KeyboardInterrupt:
            pass
        finally:
            self.ipc.close_server()

    def _handle_cpp_request(self, conn, raw_data):
        try:
            p = 0
            magic = raw_data[p]; p += 1
            cmd = raw_data[p]; p += 1
            assert magic == CPP_MAGIC

            if cmd == CPP_CMD_FREE:
                alloc_id = struct.unpack(">Q", raw_data[p:p+8])[0]; 
                p += 8
                if self.lazying == False:
                    self.active_id[alloc_id] = 1
                return

            alloc_id = struct.unpack(">Q", raw_data[p:p+8])[0]; p += 8
            ndim = struct.unpack(">I", raw_data[p:p+4])[0]; p += 4
            sizes = []
            for _ in range(ndim):
                sizes.append(struct.unpack(">q", raw_data[p:p+8])[0]); p += 8
            
            strides = []
            if cmd == CPP_CMD_STRIDED:
                for _ in range(ndim):
                    strides.append(struct.unpack(">q", raw_data[p:p+8])[0]); p += 8

            scalar_type_id = raw_data[p]; p += 1
            device_index = struct.unpack(">H", raw_data[p:p+2])[0]; p += 2
            size_bytes = struct.unpack(">Q", raw_data[p:p+8])[0]; p += 8

            dtype = self._scalar_type_to_torch_dtype(scalar_type_id)
            device = torch.device("cuda", device_index)

            assert self.device is not None

            if cmd == CPP_CMD_STRIDED:
                empty_tensor = torch.empty_strided(sizes, strides, dtype=dtype, device=device)
            else:
                empty_tensor = torch.empty(sizes, dtype=dtype, device=device)

            self.active_id[alloc_id] = 0
            self.id_to_empty[alloc_id] = empty_tensor
 
            data_ptr = empty_tensor.data_ptr()
            ipc_handle_bytes = _get_cuda_ipc_handle(data_ptr)

            resp = bytearray()
            resp.append(CPP_MAGIC)
            resp.append(CPP_STATUS_OK)
            resp.extend(ipc_handle_bytes)
            resp.extend(struct.pack(">Q", alloc_id))
            resp.extend(struct.pack(">Q", size_bytes))

            self.ipc.send_raw(conn, bytes(resp))

        except Exception as e:
            print(f"[Server-CPP] Error handling CPP request: {e}")
    @staticmethod
    def _scalar_type_to_torch_dtype(scalar_type_id):
        mapping = {
            0: torch.uint8,
            1: torch.int8,
            2: torch.int16,
            3: torch.int32,
            4: torch.int64,
            5: torch.float16,
            6: torch.float32,
            7: torch.float64,
            8: torch.complex32,
            9: torch.complex64,
            10: torch.complex128,
            11: torch.bool,
            15: torch.bfloat16,
        }
        assert scalar_type_id in mapping, f"Unsupported ScalarType id: {scalar_type_id}"
        return mapping[scalar_type_id]

from torch.utils._python_dispatch import TorchDispatchMode
import atexit
def get_storage_id(tensor):
    try:
        storage = tensor.untyped_storage()
        assert hasattr(storage, '_cdata')
        return storage._cdata
    except Exception:
        return id(tensor)


TARGET_OPS = {
    "shape_factory": {
        'empty', 'zeros', 'ones',
        'rand', 'randn', 'randint'
    },
    "like_factory": {
        'empty_like', 'zeros_like', 'ones_like',
        'rand_like', 'randn_like', 'randint_like'
    },
    "data_copy": {
        'lift_fresh'
    }
}
class TensorFactoryInterceptor(TorchDispatchMode):
    
    def __init__(self, target_device, socket_path=SOCKET_PATH,af = False):
        self.socket_path = socket_path
        self.target_device = torch.device(target_device.type,torch.cuda.current_device())
        self.is_server_creator = False 
        self.server_pid = None
        self.af = af

        self.cpp_alloc_id = 1

        self.current_group = None 
        self._bypass = False 
        self._ensure_connection()
        self.captured_tensors = {}
        self.cpu_cached_tensors = {}    
        atexit.register(self._on_script_exit)
    
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Intercept tensor factory ops and return lazy placeholder tensors."""
        if kwargs is None:
            kwargs = {}

        if self._bypass or torch._dynamo.is_compiling():
            return func(*args, **kwargs)

        op_name = func.overloadpacket.__name__
        op_category = None
        for category, ops in TARGET_OPS.items():
            if op_name in ops:
                op_category = category
                break

        if not op_category:
            return func(*args, **kwargs)

        target_size = None
        target_dtype = kwargs.get("dtype", None)
        cpu_cache_tensor = None 

        if op_category == "shape_factory":
            target_size = next((a for a in args if isinstance(a, (tuple, list, torch.Size))), args[0] if args else tuple())
            if target_dtype is None: target_dtype = torch.get_default_dtype()

            req_device = kwargs.get("device", self.target_device)
            if isinstance(req_device, str):
                req_device = torch.device(req_device)
            if req_device.index == None:
                req_device = torch.device(req_device.type,torch.cuda.current_device())
            if req_device != self.target_device:
                return func(*args, **kwargs)
            kwargs["device"] = req_device
            if "dtype" not in kwargs:
                kwargs["dtype"] = target_dtype

        elif op_category == "like_factory":
            source_tensor = args[0]
            req_device = kwargs.get("device", source_tensor.device)
            if isinstance(req_device, str):
                req_device = torch.device(req_device)
            if req_device.index == None:
                req_device = torch.device(req_device.type,torch.cuda.current_device())
            if req_device != self.target_device:
                return func(*args, **kwargs)
            kwargs["device"] = req_device

            target_size = source_tensor.shape
            if target_dtype is None: target_dtype = source_tensor.dtype
            if "dtype" not in kwargs:
                kwargs["dtype"] = target_dtype
            
            op_name = op_name.replace('_like', '')
            if "dtype" not in kwargs: kwargs["dtype"] = source_tensor.dtype
            if "layout" not in kwargs: kwargs["layout"] = source_tensor.layout
            if "requires_grad" not in kwargs: kwargs["requires_grad"] = source_tensor.requires_grad
            
            if op_name == "randint":
                args = args[1:] + (target_size,) 
            else:
                args = (target_size,) + args[1:]
        
        else:
            source_tensor = args[0]     
            req_device = kwargs.get("device", source_tensor.device)
            if isinstance(req_device, str):
                req_device = torch.device(req_device)
            if req_device.index == None:
                req_device = torch.device(req_device.type,torch.cuda.current_device())
            if req_device != self.target_device:
                return func(*args, **kwargs)
            kwargs["device"] = req_device  

            target_size = source_tensor.shape
            if target_dtype is None: target_dtype = source_tensor.dtype
            if "dtype" not in kwargs:
                kwargs["dtype"] = target_dtype
            self._bypass = True
            cpu_cache_tensor = source_tensor.detach().cpu().pin_memory()
            self._bypass = False

            if "dtype" not in kwargs: kwargs["dtype"] = source_tensor.dtype
            if "layout" not in kwargs: kwargs["layout"] = source_tensor.layout
            if "requires_grad" not in kwargs: kwargs["requires_grad"] = source_tensor.requires_grad
            args = (target_size,) + args[1:]
            op_name = "zeros"
        
        self._bypass = True

        os.environ["PYTORCH_IPC_ALLOC"] = "1"
        real_tiny_storage = torch.empty(1, dtype=target_dtype, device=req_device)
        os.environ["PYTORCH_IPC_ALLOC"] = "0"
        zero_strides = [0] * len(target_size)
        local_tensor = real_tiny_storage.as_strided(target_size, zero_strides)

        self._bypass = False

        self.captured_tensors[get_storage_id(local_tensor)] = (op_name, args, kwargs,self.cpp_alloc_id)
        self.cpp_alloc_id += 1

        if cpu_cache_tensor is not None:
            self.cpu_cached_tensors[get_storage_id(local_tensor)] = cpu_cache_tensor
        return local_tensor

    def _ipc_allocate(self, op_name, args, kwargs,tensor_name,alloc_id):
        self._bypass = True

        ipc = IPCSocket(self.socket_path)
        s = ipc.connect()
        assert tensor_name != None
        ipc.send(s, {
            "cmd": "ALLOC",
            "func": op_name, 
            "args": args, 
            "kwargs": kwargs, 
            "name": tensor_name,
            "id": alloc_id
        })
        
        resp = ipc.recv(s)
        s.close()

        if isinstance(resp, dict) and "handle" in resp:
            tensor = resp["handle"] 
            self._bypass = False
            return tensor
        
        assert False,"IPC ALLOC ERROR!"
    
    def _on_script_exit(self):
        print_colored(f"\n[IPC Hook] script finished, entering resource-release phase...")
        if self.af:
            try:
                ipc = IPCSocket(self.socket_path)
                s = ipc.connect()    
                ipc.send(s, {"cmd": "EXIT"})
                s.close()
            except Exception as e:
                print(f"Exit failed: {e}")
          
    def scope(self, *args, **kwargs):
        """Return a context manager that enters/exits a managed allocation scope."""
        tool = self

        class _ManagedScope:
            def __enter__(self_inner):
                """Enter the scope and record the caller's frame."""
                tool._scope_root_frame = sys._getframe(1)
                tool.enter(*args, **kwargs)
                return tool

            def __exit__(self_inner, exc_type, exc_val, exc_tb):
                """Exit the scope and materialize allocations."""
                tool.exit()
                tool._scope_root_frame = None
                return False

        return _ManagedScope()

    def enter(self, name: str):
        """Enter a managed allocation scope and begin allocation on the daemon."""
        self.current_group = name

        role = "Creator" if self.is_server_creator else "Connector"
        print_colored(f">>> [IPC] {self.current_group} Interceptor Activated. Role: {role}, PID: {self.server_pid}")
        
        self.__enter__()
        try:
            ipc = IPCSocket(self.socket_path)
            s = ipc.connect()    
            ipc.send(s, {"cmd": "BEGIN_ALLOC","name":name})
            s.close()
        except Exception as e:
            print(f"Enter failed: {e}")

    def exit(self):
        """Close the scope and materialize placeholder tensors into IPC memory."""
        self.__exit__(None, None, None)

        try:
            ipc = IPCSocket(self.socket_path)
            s = ipc.connect()    
            ipc.send(s, {"cmd": "LAZY_BEGIN"})
            resp = ipc.recv(s)
            s.close()
        except Exception as e:
            print(f"Exit failed: {e}")

        print_colored(f">>> [IPC] Lazy alloc from IPC...")
        materialized_count = 0
        frame = getattr(self, "_scope_root_frame", None)
        if frame is None:
            frame = inspect.currentframe().f_back
        roots = {**frame.f_globals, **frame.f_locals}
        visited_ids = set()
        visited_ids.add(id(self)) 
        print_colored(f">>> [IPC] Scanning roots: {list(roots.keys())[:10]} ...")
        queue = collections.deque()
        for name, root_obj in roots.items():
            if name.startswith("__"): continue
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
                    op_name, args, kwargs, alloc_id = self.captured_tensors[sid]
                    full_name = path if path else f"auto_found_{sid}"

                    ipc_tensor = self._ipc_allocate(op_name, args, kwargs, full_name, alloc_id)

                    req_grad = obj.requires_grad
                    with torch.no_grad():
                        obj.set_(ipc_tensor)
                    obj.requires_grad = req_grad

                    if sid in self.cpu_cached_tensors:
                        obj.copy_(self.cpu_cached_tensors[sid], non_blocking=True)
                continue

            if isinstance(obj, (str, int, float, bool, bytes, type(None), types.ModuleType, types.FunctionType, type)):
                continue

            if isinstance(obj, collections.abc.Mapping):
                for key, value in obj.items():
                    if isinstance(key, str):
                        queue.append((value, f"{path}['{key}']"))

            elif isinstance(obj, collections.abc.Iterable):
                try:
                    for i, value in enumerate(obj):
                        queue.append((value, f"{path}[{i}]"))
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
                        if isinstance(slots, str): slots = [slots]
                        for slot in slots:
                            if hasattr(obj, slot):
                                queue.append((getattr(obj, slot), f"{path}.{slot}"))

        print_colored(f">>> [IPC] Auto-scan finished. Materialized {materialized_count} tensors.")
        del frame
        self.captured_tensors.clear()
        self.cpu_cached_tensors.clear()
        try:
            ipc = IPCSocket(self.socket_path)
            s = ipc.connect()    
            ipc.send(s, {"cmd": "LAZY_END"})
            resp = ipc.recv(s)
            s.close()
        except Exception as e:
            print(f"Exit failed: {e}")
        
    def get_size(self, name: str):
        try:
            ipc = IPCSocket(self.socket_path)
            s = ipc.connect()    
            ipc.send(s, {"cmd": "GET_SIZE","name":name})
            resp = ipc.recv(s)
            s.close()
        except Exception as e:
            print(f"Get Size failed: {e}")
        return resp["size"]

    def _ensure_connection(self):
        ipc = IPCSocket(self.socket_path)
        try:
            s = ipc.connect()
            self.is_server_creator = False
            self._handshake(ipc, s)
            s.close()
            return
        except (ConnectionRefusedError, FileNotFoundError):
            pass

        current_script = os.path.abspath(__file__)
        proc = subprocess.Popen([sys.executable, current_script, '--run-server'], start_new_session=True)
        print_colored(f"[Tool] starting child process (PID: {proc.pid})...")
        self.is_server_creator = True
        
        start_time = time.time()
        while time.time() - start_time < 10:
            try:
                s = ipc.connect()
                self._handshake(ipc, s)
                s.close()
                return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("Server failed to start")

    def _handshake(self, ipc, sock):
        ipc.send(sock, {"cmd": "INIT", "device": str(self.target_device)})
        self._bypass = True
        try:
            response = ipc.recv(sock)
            self.server_pid = response.get("server_pid", "Unknown")
        finally:
            self._bypass = False  
    
    def process_vllm_kv_memory(self, m):
        try:
            ipc = IPCSocket(self.socket_path)
            s = ipc.connect()
            ipc.send(s, {"cmd": "KV_MEMORY","memory":m})
            resp = ipc.recv(s)
            s.close()
        except Exception as e:
            print(f"Failed: {e}")
        return resp["kv_memory"]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--run-server':
        IPCServer().run()
