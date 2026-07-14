import os
import socket
import struct
import multiprocessing
from multiprocessing.reduction import ForkingPickler

SOCKET_PATH = "/tmp/uipc_socket_v3"
CPP_MAGIC = 0xAB
multiprocessing.current_process().authkey = b'mhy_secret_key'

class IPCSocket:
    """IPC channel over a UNIX domain socket.

    Uses a length-prefixed + pickle protocol, and auto-detects C++ binary
    frames (magic=0xAB) versus Python pickle frames.
    """

    def __init__(self, path: str = SOCKET_PATH):
        self.path = path
        self.server_sock = None

    def listen(self, backlog: int = 10):
        """Bind and listen on the UNIX socket, removing any stale socket file first."""
        if os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except OSError:
                if os.path.exists(self.path):
                    raise RuntimeError(f"Cannot remove existing socket: {self.path}")
        
        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(self.path)
        self.server_sock.listen(backlog)
        return self.server_sock

    def accept(self):
        """Accept and return an incoming client connection."""
        if self.server_sock is None:
            raise RuntimeError("Server socket not listening")
        return self.server_sock.accept()

    def close_server(self):
        if self.server_sock is not None:
            try:
                self.server_sock.close()
            except Exception:
                pass
            self.server_sock = None
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except Exception:
                pass

    def connect(self):
        """Connect to the UNIX socket and return the client socket."""
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.path)
        return s

    def send(self, sock: socket.socket, data) -> None:
        """Pickle and send data as a length-prefixed frame."""
        buf = ForkingPickler.dumps(data)
        sock.sendall(struct.pack(">I", len(buf)))
        sock.sendall(buf)

    def send_raw(self, sock: socket.socket, data: bytes) -> None:
        sock.sendall(struct.pack(">I", len(data)))
        sock.sendall(data)

    def _recv_body(self, sock: socket.socket):
        raw_len = sock.recv(4)
        if not raw_len:
            return None
        msg_len = struct.unpack(">I", raw_len)[0]
        buf = bytearray(msg_len)
        view = memoryview(buf)
        pos = 0
        while pos < msg_len:
            read = sock.recv_into(view[pos:], msg_len - pos)
            if not read:
                break
            pos += read
        return bytes(buf)

    def recv(self, sock: socket.socket):
        """Read one frame and decode it with pickle (Python protocol)."""
        raw_len = sock.recv(4)
        if not raw_len:
            return None
        msg_len = struct.unpack(">I", raw_len)[0]
        buf = bytearray(msg_len)
        view = memoryview(buf)
        pos = 0
        while pos < msg_len:
            read = sock.recv_into(view[pos:], msg_len - pos)
            if not read:
                break
            pos += read
        return ForkingPickler.loads(buf)

    def recv_auto(self, sock: socket.socket):
        """Receive a frame, auto-detecting a C++ binary (magic=0xAB) versus Python pickle payload."""
        body = self._recv_body(sock)
        if body is None:
            return None, None
        if len(body) > 0 and body[0] == CPP_MAGIC:
            return "cpp", body
        else:
            return "python", ForkingPickler.loads(body)