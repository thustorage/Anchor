"""Minimal AMD IPC-forward-allocation smoke test (alloc a daemon-served tensor, then write/read/matmul it)."""
import os, sys, time
sys.path.insert(0, "/root/mhy/IPC/ipc_tools")
import torch
from tool import TensorFactoryInterceptor

print("torch", torch.__version__, "hip", torch.version.hip, "dev", torch.cuda.get_device_name(0), flush=True)

tool = TensorFactoryInterceptor(target_device=torch.device("cuda"))
tool.enter("amd_smoke")
try:
    x = torch.ones(256, 256, device="cuda")
    torch.cuda.synchronize()
    print("alloc OK  ptr=0x%x  sum=%.1f" % (x.data_ptr(), x.sum().item()), flush=True)
    y = (x @ x)
    torch.cuda.synchronize()
    print("matmul OK  y[0,0]=%.1f (expect 256)" % y[0, 0].item(), flush=True)
    x.add_(1.0); torch.cuda.synchronize()
    print("write OK   sum=%.1f (expect 131072)" % x.sum().item(), flush=True)
    print("SMOKE_PASS", flush=True)
finally:
    tool.exit()
