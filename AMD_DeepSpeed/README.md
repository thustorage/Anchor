# AMD_DeepSpeed — AMD-exclusive DeepSpeed parts

Everything else in DeepSpeed is **shared** with NVIDIA and lives in the top-level
`DeepSpeed/` directory (the Anchor consistency protocol in
`deepspeed/runtime/zero/stage3.py` and `stage_1_and_2.py` is portable Python and
identical for both platforms).

This directory holds **only** the parts that cannot be shared with NVIDIA: the
ROCm/HIP inference kernels (there are `cuda_*` counterparts already present in the
shared `DeepSpeed/` tree that NVIDIA uses instead).

```
deepspeed/inference/v2/kernels/core_ops/
├── core_ops_hip.cpp          # HIP registration (AMD analogue of core_ops.cpp)
├── hip_layer_norm/           # AMD analogue of cuda_layer_norm/
├── hip_linear/               # AMD analogue of cuda_linear/
└── hip_rms_norm/             # AMD analogue of cuda_rms_norm/
```

## How to use on an AMD/ROCm node

The paths mirror their location inside the shared `DeepSpeed/` tree. Overlay them
before building DeepSpeed on ROCm:

```bash
cp -r AMD_DeepSpeed/deepspeed  DeepSpeed/     # overlay hip_* kernels next to cuda_*
cd DeepSpeed && DS_BUILD_OPS=0 pip install -e .   # build ROCm ops as needed
```

On NVIDIA nodes, do **not** overlay these — the shared `DeepSpeed/` already has the
`cuda_*` kernels.

> Note: these are the only DeepSpeed files that are AMD-exclusive. The AMD-only
> incremental low-memory flatten that previously existed in `stage_1_and_2.py` was
> intentionally dropped (both platforms now use the stock
> `flatten_dense_tensors_aligned` path).
