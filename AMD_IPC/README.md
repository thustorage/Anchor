# Anchor — Figure 9 (AMD training fault-recovery breakdown)

This directory reproduces **Figure 9** of the paper: the breakdown of **AMD (MI100)
training fault-recovery time** across **DS Native · CheckFreq · In-Memory · Anchor**.
It is self-contained — running the experiments *and* drawing the figure is **one
python command**. There is intentionally **no vLLM inference** material and no other
charts.

## Layout
```
AMD_IPC/
├── reproduce_figure9.py         # ONE script: run experiments -> JSON logs -> Figure 9
├── ds_script/                   # the training fault-recovery benches + run scripts
│   ├── ds_hf_checkpoint_bench.py         # DS Native baseline (checkpoint to disk)
│   ├── checkfreq_hf_checkpoint_bench.py  # CheckFreq baseline (async snapshot)
│   ├── memory_hf_checkpoint_bench.py     # In-Memory baseline (shared CPU buffer)
│   ├── pccheck_hf_checkpoint_bench.py    # (dependency of memory bench; not plotted)
│   ├── ipc_hf_attach_bench.py            # ANCHOR (live IPC daemon)
│   └── run_{ds,checkfreq,memory,ipc}_hf_three_models.sh
├── ipc_tools/                   # Anchor daemon + interceptors (dual-target ROCm/CUDA)
│   ├── tool.py · ds_tool.py · ipc_socket.py · amd_ipc_smoke.py
├── plot/
│   ├── update_plot_fault_recovery_stacked.py   # the Figure 9 plotting script
│   ├── make_figure9.sh                          # plot-only helper (bash)
│   └── figures/update_lab1_fault_recovery_stacked.{png,pdf}
└── experiment_logs/report1/     # the JSON experiment logs (4 methods × 3 models)
    └── {ds_native,checkfreq,memory,ipc}/{qwen3_0_6b,llama3_2_1b,qwen2_5_1_5b}.json
```

## Reproduce in one step

```bash
conda activate gpm                     # ROCm torch on the AMD (MI100) node
python reproduce_figure9.py            # run all 4 methods × 3 models -> logs -> Figure 9
```
Options:
```bash
python reproduce_figure9.py --plot-only        # skip experiments, just (re)draw from existing logs
python reproduce_figure9.py --methods ipc      # a subset of {ds_native,checkfreq,memory,ipc}
python reproduce_figure9.py --gpu 0 --python-bin /home/hjr/anaconda3/envs/gpm/bin/python
```
Output: `plot/figures/update_lab1_fault_recovery_stacked.{png,pdf}`.

> **Where it runs:** the full run needs the MI100 + ROCm torch and the models/dataset
> below, so run it on the AMD node inside `gpm`. `--plot-only` works anywhere with
> matplotlib (e.g. `--python-bin /root/miniconda3/bin/python` on this host).

## How the logs are generated

`reproduce_figure9.py` invokes one run script per baseline (setting `PYTHON_BIN`,
`GPU_ID`, `BASELINE_NAME`, `REPORT_DIR`, and a `PYTHONPATH` that points at the bundled
`ipc_tools/` + `ds_script/`). Each run script trains **three models**
(Qwen3-0.6B, LLaMA3.2-1B, Qwen2.5-1.5B) with **DeepSpeed ZeRO-3, bf16**, injects a
fault at the epoch boundary, then measures recovery, writing
`experiment_logs/report1/<method>/<model>.json`:

| Method | Bench | Recovery mechanism |
|---|---|---|
| `ds_native` | `ds_hf_checkpoint_bench.py` | reload from a disk checkpoint |
| `checkfreq` | `checkfreq_hf_checkpoint_bench.py` | async CPU snapshot + background persist |
| `memory` | `memory_hf_checkpoint_bench.py` | restore from a shared-CPU-memory buffer |
| `ipc` (Anchor) | `ipc_hf_attach_bench.py` | re-attach to GPU state held by the IPC daemon |

Each JSON uses the current `IPC/ds_script/fault` schema: `recovery_total_seconds`
plus the breakdown `engine_init_seconds` / `load_seconds` / `other_seconds` (which
sum to the total and are what Figure 9 stacks), a separate `train_block_seconds`,
and `model` / `model_label` / `batch_size` / `seq_len` / `zero_stage` / `gpu_ids`
(ipc adds `checkpoint_backend`; baselines add `expected_world_size`).

**Environment / inputs (AMD node):** `gpm` env (torch 2.9.1a0 / HIP 6.4, DeepSpeed
0.17.7), ROCm 6.4.3, **kernel `5.15.0-152-generic`** (ROCm-validated — see note below),
GPU selected via `HIP_VISIBLE_DEVICES`. Models at `/root/mhy/model/{qwen3-0.6b,
llama3.2-1b, qwen2.5-1.5b-it}`, dataset at `/root/mhy/datasets/wikitext/wikitext-2-raw-v1`.
Sanity-check the IPC path first with `python ipc_tools/amd_ipc_smoke.py` (expect `SMOKE_PASS`).

> Kernel note: the IPC/KFD path is unstable on unvalidated kernels (6.11+ showed
> `amdgpu ih ring buffer overflow` + IPC memory-access faults); keep `5.15.0-152` +
> ROCm 6.4.3 for gfx908.

## Figure 9 — what it draws
Grouped stacked bars: one group per model, four bars per group
(DS Native, CheckFreq, In-Memory, Anchor), each stacked into
**Engine Init · Ckpt Load/Attach · Other**.

## Reference numbers (total fault-to-recovery seconds)
The logs currently shipped were produced on **2026-04-02** (MI100). The figure
decomposes these totals:

| Model | DS Native | CheckFreq | In-Memory | **Anchor** |
|---|--:|--:|--:|--:|
| Qwen3-0.6B  | 23.40 | 23.50 | 19.88 | **17.78** |
| LLaMA3.2-1B | 25.75 | 26.15 | 20.60 | **17.25** |
| Qwen2.5-1.5B| 28.63 | 28.75 | 21.56 | **18.99** |

Anchor recovers fastest, with near-zero checkpoint-load/attach (live IPC daemon) —
the point of Figure 9.
