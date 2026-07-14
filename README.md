# Anchor SOSP 2026 Artifact Evaluation

This repository contains the artifact evaluation workflow for the Anchor SOSP 2026
submission (Submission Id 846, "Anchor: Mitigating Shallow Disruptions with Decoupled
Memory"). Anchor is implemented atop DeepSpeed (training) and vLLM (inference); the artifact
reproduces the recovery, end-to-end, and overhead results in the evaluation. It is intended to
be run on the prepared evaluation machine with the existing `gpm` conda environment.

## Table of Contents

- [Evaluating the Artifact](#evaluating-the-artifact)
- [Artifact Overview](#artifact-overview)
- [Environment Setup](#environment-setup)
- [Dependencies](#dependencies)
- [Output Layout](#output-layout)
- [Notes](#notes)

## Evaluating the Artifact

Activate the prepared environment. The workflow has three stages — run the fault-injection
experiments, extract the per-run JSON reports into summary CSVs, then render the figures:

```bash
conda activate gpm

# 1. Run all experiments via the top-level driver (sets IPC_TOOL_VERBOSE and runs both suites).
#    The optional mode arg is single | multi | all (default all = single-GPU then multi-GPU).
bash run.sh                # single-GPU then multi-GPU points (default)
#      bash run.sh single   # single-GPU points only (shorter subset)
#      bash run.sh multi    # multi-GPU points only
#    Equivalently, run one side directly (same single|multi|all arg):
#      bash IPC/ds_script/run_ds.sh [single|multi|all]      # DeepSpeed: overhead + fault + e2e
#      bash IPC/vllm_script/run_vllm.sh [single|multi|all]  # vLLM: e2e + overhead

# 2. Extract reports -> summary CSVs.
(cd IPC/ds_script   && python extract_ds_perf.py)
(cd IPC/vllm_script && python extract_vllm_perf.py)

# 3. Render figures.
cd IPC/plot
python plot_recovery_combined_all_csv.py     # -> Figure 6
python plot_throughput_combined_all_csv.py   # -> Figure 7
python plot_overhead_combined_all_csv.py     # -> Figure 8
```

`run.sh` exports `IPC_TOOL_VERBOSE=1` so the Anchor daemons print startup/handshake
diagnostics. Single-GPU vs multi-GPU is selected by the mode arg (`single` / `multi`, or
`all` for both); it is passed through from `run.sh` to `run_ds.sh` / `run_vllm.sh` and on to
every sub-suite. The full run (mode `all`) takes under about 20 hours, dominated by the e2e
suites (each e2e point runs thousands of training steps / 10k inference requests with repeated
fault injection); we recommend `tmux`. To wipe generated reports and start clean:

```bash
bash IPC/ds_script/clean_reports.sh
bash IPC/vllm_script/clean_reports.sh
```

The repository ships the per-run reports from a completed run, so all figures can be
regenerated immediately by running only stages 2 and 3 (extract and plot) and skipping the
multi-hour stage 1.

## Artifact Overview

The artifact reproduces the evaluation figures:

- **Figure 6 — Fault-recovery breakdown (training + inference).** Training from the DeepSpeed
  `fault` suite; inference from the vLLM `e2e` recovery breakdown. Rendered by
  `IPC/plot/plot_recovery_combined_all_csv.py`.
- **Figure 7 — End-to-end performance under injected failures.** Training throughput
  (steps/s) from the DeepSpeed `e2e` suite; inference latency (ms/req) from the vLLM `e2e`
  suite. Rendered by `IPC/plot/plot_throughput_combined_all_csv.py`.
- **Figure 8 — Overhead of Anchor.** GPU memory and latency/throughput from the DeepSpeed and
  vLLM `overhead` suites. Rendered by `IPC/plot/plot_overhead_combined_all_csv.py`.
- **Figure 9 — AMD training fault-recovery breakdown.** Reproduced in one step by
  `python AMD_IPC/reproduce_figure9.py` (runs the four training baselines over three models,
  writes the per-run logs, and draws the figure; `--plot-only` redraws from existing logs,
  `--methods` selects a subset). Run this on the AMD (ROCm) node inside `gpm`; `--gpu` selects
  the device.
- **Table 2 — Training blocking time.** Reported as a table (no figure); read directly from
  the `train_block_seconds` column of `IPC/ds_script/ds_fault_summary.csv` (produced by
  `extract_ds_perf.py`).

Repository components:

- `run.sh`: top-level driver that runs both suites with `IPC_TOOL_VERBOSE=1`; takes a
  `single|multi|all` mode arg (default `all`).
- `IPC/ds_script/`: DeepSpeed (training) benchmarks, split into `fault/`, `e2e/`, `overhead/`,
  plus `run_ds.sh` (takes `single|multi|all`), `clean_reports.sh`, and `extract_ds_perf.py`.
- `IPC/vllm_script/`: vLLM (inference) benchmarks, `e2e/` and `overhead/`, plus
  `run_vllm.sh` (takes `single|multi|all`), `clean_reports.sh`, and `extract_vllm_perf.py`.
- `IPC/plot/`: the three figure scripts (Figures 6–8) that read the summary CSVs.
- `IPC/ipc_tools/`: the Anchor daemon and tensor-factory interceptors (`ds_tool.py`,
  `vllm_tool.py`, `tool.py`, `ipc_socket.py`); imported by the benchmarks, no install.
- `DeepSpeed/`, `vllm/`, `pytorch/`: vendored, Anchor-patched frameworks, installed editable
  into `gpm`.
- `AMD_DeepSpeed/`, `AMD_IPC/`, `AMD_pytorch/`: the AMD-platform variant for Figure 9.

## Environment Setup

The prepared machine provides:

- NVIDIA A800-SXM4-80GB GPUs (≥2 for the multi-GPU / TP>1 points); CUDA 12.9 toolchain and
  driver.
- The conda environment named `gpm`, with the vendored, Anchor-patched `torch`, `deepspeed`,
  and `vllm` installed editable (so the patches live in `./pytorch`, `./DeepSpeed`, `./vllm`).
- A large `/dev/shm` tmpfs; it scales with the number of ranks, e.g. 400 GB for the 8-GPU
  configuration.
- Local model and dataset caches at the paths the workload scripts default to.

```bash
conda activate gpm
```

The Anchor tool path is resolved relative to the repository, so the tree can be relocated;
`VLLM_IPC_TOOLS_PATH` overrides it if needed.

## Dependencies

The `gpm` environment already contains the patched frameworks and all Python dependencies.
A `pip freeze` snapshot is provided in `requirements.txt` for inspection.

The vendored frameworks (`torch`, `torchvision`, `deepspeed`, `vllm`) are editable source
builds and appear as `-e` / editable entries pointing at `./pytorch`, `./vision`,
`./DeepSpeed`, and `./vllm`; they cannot be recreated by a plain `pip install` from the
snapshot. Reviewers should therefore use the prepared `gpm` environment directly rather than
reinstalling from `requirements.txt`.

## Output Layout

- `IPC/ds_script/{fault,e2e,overhead}/report1*/`: per-run JSON reports for the DeepSpeed
  suites.
- `IPC/vllm_script/e2e/report1/` and `IPC/vllm_script/overhead/report{,_multigpu}/`: per-run
  JSON reports for the vLLM suites.
- `IPC/ds_script/ds_{fault,e2e,overhead}_summary.csv` and
  `IPC/vllm_script/vllm_{e2e,overhead}_summary.csv`: summary CSVs produced by the extractors.
- `IPC/plot/{recovery,throughput,overhead}_combined_all.{png,pdf}`: the rendered figures.

## Notes

- The paper system name is Anchor. Internal names, env vars, and directory names use "IPC"
  (the mechanism's working name); reviewer-facing figures use Anchor.
- Single vs multi-GPU is selected by a `single|multi|all` mode arg threaded through
  `run.sh` → `run_ds.sh` / `run_vllm.sh` → each sub-suite (the per-suite scripts also accept
  the same `single|multi` arg when run directly).
- The three figure scripts read only the summary CSVs, so figures can be regenerated in
  seconds after a run without rerunning experiments.
