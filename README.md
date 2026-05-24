# VRAM Suite

VRAM Suite is a Python framework for GPU memory diagnostics in local AI inference workflows.

The project is focused on understanding, recording, and later predicting VRAM behavior before a workflow fails with an out-of-memory error.

Current status: **pre-alpha / v0.1-alpha foundation**

## What it does now

VRAM Suite currently provides:

- CLI diagnostics through `vramsuite doctor`
- Public Python API through `import vramsuite`
- Structured doctor API through `run_doctor()`
- System/runtime fingerprint collection
- Optional PyTorch/CUDA detection
- NVIDIA GPU memory detection through NVML using `ctypes`
- Driver-level VRAM information without requiring PyTorch
- `.vramcard` JSON profile generation
- Rich terminal reports
- Optional bounded CUDA allocation probe through PyTorch
- Basic OOM risk estimation through `--estimate-mb`

The allocation probe is disabled by default and only runs when explicitly requested with `--probe`.

The OOM risk estimator is also explicit. It only runs when `--estimate-mb` is provided.

The current `.vramcard` includes:

- OS and Python runtime information
- PyTorch availability
- CUDA availability through PyTorch, if installed
- NVML availability
- NVIDIA GPU name
- Total/free/used driver VRAM
- Optional bounded probe result
- Optional OOM risk estimate
- Memory source information

## Why

Local AI workflows often fail with OOM errors only after the workflow has already started.

VRAM Suite is being built to make GPU memory behavior more visible and predictable.

The long-term goal is to help answer questions like:

- How much VRAM is available right now?
- What is consuming memory before inference starts?
- How much memory is safely allocatable?
- Is this workflow likely to fail with OOM?
- Which part of the stack affects memory usage?
- How much safety margin should be left before running a workflow?

## Installation

This project currently uses `uv`.

```bash
uv sync
```

Run the CLI:

```bash
uv run vramsuite doctor
```

## CLI usage

Show diagnostics and save `system.vramcard.json`:

```bash
uv run vramsuite doctor
```

Run diagnostics without writing a `.vramcard` file:

```bash
uv run vramsuite doctor --no-write-card
```

Write the `.vramcard` to a custom path:

```bash
uv run vramsuite doctor --output my-system.vramcard.json
```

Print the generated `.vramcard` JSON to stdout:

```bash
uv run vramsuite doctor --json
```

Show additional diagnostic information:

```bash
uv run vramsuite doctor --verbose
```

Run the bounded CUDA allocation probe:

```bash
uv run vramsuite doctor --probe
```

Run the probe with custom safety limits:

```bash
uv run vramsuite doctor --probe --probe-max-mb 8192 --probe-step-mb 256 --probe-free-floor-mb 2048
```

Estimate OOM risk for a required VRAM amount:

```bash
uv run vramsuite doctor --estimate-mb 8000
```

Run the probe and estimate OOM risk against the probed safe budget:

```bash
uv run vramsuite doctor --probe --probe-max-mb 12288 --probe-step-mb 256 --probe-free-floor-mb 2048 --estimate-mb 8000
```

## Bounded CUDA allocation probe

VRAM Suite includes an optional bounded CUDA allocation probe.

The probe is designed to be conservative:

- It does not run by default.
- It requires explicit `--probe`.
- It uses PyTorch CUDA only if available.
- It allocates memory in configurable steps.
- It never attempts more than `--probe-max-mb`.
- It keeps a configurable free VRAM floor through `--probe-free-floor-mb`.
- It releases allocated tensors before returning.
- It is not a full VRAM exhaustion test.

Example:

```bash
uv run vramsuite doctor --probe --probe-max-mb 8192 --probe-step-mb 256
```

Example result:

```text
Safe Probe
Available             True
Backend               torch-cuda
Attempted MB          8192
Allocated MB          8192
Safe Allocatable MB   6963
Safety Margin MB      1228
Error                 None
```

This confirms that the current process could allocate the configured amount within the probe limits.

It does **not** mean the whole GPU was stress-tested, and it does **not** intentionally push the GPU to OOM.

## OOM risk estimation

VRAM Suite includes a basic OOM risk estimator.

It can estimate whether a requested VRAM amount is likely to fit inside the currently known memory budget.

Example without probe:

```bash
uv run vramsuite doctor --estimate-mb 8000
```

In this mode, the risk estimator uses driver-level free VRAM from NVML.

Example with probe:

```bash
uv run vramsuite doctor --probe --probe-max-mb 12288 --probe-step-mb 256 --estimate-mb 8000
```

In this mode, the risk estimator uses the safer probed budget.

Example result:

```text
OOM Risk Estimate
Available      True
Required MB    8000
Available MB   10444
Remaining MB   2444
Usage Ratio    76.60%
Risk Level     medium
Reason         Required memory is close to available memory.
```

Risk levels:

- `low` — required memory is well below the available budget
- `medium` — required memory is close to the available budget
- `high` — required memory is very close to the available budget
- `critical` — required memory exceeds the available budget
- `unknown` — not enough information to estimate risk

## Example output

```text
VRAM Suite Doctor
Status: pre-alpha / v0.1-alpha foundation

Runtime
OS        Windows
Python    3.12.9

PyTorch / CUDA
Torch available    True
Torch version      2.12.0+cu130
CUDA available     True
Torch CUDA         13.0

NVML / Driver Memory
NVML available     True
NVML devices       1

NVML Devices
0  NVIDIA GeForce RTX 5080  16303 MB total  14648 MB free  1654 MB used

VRAMCard Memory
Driver total MB          16303
Driver free at scan MB   14648
Driver used at scan MB   1654
Process allocatable MB   12288
Safe allocatable MB      10444
Safety margin MB         1843
Source                   nvml

Safe Probe
Available             True
Backend               torch-cuda
Attempted MB          12288
Allocated MB          12288
Safe Allocatable MB   10444
Safety Margin MB      1843
Error                 None

OOM Risk Estimate
Required MB    8000
Available MB   10444
Remaining MB   2444
Usage Ratio    76.60%
Risk Level     medium
```

## `.vramcard`

VRAM Suite uses its own JSON-based profile format called `.vramcard`.

Example:

```json
{
  "schema": "vramcard@0.1.0-alpha",
  "gpu": {
    "index": 0,
    "name": "NVIDIA GeForce RTX 5080",
    "total_vram_mb": 16303,
    "compute_capability": "12.0",
    "major": 12,
    "minor": 0,
    "multi_processor_count": 84,
    "source": "nvml+torch"
  },
  "memory": {
    "driver_total_mb": 16303,
    "driver_free_at_scan_mb": 14648,
    "driver_used_at_scan_mb": 1654,
    "process_allocatable_mb": 12288,
    "safe_allocatable_mb": 10444,
    "safety_margin_mb": 1843,
    "source": "nvml"
  },
  "probe": {
    "available": true,
    "backend": "torch-cuda",
    "attempted_mb": 12288,
    "allocated_mb": 12288,
    "safe_allocatable_mb": 10444,
    "safety_margin_mb": 1843,
    "error": null
  },
  "risk_estimate": {
    "available": true,
    "required_mb": 8000,
    "available_mb": 10444,
    "remaining_mb": 2444,
    "usage_ratio": 0.766,
    "risk_level": "medium"
  }
}
```

When `--probe` is not used, probe-related values can be `null`.

When `--estimate-mb` is not used, `risk_estimate` is not included.

## Python API

VRAM Suite can also be used as a Python library.

Basic usage:

```python
import vramsuite

result = vramsuite.run_doctor()

print(result["gpu"])
print(result["memory"])
```

`run_doctor()` returns structured diagnostic data:

```text
fingerprint
vramcard
runtime
torch
nvml
gpu
memory
probe
risk_estimate
```

Run doctor with probe enabled:

```python
import vramsuite

result = vramsuite.run_doctor(
    with_probe=True,
    probe_max_mb=1024,
    probe_step_mb=128,
    probe_floor_mb=2048,
)

print(result["probe"])
```

Run doctor with probe and OOM risk estimation:

```python
import vramsuite

result = vramsuite.run_doctor(
    with_probe=True,
    probe_max_mb=8192,
    probe_step_mb=256,
    probe_floor_mb=2048,
    estimate_mb=8000,
)

print(result["risk_estimate"])
```

Create a `.vramcard` directly:

```python
import vramsuite

card = vramsuite.create_vramcard()
vramsuite.save_vramcard("system.vramcard.json", card=card)
```

Direct imports are also available:

```python
from vramsuite.core.fingerprint import collect_fingerprint
from vramsuite.core.vramcard import create_vramcard, save_vramcard

fingerprint = collect_fingerprint()
card = create_vramcard(fingerprint=fingerprint)

save_vramcard("system.vramcard.json", card=card)
```

## Architecture

Current modules:

```text
vramsuite/
  __init__.py        Public Python API
  cli/
    main.py          CLI entrypoint
  core/
    doctor.py        Structured diagnostics API
    fingerprint.py   Runtime, PyTorch and NVML fingerprint collection
    probe.py         Bounded CUDA allocation probe
    reports.py       Rich terminal report rendering
    risk.py          Basic OOM risk estimation
    runtime.py       OS/Python/runtime detection
    vramcard.py      .vramcard creation and loading/saving
  backends/
    nvml.py          NVIDIA NVML reader through ctypes
```

Current data flow:

```text
CLI
  -> run_doctor()
      -> collect_fingerprint()
          -> runtime info
          -> optional PyTorch info
          -> NVML driver memory info
      -> create_vramcard()
      -> optional bounded CUDA allocation probe
      -> optional OOM risk estimate
  -> render doctor report
  -> optionally save .vramcard
```

## Roadmap

Planned next steps:

- Improve probe reporting
- Add availability source reporting for OOM risk estimates
- Add optional memory-touch probe mode
- Add probe hold/debug mode
- Add process-level memory tracking
- Add workflow profile format
- Add ComfyUI workflow analysis
- Add model file inspection
- Add automatic workflow memory estimation
- Add profile validation and schema checks
- Add optional ComfyUI integration

## Status

This project is currently in early pre-alpha.

The current version is focused on building a clean foundation:

- stable CLI
- clean module separation
- `.vramcard` format
- NVML memory reader
- public Python API
- bounded CUDA allocation probe
- basic OOM risk estimation

The current version already provides a working diagnostic pipeline:

```text
NVML -> fingerprint -> vramcard -> optional probe -> optional risk estimate -> CLI/API output
```

The next major step is workflow-level profiling and memory prediction.

## Author

Created by **k1n0F**.
