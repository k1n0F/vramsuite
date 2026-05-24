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

The allocation probe is disabled by default and only runs when explicitly requested with `--probe`.

The current `.vramcard` includes:

- OS and Python runtime information
- PyTorch availability
- CUDA availability through PyTorch, if installed
- NVML availability
- NVIDIA GPU name
- Total/free/used driver VRAM
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

Show basic diagnostics and save `system.vramcard.json`:

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

Show additional diagnostic details:

```bash
uv run vramsuite doctor --verbose
```
Run the bounded CUDA allocation probe:

```bash
uv run vramsuite doctor --probe
```

Run the probe with custom limits:

```bash
uv run vramsuite doctor --probe --probe-max-mb 8192 --probe-step-mb 256 --probe-free-floor-mb 2048
```
### Probe section

```markdown
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

Example:

```bash
uv run vramsuite doctor --probe --probe-max-mb 8192 --probe-step-mb 256
```

Example result:
```bash
Safe Probe
Available             True
Backend               torch-cuda
Attempted MB          8192
Allocated MB          8192
Safe Allocatable MB   6963
Safety Margin MB      1228
Error                 None
```
This is not a full VRAM exhaustion test. It is a bounded diagnostic probe intended to confirm that the current process can allocate memory safely within configured limits.

## Example output

```text
VRAM Suite Doctor
Status: pre-alpha / v0.1-alpha foundation

Runtime
OS        Windows
Python    3.12.9

PyTorch / CUDA
Torch available    False
CUDA available     False

NVML / Driver Memory
NVML available     True
NVML devices       1

NVML Devices
0  NVIDIA GeForce RTX 5080  16303 MB total  13588 MB free  2714 MB used

VRAMCard Memory
Driver total MB          16303
Driver free at scan MB   13588
Driver used at scan MB   2714
Source                   nvml
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
    "source": "nvml"
  },
  "memory": {
    "driver_total_mb": 16303,
    "driver_free_at_scan_mb": 13588,
    "driver_used_at_scan_mb": 2714,
    "process_allocatable_mb": null,
    "safe_allocatable_mb": null,
    "safety_margin_mb": null,
    "source": "nvml"
  }
}
```

`process_allocatable_mb`, `safe_allocatable_mb`, and `safety_margin_mb` are intentionally `null` for now.

They will be filled by the future allocation probe.

## Python API

Basic usage:

```python
import vramsuite

fingerprint = vramsuite.collect_fingerprint()
card = vramsuite.create_vramcard()

print(card["memory"])
print(result["memory"])
```
run_doctor() returns structured diagnostic data:

```bash
fingerprint
vramcard
runtime
torch
nvml
gpu
memory
probe
```

Run doctor with probe enabled:

```bash
import vramsuite

result = vramsuite.run_doctor(
    with_probe=True,
    probe_max_mb=1024,
    probe_step_mb=128,
    probe_free_floor_mb=2048,
)

print(result["probe"])
```

Direct imports are also available:

```python
from vramsuite.core.fingerprint import collect_fingerprint
from vramsuite.core.vramcard import create_vramcard, save_vramcard

fingerprint = collect_fingerprint()
card = create_vramcard(fingerprint=fingerprint)

save_vramcard("system.vramcard.json", card=card)
```

```markdown
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
    runtime.py       OS/Python/runtime detection
    vramcard.py      .vramcard creation and loading/saving
  backends/
    nvml.py          NVIDIA NVML reader through ctypes

Current data flow:

```text
CLI
  -> run_doctor()
      -> collect_fingerprint()
          -> runtime info
          -> optional PyTorch info
          -> NVML driver memory info
      -> create_vramcard()
      -> optional bounded probe
  -> render doctor report
  -> optionally save .vramcard
```

### Roadmap update

```markdown
## Roadmap

Planned next steps:

- Improve probe reporting
- Add optional memory-touch probe mode
- Add probe hold/debug mode
- Add process-level memory tracking
- Add workflow profile format
- Add ComfyUI workflow analysis
- Add model file inspection
- Add OOM risk estimation
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

The probe and prediction layer will come next.

## Author

Created by **k1n0F**.