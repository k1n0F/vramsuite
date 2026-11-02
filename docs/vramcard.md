# `.vramcard` format

`.vramcard` is the JSON profile VRAM Suite writes for one GPU on one machine.

It records what the GPU is, what the software stack around it is, and how much memory was free when the scan ran. Two cards from different machines, drivers, or runtimes (Windows vs WSL) can be compared field by field.

Current schema: `vramcard@0.1.0-alpha`

## Creating a card

```bash
uv run vramsuite doctor
```

This writes `system.vramcard.json` in the current directory.

Custom path:

```bash
uv run vramsuite doctor --output my-system.vramcard.json
```

Print instead of saving:

```bash
uv run vramsuite doctor --no-write-card --json
```

## Sections

```text
schema          format version
generated_at    UTC timestamp of the scan
gpu             short GPU summary
device          detailed native device profile
environment     OS and Python runtime
stack           PyTorch metadata, CUDA runtime, cuBLAS, NVML status
memory          driver memory at scan time (+ probe results)
probe           optional, with --probe
compute_check   optional, with --compute-check
risk_estimate   optional, with --estimate-mb
known_issues    detected issues (reserved, currently empty)
meta            who created the card and its status
```

### `gpu`

A short summary for reports and comparisons:

```json
"gpu": {
  "index": 0,
  "name": "NVIDIA GeForce RTX 5080",
  "total_vram_mb": 16303,
  "compute_capability": "12.0",
  "major": 12,
  "minor": 0,
  "multi_processor_count": 84,
  "source": "nvml"
}
```

`source` names where the values came from. Higher-priority sources win: `nvml`, then the native CUDA runtime, then torch metadata.

### `device`

The full native profile: NVML details, driver versions, CUDA device attributes, and derived values.

See [Device profile](device-profile.md) for every field.

### `environment`

```json
"environment": {
  "os_name": "Windows",
  "os_version": "10.0.26200",
  "platform": "Windows-11-10.0.26200-SP0",
  "python_version": "3.12.9",
  "python_executable": "...\\.venv\\Scripts\\python.exe",
  "is_wsl": false,
  "is_container": false
}
```

### `stack`

What is installed around the GPU:

```json
"stack": {
  "torch_available": false,
  "torch_version": null,
  "cuda_available": false,
  "torch_cuda_version": null,
  "cuda_device_count": 0,
  "cuda_runtime": {
    "available": true,
    "library_path": "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin\\x64\\cudart64_13.dll",
    "runtime_version_text": "13.4",
    "driver_version_text": "13.4",
    "device_count": 1,
    "free_memory_mb": 14985,
    "total_memory_mb": 16275
  },
  "cublas": {
    "available": true,
    "library_path": "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin\\x64\\cublas64_13.dll",
    "version_text": "13.1.0"
  },
  "nvml_available": true,
  "nvml_error": null,
  "nvml_device_count": 1
}
```

The `torch_*` fields come from package metadata only. PyTorch is never imported, so `cuda_available` is `null` when torch is installed and `false` when it is not. `torch_cuda_version` is read from the version tag, for example `2.12.0+cu130` → `13.0`.

`library_path` shows which DLL / `.so` was actually loaded. This is useful when several CUDA versions are installed.

### `memory`

```json
"memory": {
  "driver_total_mb": 16303,
  "driver_free_at_scan_mb": 5141,
  "driver_used_at_scan_mb": 11161,
  "process_allocatable_mb": null,
  "safe_allocatable_mb": null,
  "safety_margin_mb": null,
  "source": "nvml",
  "cuda_runtime_free_mb": 14985
}
```

`driver_*` values come from NVML. `process_allocatable_mb`, `safe_allocatable_mb` and `safety_margin_mb` are filled only by `--probe`.

`cuda_runtime_free_mb` is free memory as `cudaMemGetInfo` sees it. On Windows (WDDM) it is often higher than `driver_free_at_scan_mb`: Windows can page other processes' allocations out of VRAM, so the CUDA runtime reports memory that is not free right now but can be made free.

### `probe`, `risk_estimate`

See [Allocation probe and OOM risk](probe-and-risk.md).

### `compute_check`

Present only with `--compute-check`:

```json
"compute_check": {
  "cublas_sgemm_fp32": { "passed": true, "max_abs_error": 0.0, "error": null },
  "cublaslt_gemm_fp32": { "passed": true, "max_abs_error": 0.0, "error": null },
  "cublaslt_gemm_fp8_e4m3": { "passed": true, "max_abs_error": 0.0, "error": null }
}
```

See [Device profile](device-profile.md#compute-check).

## Comparing cards

`compare` shows the differences between two cards:

```bash
uv run vramsuite compare windows.vramcard.json wsl.vramcard.json
```

Compared sections:

- Runtime — OS, platform, Python version, WSL, container
- GPU — name, total VRAM, compute capability, source
- Memory — driver total / free / used, probe budgets, source
- Probe — backend, attempted / allocated / safe MB, error
- Risk — required / available MB, availability source, usage ratio, risk level

Each row has `Left`, `Right`, `Delta` (`right - left` for numbers) and `Changed`.

Print the comparison as JSON:

```bash
uv run vramsuite compare a.vramcard.json b.vramcard.json --json
```

```json
{
  "changed": true,
  "total_rows": 30,
  "changed_rows": 4,
  "rows": [
    {
      "section": "Memory",
      "field": "Driver Free at Scan MB",
      "left": 14648,
      "right": 5141,
      "delta": -9507,
      "changed": true
    }
  ]
}
```

## Validating a card

`validate` checks that a card is intact and that its numbers agree with each other:

```bash
uv run vramsuite validate system.vramcard.json
```

- **Structure:** `schema` is a `.vramcard` schema, `generated_at` is an ISO timestamp, the `gpu` / `environment` / `stack` / `memory` sections exist, and known fields have the right types.
- **Sanity:** compute capability matches `major.minor`; no negative MB; free / used do not exceed total; `free + used` is close to total; `safe_allocatable <= process_allocatable`; the memory budgets match the `probe` section; `risk_level` is a known level.

Errors mean the card is broken or was edited by hand, and it should not be used for `compare` or `estimate`. Warnings are possible but suspicious values, for example a schema from a newer VRAM Suite or no GPU detected.

```text
| Level   | Field                      | Problem                                             |
|---------+----------------------------+-----------------------------------------------------|
| error   | gpu.compute_capability     | 8.9 disagrees with major/minor                      |
| error   | memory.safe_allocatable_mb | 9000 MB is more than process_allocatable_mb 6144 MB |
| error   | memory.safe_allocatable_mb | disagrees with probe.safe_allocatable_mb            |
Invalid
```

The exit code is `1` when the card is invalid, so it can be used in scripts and CI. `--json` prints `schema`, `valid` and the list of `issues` (`level`, `field`, `message`).

## Python API

```python
import vramsuite

card = vramsuite.create_vramcard()
vramsuite.save_vramcard("system.vramcard.json", card=card)

loaded = vramsuite.load_vramcard("system.vramcard.json")
```

```python
from vramsuite.core.compare import compare_vramcards

diff = compare_vramcards(left_card, right_card)
print(diff["changed_rows"])
```
