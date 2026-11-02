# Device profile

`vramsuite doctor` collects a detailed native profile of the GPU and stores it in the `device` section of `.vramcard`.

The profile answers questions like:

- Which architecture and compute capability is this GPU?
- How much shared memory can one block use?
- What is the theoretical memory bandwidth?
- Which PCIe link is the card actually running at?
- Which driver and CUDA driver version are installed?

These are the numbers a kernel author, a model loader, or a memory estimator needs before doing anything with the GPU.

## Usage

Collect the profile and save it:

```bash
uv run vramsuite doctor
```

Print the full profile as JSON:

```bash
uv run vramsuite doctor --no-write-card --json
```

Run the native compute check as well:

```bash
uv run vramsuite doctor --compute-check
```

## Where the data comes from

The profile is built from two native sources:

- **NVML** (`nvml.dll` / `libnvidia-ml.so`) — identity, PCIe link, clocks, power, temperature, utilization, driver versions
- **CUDA driver API** (`nvcuda.dll` / `libcuda.so`) — `cuDeviceGetAttribute` limits: SMs, threads, registers, shared memory, L2. Installed with the driver; the CUDA runtime (`cudart`) is used instead only if the driver API is unavailable

When both sources report the same value, NVML wins. Lower-priority sources only fill gaps:

```text
NVML  >  CUDA driver API (or runtime)  >  torch metadata
```

### GPU matching

NVML and CUDA can number GPUs differently. On a multi-GPU system, NVML device 0 is not guaranteed to be CUDA device 0.

VRAM Suite matches them by PCI address (`domain:bus:device`). The `nvml_match` field records how the match was made:

- `pci_bus_id` — matched by PCI address (the normal case)
- `single_device` — only one NVML device exists, so it is used
- `index_fallback` — PCI address was not available, matched by index
- `none` — no NVML device could be matched

## Example

Real output on an RTX 5080 (shortened):

```json
"device": {
  "cuda_device_index": 0,
  "nvml_device_index": 0,
  "nvml_match": "pci_bus_id",
  "nvml": {
    "architecture": "Blackwell",
    "compute_capability": "12.0",
    "sm_arch": "sm_120",
    "cuda_cores": 10752,
    "memory_bus_width_bits": 256,
    "memory_reserved_mb": 326,
    "bar1_total_mb": 16384,
    "driver_model": "WDDM",
    "pci": {
      "bus_id": "00000000:01:00.0",
      "link_gen_current": 2,
      "link_gen_max": 5,
      "link_width_current": 8,
      "link_width_max": 16
    },
    "clocks_mhz": {
      "graphics_max": 3090,
      "memory_max": 15001
    },
    "power_w": {
      "usage": 32.606,
      "limit": 360.0
    },
    "unavailable": {}
  },
  "nvml_system": {
    "driver_version": "617.14",
    "cuda_driver_version_text": "13.4",
    "unavailable": {}
  },
  "cuda_attributes": {
    "multiprocessor_count": 84,
    "max_threads_per_multiprocessor": 1536,
    "max_registers_per_multiprocessor": 65536,
    "max_shared_memory_per_block": 49152,
    "max_shared_memory_per_block_optin": 101376,
    "max_shared_memory_per_multiprocessor": 102400,
    "l2_cache_size": 67108864,
    "max_persisting_l2_cache_size": 41943040,
    "memory_clock_rate_khz": 15001000,
    "global_memory_bus_width_bits": 256
  },
  "derived": {
    "theoretical_bandwidth_gb_s": 960.1,
    "max_shared_memory_per_block_optin_kb": 99.0
  }
}
```

### Reading the PCIe link

`link_gen_current` is often lower than `link_gen_max` while the GPU is idle. The card drops to a slower PCIe generation to save power and goes back up under load. This is normal.

`link_width_current` is different: it does not change with load. A value of `8` with `link_width_max` of `16` means the slot runs at x8. On many motherboards this is expected when M.2 / NVMe drives share PCIe lanes with the GPU slot.

### Derived values

`derived` holds values computed from the raw attributes:

- `theoretical_bandwidth_gb_s` — `memory_clock × 2 × bus_width / 8`. This is the ceiling; real kernels reach a fraction of it.
- `max_shared_memory_per_block_optin_kb` — the largest shared memory a kernel can request per block after opting in. Attention kernels size their tiles from this value.

### Unavailable values

Not every GPU or driver supports every NVML query. When a query fails, the value is left out and the reason is recorded in `unavailable`:

```json
"unavailable": {
  "nvmlDeviceGetFanSpeed": "Not Supported (code 3)",
  "nvmlDeviceGetClockInfo(2)": "missing_symbol"
}
```

An empty `unavailable` means every query succeeded.

## Compute check

`--compute-check` runs small GEMMs through native cuBLAS and cuBLASLt and compares the result with a reference computed on the CPU.

```bash
uv run vramsuite doctor --compute-check
```

Example result:

```text
                       Compute Check
┌────────────────────────┬────────┬───────────────┬───────┐
│ Check                  │ Passed │ Max abs error │ Error │
├────────────────────────┼────────┼───────────────┼───────┤
│ cublas_sgemm_fp32      │ True   │ 0.0           │       │
│ cublaslt_gemm_fp32     │ True   │ 0.0           │       │
│ cublaslt_gemm_fp8_e4m3 │ True   │ 0.0           │       │
└────────────────────────┴────────┴───────────────┴───────┘
```

The checks:

- `cublas_sgemm_fp32` — classic cuBLAS SGEMM, 256×256
- `cublaslt_gemm_fp32` — cuBLASLt FP32 matmul, 256×256
- `cublaslt_gemm_fp8_e4m3` — cuBLASLt FP8 E4M3 × E4M3 matmul with FP32 output, 128×128

The FP8 check uses small integers in `[-8, 8]`. They are exactly representable in E4M3, so the expected error is `0.0`, not "close to zero".

Each check is isolated: if one fails, the others still run, and the failure is shown in the `Error` column instead of stopping the doctor.

FP8 GEMM needs Ada (sm_89) or newer. On older GPUs `cublaslt_gemm_fp8_e4m3` is expected to fail with a not-supported error.

## Python API

```python
import vramsuite

result = vramsuite.run_doctor(with_compute_check=True)

device = result["vramcard"]["device"]
print(device["derived"]["theoretical_bandwidth_gb_s"])
print(result["vramcard"]["compute_check"])
```
