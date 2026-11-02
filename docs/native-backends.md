# Native backends

VRAM Suite talks to NVIDIA libraries directly through Python `ctypes`.

It does not import PyTorch, Triton, CuPy, or `pynvml`. The runtime dependencies are only `typer`, `rich` and `pydantic`.

## Why native

The main reasons:

- **Observing without changing.** A diagnostic tool should not load 2 GB of PyTorch and its own CUDA libraries just to read how much memory is free.
- **No DLL conflicts.** PyTorch ships its own `cublas` / `cublasLt`. On Windows, if torch is imported first, the toolkit `cublas64_13.dll` binds to torch's `cublasLt64_13.dll` by name, and cuBLAS calls fail with `CUBLAS_STATUS_INVALID_VALUE`. VRAM Suite avoids this by never importing torch.
- **Exact control.** Every allocation, handle, and descriptor is created and released explicitly, so the tool itself does not leak memory into the numbers it reports.
- **Same path as kernels.** Custom kernels talk to the same CUDA driver, runtime and cuBLASLt. A profile collected this way describes what a native kernel will actually see.

## Libraries

```text
NVML         nvml.dll / libnvidia-ml.so        memory, device details, PCIe, clocks, processes
CUDA driver  nvcuda.dll / libcuda.so.1         allocation probe, transfer benchmark (cuMemAlloc, cuMemcpy)
cudart       cudart64_13.dll / libcudart.so    device attributes in doctor, memory for compute checks
cuBLAS     cublas64_13.dll / libcublas.so    SGEMM
cuBLASLt   cublasLt64_13.dll / libcublasLt.so  FP32 / FP8 matmul
pdh, gdi32, kernel32, ntdll (Windows)        per-process GPU memory
```

## What needs what

NVML and the CUDA driver API are installed with every NVIDIA driver. `cudart`, cuBLAS and cuBLASLt come with the CUDA Toolkit (or NVIDIA wheels in the environment).

```text
driver only        doctor (NVML part), processes, inspect, estimate, transfer, doctor --probe, validate, compare
+ CUDA Toolkit     doctor device attributes from cudart, doctor --compute-check (cuBLAS / cuBLASLt)
```

The probe and the transfer benchmark use the driver API on purpose: they only allocate and copy memory, and the driver library is always there. The driver API has no implicit context, so `CudaDriver` makes the device's **primary context** current, the same one the CUDA runtime uses. Mixing both APIs in one process (`doctor --probe --compute-check`) is safe.

## Library discovery

`backends/native_loader.py` finds the CUDA Toolkit libraries in this order (the driver library is found by its system name):

1. The CUDA Toolkit: `CUDA_PATH`, `CUDA_HOME`, then `/usr/local/cuda`, in `bin/x64`, `bin`, `lib64`
2. The system library search path (plain library name)
3. NVIDIA wheels inside the current virtual environment (`sys.prefix`)

The toolkit goes first on purpose. A venv with PyTorch also contains `cudart` inside `torch/lib`, and loading it from there mixes library versions.

CUDA 13 on Windows puts its DLLs in `%CUDA_PATH%\bin\x64`, not `bin`.

The loaded path is recorded in `.vramcard` (`stack.cuda_runtime.library_path`, `stack.cublas.library_path`).

## Module layout

```text
backends/
  native_loader.py   library discovery
  nvml.py            NVML: memory, device details, processes
  cuda.py            CUDA runtime: memory, attributes, malloc/memcpy
  cuda_driver.py     CUDA driver API: same interface, needs only the NVIDIA driver
  cublas.py          cuBLAS handle and SGEMM
  cublaslt/
    runtime.py       cuBLASLt library and function signatures
    enums.py         data types, compute types, attributes
    descriptors.py   matmul descriptor, preference
    layouts.py       matrix layouts (incl. FP8 TN layout)
    heuristics.py    algorithm selection
    workspace.py     workspace allocation
    matmul.py        cublasLtMatmul call
    errors.py        status decoding
  windows_gpu.py     PDH counters, LUID -> PCI, process info
compute/
  compute_math.py    cuBLAS SGEMM check
  cublaslt_math.py   cuBLASLt FP32 / FP8 checks
  minifloat.py       pure-Python FP8 / FP4 encode/decode
```

## Rules for native code

Every native call in VRAM Suite follows these rules:

- **Declare `argtypes` and `restype`.** Without them `ctypes` passes 64-bit `size_t` as 32-bit `int`, and the call silently uses garbage.
- **Release everything, even on failure.** Freeing several pointers continues after the first error and raises it at the end (`free_all`).
- **Clean up in `__enter__` too.** If `__enter__` fails, Python never calls `__exit__`, so partially created resources are released inside `__enter__`.
- **Report, don't swallow.** An optional NVML query that fails is recorded in `unavailable` with the reason (`Not Supported`, `missing_symbol`, or the call error).

## FP8 and FP4

`compute/minifloat.py` encodes and decodes NVIDIA minifloat formats in pure Python:

```text
e4m3   FP8, 4 exponent bits, 3 mantissa bits, max 448
e5m2   FP8, 5 exponent bits, 2 mantissa bits, max 57344
e2m1   FP4, 2 exponent bits, 1 mantissa bit,  max 6
```

Encoding uses round-to-nearest-even and saturates out-of-range values to the largest finite value, like `__NV_SATFINITE`.

cuBLASLt FP8 matmul has hardware rules:

- Ada (sm_89) or newer
- A and B use the TN layout: `transa = T`, `transb = N`
- `m` and `k` must be multiples of 16
- `E5M2 × E5M2` is not supported; `E4M3 × E4M3`, `E4M3 × E5M2`, `E5M2 × E4M3` are

FP4 GEMM needs block scales and is out of scope for VRAM Suite. It belongs to the separate kernel project.

## Windows notes

Under the WDDM driver model, Windows manages video memory. This changes some numbers:

- NVML cannot report per-process memory. VRAM Suite uses PDH counters instead (see [Process tracking](process-tracking.md)).
- `cudaMemGetInfo` free memory is often higher than NVML free memory.
- Per-process counters can sum to more than the physical VRAM.

Linux and WSL2 do not have these differences in the same form, which is one reason `.vramcard` records the runtime.
