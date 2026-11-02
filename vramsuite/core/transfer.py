"""
Host <-> device transfer benchmark through the native CUDA runtime.

Measures what offloading actually costs on this machine: how fast weights move
between system RAM and VRAM over the real PCIe link (which may be narrower or
slower than the card supports), for pinned and pageable host memory.
"""

from __future__ import annotations

import ctypes
import statistics
import time
from typing import Any

from vramsuite.backends.cuda import CudaMemcpyKind, CudaRuntime
from vramsuite.backends.nvml import collect_nvml_info_dict
from vramsuite.core.vramcard import _select_nvml_device

BYTES_PER_MB = 1024 * 1024

# Per-lane transfer rate (GT/s) and line-code efficiency for each PCIe generation.
PCIE_GENERATIONS = {
    1: (2.5, 8 / 10),
    2: (5.0, 8 / 10),
    3: (8.0, 128 / 130),
    4: (16.0, 128 / 130),
    5: (32.0, 128 / 130),
    6: (64.0, 242 / 256),
}


def pcie_theoretical_gb_s(generation: int | None, width: int | None) -> float | None:
    """One-direction link bandwidth in GB/s (10^9), before protocol overhead."""
    if generation not in PCIE_GENERATIONS or not width:
        return None
    rate, efficiency = PCIE_GENERATIONS[generation]
    return round(rate * efficiency * width / 8, 1)


def _time_copy(
        cuda: CudaRuntime,
        dst: ctypes.c_void_p,
        src: ctypes.c_void_p,
        size_bytes: int,
        kind: int,
        repeats: int,
) -> list[float]:
    # Warm-up: the first copy pays for lazy driver setup.
    cuda.memcpy(dst, src, size_bytes, kind)
    cuda.synchronize()

    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        cuda.memcpy(dst, src, size_bytes, kind)
        # cudaMemcpy may return before the DMA ends (D2D, small pageable copies).
        cuda.synchronize()
        seconds.append(time.perf_counter() - start)
    return seconds


def _result(name: str, memory: str, size_bytes: int, seconds: list[float], traffic: int = 1):
    """traffic=2 for device->device: every byte is read and written in VRAM."""
    rates = [traffic * size_bytes / s / 1e9 for s in seconds]
    return {
        "direction": name,
        "host_memory": memory,
        "size_mb": size_bytes // BYTES_PER_MB,
        "best_gb_s": round(max(rates), 2),
        "median_gb_s": round(statistics.median(rates), 2),
        "error": None,
    }


def run_transfer_benchmark(
        device_index: int = 0,
        size_mb: int = 256,
        repeats: int = 5,
) -> dict[str, Any]:
    size_bytes = size_mb * BYTES_PER_MB
    results: list[dict[str, Any]] = []

    with CudaRuntime() as cuda:
        cuda.set_device(device_index)
        attributes = cuda.device_attributes(device_index)

        device = cuda.malloc(size_bytes)
        device_copy = None
        pinned = None
        try:
            # Pageable: ordinary Python-owned memory; the driver stages it
            # through its own pinned buffer, like an unpinned torch tensor.
            pageable_buffer = ctypes.create_string_buffer(size_bytes)
            pageable = ctypes.cast(pageable_buffer, ctypes.c_void_p)

            try:
                pinned = cuda.host_malloc(size_bytes)
            except RuntimeError as exc:
                pinned_error = str(exc)
            else:
                pinned_error = None

            for memory, host in (("pinned", pinned), ("pageable", pageable)):
                for name, dst, src, kind in (
                        ("host_to_device", device, host, CudaMemcpyKind.HOST_TO_DEVICE),
                        ("device_to_host", host, device, CudaMemcpyKind.DEVICE_TO_HOST),
                ):
                    if host is None:
                        results.append({
                            "direction": name, "host_memory": memory, "size_mb": size_mb,
                            "best_gb_s": None, "median_gb_s": None, "error": pinned_error,
                        })
                        continue
                    seconds = _time_copy(cuda, dst, src, size_bytes, kind, repeats)
                    results.append(_result(name, memory, size_bytes, seconds))

            device_copy = cuda.malloc(size_bytes)
            seconds = _time_copy(
                cuda, device_copy, device, size_bytes, CudaMemcpyKind.DEVICE_TO_DEVICE, repeats
            )
            results.append(_result("device_to_device", "-", size_bytes, seconds, traffic=2))

        finally:
            first_error = None
            try:
                cuda.host_free(pinned)
            except Exception as exc:
                first_error = exc
            try:
                cuda.free_all(device, device_copy)
            except Exception as exc:
                first_error = first_error or exc
            if first_error is not None:
                raise first_error

    # Read the link right after the copies: idle GPUs drop to a lower PCIe gen.
    nvml_device, _ = _select_nvml_device(collect_nvml_info_dict(), attributes)
    pci = ((nvml_device or {}).get("details") or {}).get("pci") or {}

    return {
        "device_index": device_index,
        "name": (nvml_device or {}).get("name"),
        "size_mb": size_mb,
        "repeats": repeats,
        "pcie": {
            "link_gen_max": pci.get("link_gen_max"),
            "link_gen_after_run": pci.get("link_gen_current"),
            "link_width_current": pci.get("link_width_current"),
            "link_width_max": pci.get("link_width_max"),
            "theoretical_gb_s": pcie_theoretical_gb_s(
                pci.get("link_gen_max"), pci.get("link_width_current")
            ),
            "theoretical_full_width_gb_s": pcie_theoretical_gb_s(
                pci.get("link_gen_max"), pci.get("link_width_max")
            ),
        },
        "results": results,
    }
