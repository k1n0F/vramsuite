"""
Safe VRAM allocation probe.

This module performs an optional, bounded CUDA allocation probe.
It is intentionally conservative and must only run when explicitly requested.

Allocations go straight through the native CUDA runtime (cudaMalloc/cudaFree),
so the result reflects the driver, not a framework's caching allocator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from vramsuite.backends.cuda import CudaRuntime

PROBE_BACKEND = "cuda-runtime"
BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class ProbeResult:
    available: bool
    backend: str | None
    attempted_mb: int
    allocated_mb: int | None
    safe_allocatable_mb: int | None
    safety_margin_mb: int | None
    error: str | None
    notes: list[str]


def _round_down_to_step(value: int, step: int) -> int:
    return (value // step) * step


def _unavailable(error: str, notes: list[str]) -> ProbeResult:
    return ProbeResult(
        available=False,
        backend=PROBE_BACKEND,
        attempted_mb=0,
        allocated_mb=None,
        safe_allocatable_mb=None,
        safety_margin_mb=None,
        error=error,
        notes=notes,
    )


def run_cuda_probe(
        driver_free_mb: int | None,
        max_probe_mb: int = 1024,
        step_mb: int = 128,
        hard_free_floor_mb: int = 2048,
        safety_ratio: float = 0.85,
        max_free_ratio: float = 0.9,
        device_index: int = 0,
) -> ProbeResult:
    """
    Run a conservative CUDA allocation probe through the native CUDA runtime.

    This probe is bounded:
    - it never attempts more than max_probe_mb
    - it stops before crossing hard_free_floor_mb
    - it frees every allocation before returning, even if one free fails

    It estimates safe allocatable VRAM from the successfully allocated amount.
    """

    notes: list[str] = []

    if step_mb <= 0:
        return _unavailable(f"step_mb must be positive, got {step_mb}", [])

    if driver_free_mb is None:
        return _unavailable(
            "driver_free_mb is not available",
            ["Cannot run allocation probe without driver-level free VRAM."],
        )

    if driver_free_mb <= hard_free_floor_mb:
        return _unavailable(
            "not enough free VRAM above hard_free_floor_mb",
            [
                f"driver_free_mb={driver_free_mb}",
                f"hard_free_floor_mb={hard_free_floor_mb}",
            ],
        )

    if max_probe_mb > driver_free_mb:
        notes.append(
            f"Requested probe_max_mb={max_probe_mb} MB exceeds current driver_free_mb={driver_free_mb} MB."
        )
        notes.append("Probe will be clamped to available free VRAM safety bounds.")

    max_allowed_by_floor_mb = max(driver_free_mb - hard_free_floor_mb, 0)
    max_allowed_by_ratio_mb = int(driver_free_mb * max_free_ratio)

    probe_limit_mb = _round_down_to_step(
        min(max_probe_mb, max_allowed_by_floor_mb, max_allowed_by_ratio_mb),
        step_mb,
    )

    if probe_limit_mb < max_probe_mb:
        notes.append(
            f"Requested probe limit was clamped from {max_probe_mb} MB to {probe_limit_mb} MB."
        )

    if probe_limit_mb <= 0:
        return _unavailable(
            "probe limit is zero after applying safety limits",
            [
                f"driver_free_mb={driver_free_mb}",
                f"max_allowed_by_floor_mb={max_allowed_by_floor_mb}",
                f"hard_free_floor_mb={hard_free_floor_mb}",
            ],
        )

    try:
        cuda = CudaRuntime().__enter__()
    except Exception as exc:
        return _unavailable(
            f"CUDA runtime is not available: {exc}",
            ["Install the NVIDIA driver and CUDA Toolkit (cudart) to run the probe."],
        )

    allocations: list[Any] = []
    allocated_mb = 0
    attempted_mb = 0
    error: str | None = None

    try:
        cuda.set_device(device_index)

        for next_mb in range(step_mb, probe_limit_mb + step_mb, step_mb):
            attempted_mb = next_mb
            # cudaMalloc is synchronous: an out-of-memory error surfaces right here.
            allocations.append(cuda.malloc(step_mb * BYTES_PER_MB))
            allocated_mb = next_mb

    except Exception as exc:
        error = str(exc)

    finally:
        try:
            cuda.free_all(*allocations)
        except Exception as exc:
            error = error or f"failed to release probe allocations: {exc}"
        finally:
            cuda.__exit__(None, None, None)

    safe_allocatable_mb = max(int(allocated_mb * safety_ratio), 0)

    if error is None:
        safety_margin_mb = max(int(allocated_mb * (1 - safety_ratio)), step_mb)
        notes.append("Real CUDA allocation probe completed within configured safety bounds.")
        notes.append("This is not a full VRAM exhaustion test.")
        notes.append(f"Probe limit was capped at {probe_limit_mb} MB.")
    else:
        safety_margin_mb = max(step_mb, attempted_mb - allocated_mb)
        notes.append("CUDA allocation failed before reaching the configured probe limit.")
        notes.append("All successful allocations were released before returning.")

    return ProbeResult(
        available=True,
        backend=PROBE_BACKEND,
        attempted_mb=attempted_mb,
        allocated_mb=allocated_mb,
        safe_allocatable_mb=safe_allocatable_mb,
        safety_margin_mb=safety_margin_mb,
        error=error,
        notes=notes,
    )


def probe_result_to_dict(result: ProbeResult) -> dict[str, Any]:
    """Convert a ProbeResult dataclass to a dictionary."""
    return asdict(result)
