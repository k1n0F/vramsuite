"""
Safe VRAM allocation probe.

This module performs an optional, bounded CUDA allocation probe.
It is intentionally conservative and must only run when explicitly requested.
"""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from typing import Any


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
    if step <= 0:
        return value
    return (value // step) * step

def run_torch_cuda_probe(
        driver_free_mb: int | None,
        max_probe_mb: int = 1024,
        step_mb: int = 128,
        hard_free_floor_mb: int = 2048,
        safety_ratio: float = 0.85,
) -> ProbeResult:
    """
    Run a conservative CUDA allocation probe using PyTorch.

    This probe is bounded:
    - it never attempts more than max_probe_mb
    - it stops before crossing hard_free_floor_mb
    - it frees all allocated tensors before returning

    It estimates safe allocatable VRAM from the successfully allocated amount.
    """

    notes: list[str] = []

    if driver_free_mb is None:
        return ProbeResult(
            available=False,
            backend="torch-cuda",
            attempted_mb=0,
            allocated_mb=None,
            safe_allocatable_mb=None,
            safety_margin_mb=None,
            error="driver_free_mb is not available",
            notes=["Cannot run allocation probe without driver-level free VRAM."]
        )
    if driver_free_mb <= hard_free_floor_mb:
        return ProbeResult(
            available=False,
            backend="torch-cuda",
            attempted_mb=0,
            allocated_mb=None,
            safe_allocatable_mb=None,
            safety_margin_mb=None,
            error="not enough free VRAM above hard_free_floor_mb",
            notes=[
                f"driver_free_mb={driver_free_mb}",
                f"hard_free_floor_mb={hard_free_floor_mb}",
            ]
        )
    
    try:
        import torch
    except Exception as exc:
        return ProbeResult(
            available=False,
            backend="torch-cuda",
            attempted_mb=0,
            allocated_mb=None,
            safe_allocatable_mb=None,
            safety_margin_mb=None,
            error=f"PyTorch is not available: {exc}",
            notes=["Install a CUDA-enabled PyTorch build to run the real probe."],
        )
    
    if not torch.cuda.is_available():
        return ProbeResult(
            available=False,
            backend="torch-cuda",
            attempted_mb=0,
            allocated_mb=None,
            safe_allocatable_mb=None,
            safe_margin_mb=None,
            error="torch.cuda is not available",
            notes=["PyTorch is installed, but CUDA is not available in this environment."],
        )
    
    max_allowed_by_floor_mb = max(driver_free_mb - hard_free_floor_mb, 0)
    probe_limit_mb = min(max_probe_mb, max_allowed_by_floor_mb)
    probe_limit_mb = _round_down_to_step(probe_limit_mb, step_mb)

    if probe_limit_mb <= 0:
        return ProbeResult(
            available=False,
            backend="torch-cuda",
            attempted_mb=0,
            allocated_mb=None,
            safe_allocatable_mb=None,
            safe_margin_mb=None,
            error="probe limit is zero after applying safety limits",
            notes=[
                f"driver_free_mb={driver_free_mb}",
                f"max_allowed_by_floor_mb={max_allowed_by_floor_mb}",
                f"hard_free_floor_mb={hard_free_floor_mb}",
            ]
        )
    
    allocated_tensors: list[Any] = []
    allocated_mb = 0
    attempted_mb = 0

    try:
        # uint8 uses exactly 1 byte per element.
        # 1 MiB = 1024 * 1024 bytes.
        bytes_per_mb = 1024 * 1024

        for next_mb in range(step_mb, probe_limit_mb + step_mb, step_mb):
            chunk_mb = next_mb - allocated_mb
            attempted_mb = next_mb

            tensor = torch.empty(
                chunk_mb * bytes_per_mb,
                dtype=torch.uint8,
                device="cuda",
            )

            allocated_tensors.append(tensor)
            allocated_mb = next_mb

            # Synchronize so allocation errors surface here, not later.
            torch.cuda.synchronize()
            
        safety_margin_mb = max(int(allocated_mb * (1 - safety_ratio)), step_mb)
        safe_allocatable_mb = max(int(allocated_mb * safety_ratio), 0)

        notes.append("Real CUDA allocation probe completed within configured safety bounds.")
        notes.append("This is not a full VRAM exhaustion test.")
        notes.append(f"Probe limit was capped at {probe_limit_mb} MB.")

        return ProbeResult(
            available=True,
            backend="torch-cuda",
            attempted_mb=attempted_mb,
            allocated_mb=allocated_mb,
            safe_allocatable_mb=safe_allocatable_mb,
            safety_margin_mb=safety_margin_mb,
            error=None,
            notes=notes,
        )
    
    except Exception as exc:
        notes.append("CUDA allocation failed before reaching the configured probe limit.")
        notes.append("Allocated tensors will be released before returning.")

        safety_margin_mb = max(step_mb, attempted_mb - allocated_mb)
        safe_allocatable_mb = max(int(allocated_mb * safety_ratio), 0)

        return ProbeResult(
            available=True,
            backend="torch-cuda",
            attempted_mb=attempted_mb,
            allocated_mb=allocated_mb,
            safe_allocatable_mb=safe_allocatable_mb,
            safety_margin_mb=safety_margin_mb,
            error=str(exc),
            notes=notes,
        )
    
    finally:
        allocated_tensors.clear()
        gc.collect()

        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

def probe_result_to_dict(result: ProbeResult) -> dict[str, Any]:
    """Convert a ProbeResult dataclass to a dictionary."""
    return asdict(result)