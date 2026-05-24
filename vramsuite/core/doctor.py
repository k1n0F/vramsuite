"""
Structured doctor diagnostics API.

This module provides the internal/public API used by the CLI and by
external Python callers. It does not print anything by itself.
"""

from __future__ import annotations

from typing import Any


from vramsuite.core.fingerprint import collect_fingerprint
from vramsuite.core.vramcard import create_vramcard
from vramsuite.core.probe import probe_result_to_dict, run_torch_cuda_probe

def run_doctor(
        with_probe: bool = False,
        probe_max_mb: int = 1024,
        probe_step_mb: int = 128,
        probe_floor_mb: int = 2048,
) -> dict[str, Any]:
    """
    Run VRAM Suite doctor diagnostics and return structured data.

    This function does not print terminal tables and does not save files.
    It only collects diagnostic data and returns it as dictionaries.

    Returns:
        A dictionary containing:
        - fingerprint: raw runtime/torch/nvml fingerprint
        - vramcard: generated .vramcard dictionary
        - runtime: runtime information
        - torch: optional PyTorch/CUDA information
        - nvml: NVML driver memory information
        - gpu: selected primary GPU profile
        - memory: selected memory profile
    """

    fingerprint = collect_fingerprint()
    vramcard = create_vramcard(fingerprint=fingerprint)

    runtime_info = fingerprint.get("runtime", {})
    torch_info = fingerprint.get("torch", {})
    nvml_info = fingerprint.get("nvml", {})
    gpu_info = vramcard.get("gpu", {})
    memory_info = vramcard.get("memory", {})

    probe_info: dict[str, Any] | None = None

    if with_probe:
        probe_result = run_torch_cuda_probe(
            driver_free_mb=memory_info.get("driver_free_at_scan_mb"),
            max_probe_mb=probe_max_mb,
            step_mb=probe_step_mb,
            hard_free_floor_mb=probe_floor_mb,
        )

        probe_info = probe_result_to_dict(probe_result)

        memory_info["process_allocatable_mb"] = probe_info.get("allocated_mb")
        memory_info["safe_allocatable_mb"] = probe_info.get("safe_allocatable_mb")
        memory_info["safety_margin_mb"] = probe_info.get("safety_margin_mb")

        vramcard["memory"] = memory_info
        vramcard["probe"] = probe_info
    
    return {
            "fingerprint": fingerprint,
            "vramcard": vramcard,
            "runtime": runtime_info,
            "torch": torch_info,
            "nvml": nvml_info,
            "gpu": gpu_info,
            "memory": memory_info,
            "probe": probe_info,
        }
