"""
Structured doctor diagnostics API.

This module provides the internal/public API used by the CLI and by
external Python callers. It does not print anything by itself.
"""

from __future__ import annotations

from typing import Any


from vramsuite.core.fingerprint import collect_fingerprint
from vramsuite.core.vramcard import create_vramcard

def run_doctor() -> dict[str, Any]:
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
    
    return {
            "fingerprint": fingerprint,
            "vramcard": vramcard,
            "runtime": runtime_info,
            "torch": torch_info,
            "nvml": nvml_info,
            "gpu": gpu_info,
            "memory": memory_info,
        }
