"""
VRAM Suite .vramcard module.

Planned for v0.1-alpha:
- create initial system.vramcard structure
- save/load .vramcard JSON
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any



from vramsuite.core.fingerprint import collect_fingerprint

VRAMCARD_SCHEMA_VERSION = "vramcard@0.1.0-alpha"

def _empty_gpu_info() -> dict[str, Any]:
        """Return empty GPU info for systems without CUDA GPU detection."""
        return {
            "index": None,
            "name": None,
            "total_vram_mb": None,
            "compute_capability": None,
            "major": None,
            "minor": None,
            "multi_processor_count": None,
         }



def _get_primary_nvml_gpu(nvml_info: dict[str, Any]) -> dict[str, Any]:
     devices = nvml_info.get("devices") or []


     if nvml_info.get("available") and devices:
        device = devices[0]
        compute_capability = (device.get("details") or {}).get("compute_capability")
        major, minor = (
            (int(part) for part in compute_capability.split("."))
            if compute_capability
            else (None, None)
        )

        return {
                "index": device.get("index"),
                "name": device.get("name"),
                "total_vram_mb": device.get("total_vram_mb"),
                "compute_capability": compute_capability,
                "major": major,
                "minor": minor,
                "multi_processor_count": None,
                "source": "nvml",
        }
     return None


def _get_primary_torch_gpu(torch_info: dict[str, Any]) -> dict[str, Any] | None:
     devices = torch_info.get("devices") or []

     if torch_info.get("cuda_available") and devices:
          device = devices[0]

          return {
                "index": device.get("index"),
                "name": device.get("name"),
                "total_vram_mb": device.get("total_vram_mb"),
                "compute_capability": device.get("compute_capability"),
                "major": device.get("major"),
                "minor": device.get("minor"),
                "multi_processor_count": device.get("multi_processor_count"),
                "source": "torch",
            }
     
     return None


def _get_primary_gpu(
          nvml_info: dict[str, Any],
          torch_info: dict[str, Any]
        ) -> dict[str, Any] | None:
    nvml_gpu = _get_primary_nvml_gpu(nvml_info)

    if nvml_gpu is not None:
        torch_gpu = _get_primary_torch_gpu(torch_info)
        
        if torch_gpu:
            nvml_gpu.update({
                "compute_capability": torch_gpu.get("compute_capability"),
                "major": torch_gpu.get("major"),
                "minor": torch_gpu.get("minor"),
                "multi_processor_count": torch_gpu.get("multi_processor_count"),
            })
        return nvml_gpu

    torch_gpu = _get_primary_torch_gpu(torch_info)

    if torch_gpu is not None:
        return torch_gpu
    
    return _empty_gpu_info()


def _get_driver_memory(
        nvml_info: dict[str, Any],
        primary_gpu: dict[str, Any]
) -> dict[str, Any]:
    devices = nvml_info.get("devices") or []

    if nvml_info.get("available") and devices:
        device = devices[0]

        return {
            "driver_total_mb": device.get("total_vram_mb"),
            "driver_free_at_scan_mb": device.get("free_vram_mb"),
            "driver_used_at_scan_mb": device.get("used_vram_mb"),
            "process_allocatable_mb": None,
            "safe_allocatable_mb": None,
            "safety_margin_mb": None,
            "source": "nvml",
        }
    return {
            "driver_total_mb": primary_gpu.get("total_vram_mb"),
            "driver_free_at_scan_mb": None,
            "driver_used_at_scan_mb": None,
            "process_allocatable_mb": None,
            "safe_allocatable_mb": None,
            "safety_margin_mb": None,
            "source": primary_gpu.get("source"),
        }


def _get_device_profile(
        nvml_info: dict[str, Any],
        cuda_runtime_info: dict[str, Any],
) -> dict[str, Any]:
    """Hardware limits of the primary GPU, as needed for kernel design."""
    devices = nvml_info.get("devices") or []
    details = devices[0].get("details") if devices else None
    attributes = cuda_runtime_info.get("attributes") or {}

    memory_clock_khz = attributes.get("memory_clock_rate_khz")
    bus_width_bits = attributes.get("global_memory_bus_width_bits")
    smem_optin = attributes.get("max_shared_memory_per_block_optin")

    # DDR formula (2 transfers per clock). Matches the 960 GB/s spec on RTX 5080
    # (GDDR7); measure with a copy benchmark before trusting it on other cards.
    theoretical_bandwidth_gb_s = (
        round(memory_clock_khz * 1000 * 2 * bus_width_bits / 8 / 1e9, 1)
        if memory_clock_khz and bus_width_bits
        else None
    )

    return {
        "nvml": details,
        "nvml_system": nvml_info.get("system"),
        "cuda_attributes": attributes or None,
        "derived": {
            "theoretical_bandwidth_gb_s": theoretical_bandwidth_gb_s,
            "max_shared_memory_per_block_optin_kb": (
                smem_optin / 1024 if smem_optin else None
            ),
        },
    }


def create_vramcard(fingerprint: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a basic system.vramcard dictionary."""
    if fingerprint is None:
        fingerprint = collect_fingerprint()

    runtime_info = fingerprint["runtime"]
    torch_info = fingerprint["torch"]
    nvml_info = fingerprint["nvml"]
    cuda_runtime_info = fingerprint.get("cuda_runtime", {})
    cublas_info = fingerprint.get("cublas", {})


    primary_gpu = _get_primary_gpu(
        nvml_info=nvml_info,
        torch_info=torch_info,
    )

    memory_info = _get_driver_memory(
        nvml_info=nvml_info,
        primary_gpu=primary_gpu,
    )
    # cudaMemGetInfo; on Windows WDDM it can exceed NVML free (WDDM may page
    # other processes' VRAM out), so both are kept.
    memory_info["cuda_runtime_free_mb"] = cuda_runtime_info.get("free_memory_mb")

    device_profile = _get_device_profile(nvml_info, cuda_runtime_info)
    cuda_runtime_summary = {
        key: value for key, value in cuda_runtime_info.items() if key != "attributes"
    }

    return {
        "schema": VRAMCARD_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),

        "gpu": primary_gpu,
        "device": device_profile,

        "environment": runtime_info,

        "stack": {
            "torch_available": torch_info.get("available"),
            "torch_version": torch_info.get("version"),
            "cuda_available": torch_info.get("cuda_available"),
            "torch_cuda_version": torch_info.get("cuda_version"),
            "cuda_device_count": torch_info.get("device_count"),

            "cuda_runtime": cuda_runtime_summary,
            "cublas": cublas_info,

            "nvml_available": nvml_info.get("available"),
            "nvml_error": nvml_info.get("error"),
            "nvml_device_count": nvml_info.get("device_count"),
         },

        "memory": memory_info,
        "known_issues": [],
        "meta": {
            "created_by": "VRAM Suite",
            "status": "pre-alpha",
            "notes":"initial v0.1-alpha vramcard. Safe allocation probe is not implemented yet.",
         }

    }


def save_vramcard(
    path: str | Path = "system.vramcard.json",
    card: dict[str, Any] | None = None,
) -> Path:
    """Save a .vramcard JSON file."""
    output_path = Path(path)

    if card is None:
        card = create_vramcard()

    output_path.write_text(
        json.dumps(card, indent=2, ensure_ascii=False),
        encoding="utf-8",
     )

    return output_path


def load_vramcard(path: str | Path) -> dict[str, Any]:
    """Load a .vramcard JSON file."""
    input_path = Path(path)
    return json.loads(input_path.read_text(encoding="utf-8"))