"""
VRAM Suite .vramcard module.

Planned for v0.1-alpha:
- create initial system.vramcard structure
- save/load .vramcard JSON
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
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

        return {
                "index": device.get("index"),
                "name": device.get("name"),
                "total_vram_mb": device.get("total_vram_mb"),
                "compute_capability": None,
                "major": None,
                "minor": None,
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


def create_vramcard() -> dict[str, Any]:
    """Create a basic system.vramcard dictionary."""
    fingerprint = collect_fingerprint()

    runtime_info = fingerprint["runtime"]
    torch_info = fingerprint["torch"]
    nvml_info = fingerprint["nvml"]


    primary_gpu = _get_primary_gpu(
        nvml_info=nvml_info,
        torch_info=torch_info,
    )

    memory_info = _get_driver_memory(
        nvml_info=nvml_info,
        primary_gpu=primary_gpu,
    )

    return {
        "schema": VRAMCARD_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),

        "gpu": primary_gpu,

        "environment": runtime_info,

        "stack": {
            "torch_available": torch_info.get("available"),
            "torch_version": torch_info.get("version"),
            "cuda_available": torch_info.get("cuda_available"),
            "torch_cuda_version": torch_info.get("cuda_version"),
            "cuda_device_count": torch_info.get("device_count"),
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


def save_vramcard(path: str | Path = "system.vramcard.json") -> Path:
    """Create and save system.vramcard.json."""
    output_path = Path(path)
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