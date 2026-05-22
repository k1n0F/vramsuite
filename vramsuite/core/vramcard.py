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


def _get_primary_gpu(torch_info: dict[str, Any]) -> dict[str, Any]:
    """Return primary CUDA GPU info or an empty fallback."""
    devices = torch_info.get("devices") or []
    if torch_info.get("cuda_available") and devices:
        return devices[0]

    return _empty_gpu_info()

def create_vramcard() -> dict[str, Any]:
    """Create a basic system.vramcard dictionary."""
    fingerprint = collect_fingerprint()

    runtime_info = fingerprint["runtime"]
    torch_info = fingerprint["torch"]
    primary_gpu = _get_primary_gpu(torch_info)

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
         },

        "memory": {
            "driver_total_mb": primary_gpu.get("total_vram_mb"),
            "driver_free_at_scan_mb": None,
            "process_allocatable_mb": None,
            "safe_allocatable_mb": None,
            "safety_margin_mb": None,
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