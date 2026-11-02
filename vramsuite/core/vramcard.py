"""
VRAM Suite .vramcard module: create, save and load system.vramcard JSON.
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



def _pci_address(domain: Any, bus: Any, device: Any) -> tuple[int, int, int] | None:
    if domain is None or bus is None or device is None:
        return None
    return int(domain), int(bus), int(device)


def _select_nvml_device(
        nvml_info: dict[str, Any],
        cuda_attributes: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """
    Find the NVML device that is the same physical GPU as the selected CUDA device.

    NVML enumerates by PCI order, CUDA by CUDA_DEVICE_ORDER / CUDA_VISIBLE_DEVICES,
    so indices can differ on multi-GPU systems. Match by PCI domain:bus:device.
    """
    devices = nvml_info.get("devices") if nvml_info.get("available") else None
    if not devices:
        return None, "none"

    cuda_pci = _pci_address(
        cuda_attributes.get("pci_domain_id"),
        cuda_attributes.get("pci_bus_id"),
        cuda_attributes.get("pci_device_id"),
    )

    if cuda_pci is not None:
        for device in devices:
            pci = (device.get("details") or {}).get("pci") or {}
            if _pci_address(pci.get("domain"), pci.get("bus"), pci.get("device")) == cuda_pci:
                return device, "pci_bus_id"

    # No CUDA runtime (or no PCI info): index 0 is only unambiguous with one GPU.
    return devices[0], "single_device" if len(devices) == 1 else "index_fallback"


def _get_primary_gpu(
        nvml_device: dict[str, Any] | None,
        cuda_attributes: dict[str, Any],
        cuda_source: str,
        torch_info: dict[str, Any],
) -> dict[str, Any]:
    """Source priority: NVML > native CUDA (driver or runtime) > torch (lower ones only fill gaps)."""
    gpu = _empty_gpu_info()
    gpu["source"] = None

    if nvml_device is not None:
        compute_capability = (nvml_device.get("details") or {}).get("compute_capability")
        major, minor = (
            (int(part) for part in compute_capability.split("."))
            if compute_capability
            else (None, None)
        )
        gpu.update({
            "index": nvml_device.get("index"),
            "name": nvml_device.get("name"),
            "total_vram_mb": nvml_device.get("total_vram_mb"),
            "compute_capability": compute_capability,
            "major": major,
            "minor": minor,
            "source": "nvml",
        })

    native = {
        "major": cuda_attributes.get("compute_capability_major"),
        "minor": cuda_attributes.get("compute_capability_minor"),
        "multi_processor_count": cuda_attributes.get("multiprocessor_count"),
    }
    if native["major"] is not None and native["minor"] is not None:
        native["compute_capability"] = f"{native['major']}.{native['minor']}"

    torch_devices = torch_info.get("devices") if torch_info.get("cuda_available") else None
    torch_device = torch_devices[0] if torch_devices else {}

    for fallback, source in ((native, cuda_source), (torch_device, "torch")):
        for key in ("name", "total_vram_mb", "compute_capability", "major", "minor",
                    "multi_processor_count"):
            if gpu.get(key) is None and fallback.get(key) is not None:
                gpu[key] = fallback[key]
                gpu["source"] = gpu["source"] or source

    return gpu


def _get_driver_memory(
        nvml_device: dict[str, Any] | None,
        primary_gpu: dict[str, Any]
) -> dict[str, Any]:
    if nvml_device is not None:
        return {
            "driver_total_mb": nvml_device.get("total_vram_mb"),
            "driver_free_at_scan_mb": nvml_device.get("free_vram_mb"),
            "driver_used_at_scan_mb": nvml_device.get("used_vram_mb"),
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
        nvml_device: dict[str, Any] | None,
        nvml_match: str,
        cuda_device_index: int | None,
        attributes: dict[str, Any],
) -> dict[str, Any]:
    """Hardware limits of the primary GPU, as needed for kernel design."""
    details = nvml_device.get("details") if nvml_device else None

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
        "cuda_device_index": cuda_device_index,
        "nvml_device_index": nvml_device.get("index") if nvml_device else None,
        "nvml_match": nvml_match,
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
    cuda_driver_info = fingerprint.get("cuda_driver") or {}
    cuda_runtime_info = fingerprint.get("cuda_runtime", {})
    cublas_info = fingerprint.get("cublas", {})

    # The driver API comes with every NVIDIA driver; cudart only with the CUDA
    # Toolkit. Both report the same attributes and the same free memory.
    if cuda_driver_info.get("available"):
        cuda, cuda_source, cuda_device_index = cuda_driver_info, "cuda_driver", cuda_driver_info.get("device_index")
    else:
        cuda, cuda_source, cuda_device_index = cuda_runtime_info, "cuda_runtime", cuda_runtime_info.get("current_device")
    cuda_attributes = cuda.get("attributes") or {}

    # One physical GPU for every section below, matched across NVML and CUDA.
    nvml_device, nvml_match = _select_nvml_device(nvml_info, cuda_attributes)

    primary_gpu = _get_primary_gpu(
        nvml_device=nvml_device,
        cuda_attributes=cuda_attributes,
        cuda_source=cuda_source,
        torch_info=torch_info,
    )

    memory_info = _get_driver_memory(
        nvml_device=nvml_device,
        primary_gpu=primary_gpu,
    )
    # cuMemGetInfo / cudaMemGetInfo; on Windows WDDM it can exceed NVML free
    # (WDDM may page other processes' VRAM out), so both are kept. The field
    # name predates the driver API backend and is kept for compatibility.
    memory_info["cuda_runtime_free_mb"] = cuda.get("free_memory_mb")

    device_profile = _get_device_profile(
        nvml_info, nvml_device, nvml_match, cuda_device_index, cuda_attributes
    )
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

            "cuda_driver": {key: value for key, value in cuda_driver_info.items() if key != "attributes"},
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
            "status": "alpha",
            "notes": "Memory values are a snapshot at generated_at. Run `vramsuite doctor --probe` for allocatable budgets.",
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