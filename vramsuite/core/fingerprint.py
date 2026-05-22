"""
GPU and runtime fingerprint collection.

Planned for v0.1-alpha:
- collect basic runtime info
- detect optional PyTorch
- detect CUDA availability
- collect GPU info if available
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


from vramsuite.core.runtime import collect_runtime_info, runtime_info_to_dict


@dataclass(frozen=True)
class TorchInfo:
    available: bool
    version: str | None
    cuda_available: bool
    cuda_version: str | None
    device_count: int
    devices: list[dict[str, Any]]

def collect_torch_info() -> TorchInfo:
    """Collect PyTorch/CUDA information if torch is installed."""
    try:
        import torch
    except Exception:
        return TorchInfo(
            available=False,
            version=None,
            cuda_available=False,
            cuda_version=None,
            device_count=0,
            devices=[],
            )

    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []

    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index":index,
                    "name": props.major,
                    "total_vram_mb": int(props.total_memory // (1024 * 1024)),
                    "major": props.major,
                    "minor": props.minor,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                    }
                
                )

    return TorchInfo(
        available=True,
        version=str(torch.__version__),
        cuda_available=cuda_available,
        cuda_version=getattr(torch.version, "cuda", None),
        device_count=len(devices),
        devices=devices,
        )

def collect_fingerprint() -> dict[str, Any]:
    """Collect a basic VRAM Suite system fingerprint."""
    runtime = collect_runtime_info()
    torch_info = collect_torch_info()

    return {
        "runtime": runtime_info_to_dict(runtime),
        "torch": asdict(torch_info),
        }