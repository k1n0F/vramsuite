"""
GPU and runtime fingerprint collection.

Planned for v0.1-alpha:
- collect basic runtime info
- detect optional PyTorch
- detect CUDA availability
- collect GPU info if available
"""

from __future__ import annotations

import importlib.metadata
import re

from dataclasses import asdict, dataclass
from typing import Any

from vramsuite.core.runtime import collect_runtime_info, runtime_info_to_dict
from vramsuite.backends.nvml import collect_nvml_info_dict
from vramsuite.backends.cuda import collect_cuda_runtime_info_dict
from vramsuite.backends.cublas import collect_cublas_info_dict

@dataclass(frozen=True)
class TorchInfo:
    available: bool
    version: str | None
    cuda_available: bool | None
    cuda_version: str | None
    device_count: int | None
    devices: list[dict[str, Any]]


def _cuda_version_from_local_tag(version: str) -> str | None:
    """'2.12.0+cu130' -> '13.0', '2.4.1+cu118' -> '11.8'; None for CPU builds."""
    match = re.search(r"\+cu(\d+)", version)
    if not match:
        return None
    digits = match.group(1)
    return f"{digits[:-1]}.{digits[-1]}"


def collect_torch_info() -> TorchInfo:
    """
    Detect an installed PyTorch from package metadata, without importing it.

    Importing torch loads its own cudart/cuBLAS/cuBLASLt DLLs into this process.
    Windows then resolves our toolkit cuBLAS dependency (cublasLt64_13.dll) by
    name to torch's copy, mixing two cuBLAS versions (CUBLAS_STATUS_INVALID_VALUE),
    and torch's CUDA context would also skew every memory reading.
    GPU facts come from NVML / the native CUDA runtime instead.
    """
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return TorchInfo(
            available=False,
            version=None,
            cuda_available=False,
            cuda_version=None,
            device_count=0,
            devices=[],
        )

    cuda_version = _cuda_version_from_local_tag(version)

    return TorchInfo(
        available=True,
        version=version,
        # Whether torch can actually reach the GPU is not checked (see above).
        cuda_available=None,
        cuda_version=cuda_version,
        device_count=None,
        devices=[],
    )


def collect_fingerprint() -> dict[str, Any]:
    """Collect a basic VRAM Suite system fingerprint."""
    runtime = collect_runtime_info()
    torch_info = collect_torch_info()
    nvml_info = collect_nvml_info_dict()
    cuda_runtime_info = collect_cuda_runtime_info_dict()
    cublas_info= collect_cublas_info_dict()

    return {
        "runtime": runtime_info_to_dict(runtime),
        "torch": asdict(torch_info),
        "nvml": nvml_info,
        "cuda_runtime": cuda_runtime_info,
        "cublas": cublas_info,
    }