from __future__ import annotations

import ctypes
import platform
from dataclasses import asdict, dataclass
from typing import Any



NVML_SUCCESS = 0


class NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong)
    ]



@dataclass(frozen=True)
class NvmlDeviceInfo:
    index: int
    name: str | None
    total_vram_mb: int | None
    free_vram_mb: int | None
    used_vram_mb: int | None




@dataclass(frozen=True)
class NvmlInfo:
    available: bool
    error: str | None
    device_count: int
    devices: list[dict[str, Any]]


def _bytes_to_mb(value: int | None) -> int | None:
    if value is None:
        return None
    return int(value // (1024 * 1024))


def _load_nvml() -> ctypes.CDLL:
    system = platform.system().lower()


    if system == "windows":
        names = ["nvml.dll"]
    elif system == "linux":
        names = ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    else:
        names = ["libnvidia-ml.so.1", "libnvidia-ml.so", "nvml.dll"]

    last_error: Exception | None = None


    for name in names:
        try:
            return ctypes.CDLL(name)
        except Exception as exc:
            last_error = exc


    raise RuntimeError(f"Could not load NVML library: {last_error}")


def _check_nvml_result(result: int, operation: str) -> None:
    if result != NVML_SUCCESS:
        raise RuntimeError(f"{operation} failed with NVML error code {result}")


def collect_nvml_info() -> NvmlInfo:
    """
    Collect NVIDIA GPU memory information through NVML using ctypes.

    This does not require PyTorch.
    It only reports driver-level memory:
    - total VRAM
    - free VRAM
    - used VRAM

    It does not report PyTorch allocated/reserved memory.
    """

    try:
        nvml = _load_nvml()

        nvml.nvmlInit_v2.restype = ctypes.c_int
        nvml.nvmlShutdown.restype = ctypes.c_int

        nvml.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        nvml.nvmlDeviceGetCount_v2.restype = ctypes.c_int

        nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        nvml.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int

        nvml.nvmlDeviceGetMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(NvmlMemory),
        ]
        nvml.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int

        nvml.nvmlDeviceGetName.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        nvml.nvmlDeviceGetName.restype = ctypes.c_int

        _check_nvml_result(nvml.nvmlInit_v2(), "nvmlInit_v2")

        try:
            count = ctypes.c_uint()
            _check_nvml_result(
                nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)),
                "nvmlDeviceGetCount_v2",
            )

            devices: list[dict[str, Any]] = []

            for index in range(count.value):
                handle = ctypes.c_void_p()

                _check_nvml_result(
                    nvml.nvmlDeviceGetHandleByIndex_v2(
                        ctypes.c_uint(index),
                        ctypes.byref(handle),
                    ),
                    "nvmlDeviceGetHandleByIndex_v2",
                )

                name_buffer = ctypes.create_string_buffer(256)
                name: str | None = None

                try:
                    result = nvml.nvmlDeviceGetName(
                        handle,
                        name_buffer,
                        ctypes.c_uint(len(name_buffer)),
                    )
                    if result == NVML_SUCCESS:
                        name = name_buffer.value.decode("utf-8", errors="replace")
                except Exception:
                    name = None

                memory = NvmlMemory()

                _check_nvml_result(
                    nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)),
                    "nvmlDeviceGetMemoryInfo",
                )

                device = NvmlDeviceInfo(
                    index=index,
                    name=name,
                    total_vram_mb=_bytes_to_mb(memory.total),
                    free_vram_mb=_bytes_to_mb(memory.free),
                    used_vram_mb=_bytes_to_mb(memory.used),
                )

                devices.append(asdict(device))

            return NvmlInfo(
                available=True,
                error=None,
                device_count=len(devices),
                devices=devices,
            )

        finally:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass

    except Exception as exc:
        return NvmlInfo(
            available=False,
            error=str(exc),
            device_count=0,
            devices=[],
        )


def collect_nvml_info_dict() -> dict[str, Any]:
    return asdict(collect_nvml_info())