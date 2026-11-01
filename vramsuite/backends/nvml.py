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



class NvmlMemoryV2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class NvmlPciInfo(ctypes.Structure):
    _fields_ = [
        ("busIdLegacy", ctypes.c_char * 16),
        ("domain", ctypes.c_uint),
        ("bus", ctypes.c_uint),
        ("device", ctypes.c_uint),
        ("pciDeviceId", ctypes.c_uint),
        ("pciSubSystemId", ctypes.c_uint),
        ("busId", ctypes.c_char * 32),
    ]


class NvmlUtilization(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]


# nvmlClockType_t
NVML_CLOCK_GRAPHICS = 0
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2

NVML_TEMPERATURE_GPU = 0

NVML_ARCHITECTURES = {
    2: "Kepler",
    3: "Maxwell",
    4: "Pascal",
    5: "Volta",
    6: "Turing",
    7: "Ampere",
    8: "Ada",
    9: "Hopper",
    10: "Blackwell",
}

NVML_DRIVER_MODELS = {0: "WDDM", 1: "TCC", 2: "MCDM"}


class _OptionalNvml:
    """
    Calls optional NVML getters and remembers why each failed one failed.

    A missing value becomes null in the report, and the reason ("Not Supported"
    on WDDM, "missing_symbol" on older drivers, a real error otherwise) lands in
    `unavailable`, so expected gaps are distinguishable from actual faults.
    All arguments here are 32-bit ints or pointers, so default ctypes
    conversion is ABI-safe.
    """

    def __init__(self, nvml: ctypes.CDLL) -> None:
        self.nvml = nvml
        self.unavailable: dict[str, str] = {}

    def _reason(self, status: int) -> str:
        try:
            self.nvml.nvmlErrorString.restype = ctypes.c_char_p
            raw = self.nvml.nvmlErrorString(status)
            text = raw.decode("utf-8", errors="replace") if raw else "unknown"
        except Exception:
            text = "unknown"
        return f"{text} (code {status})"

    def call(self, name: str, *args: Any) -> bool:
        # Distinguish repeated getters such as nvmlDeviceGetClockInfo(SM / MEM).
        selectors = [str(arg) for arg in args if isinstance(arg, int)]
        key = f"{name}({','.join(selectors)})" if selectors else name

        try:
            function = getattr(self.nvml, name)
        except AttributeError:
            self.unavailable[key] = "missing_symbol"
            return False

        try:
            status = function(*args)
        except Exception as exc:
            self.unavailable[key] = f"call_failed: {exc}"
            return False

        if status != NVML_SUCCESS:
            self.unavailable[key] = self._reason(status)
            return False

        return True

    def uint(self, name: str, *args: Any) -> int | None:
        value = ctypes.c_uint()
        if self.call(name, *args, ctypes.byref(value)):
            return int(value.value)
        return None

    def string(self, name: str, *args: Any, size: int = 96) -> str | None:
        buffer = ctypes.create_string_buffer(size)
        if self.call(name, *args, buffer, ctypes.c_uint(size)):
            return buffer.value.decode("utf-8", errors="replace")
        return None


def _mw_to_w(value: int | None) -> float | None:
    return None if value is None else value / 1000


def _collect_system_info(nvml: ctypes.CDLL) -> dict[str, Any]:
    optional = _OptionalNvml(nvml)
    cuda_driver = ctypes.c_int()
    has_cuda_driver = optional.call("nvmlSystemGetCudaDriverVersion_v2", ctypes.byref(cuda_driver))

    return {
        "driver_version": optional.string("nvmlSystemGetDriverVersion", size=80),
        "nvml_version": optional.string("nvmlSystemGetNVMLVersion", size=80),
        "cuda_driver_version": int(cuda_driver.value) if has_cuda_driver else None,
        "cuda_driver_version_text": (
            f"{cuda_driver.value // 1000}.{(cuda_driver.value % 1000) // 10}"
            if has_cuda_driver
            else None
        ),
        "unavailable": optional.unavailable,
    }


def _collect_device_details(nvml: ctypes.CDLL, handle: ctypes.c_void_p) -> dict[str, Any]:
    optional = _OptionalNvml(nvml)
    major = ctypes.c_int()
    minor = ctypes.c_int()
    has_cc = optional.call(
        "nvmlDeviceGetCudaComputeCapability",
        handle,
        ctypes.byref(major),
        ctypes.byref(minor),
    )

    pci = NvmlPciInfo()
    has_pci = optional.call("nvmlDeviceGetPciInfo_v3", handle, ctypes.byref(pci))

    memory_v2 = NvmlMemoryV2(version=ctypes.sizeof(NvmlMemoryV2) | (2 << 24))
    has_memory_v2 = optional.call("nvmlDeviceGetMemoryInfo_v2", handle, ctypes.byref(memory_v2))

    bar1 = NvmlMemory()
    has_bar1 = optional.call("nvmlDeviceGetBAR1MemoryInfo", handle, ctypes.byref(bar1))

    utilization = NvmlUtilization()
    has_utilization = optional.call(
        "nvmlDeviceGetUtilizationRates", handle, ctypes.byref(utilization)
    )

    driver_model_current = ctypes.c_int()
    driver_model_pending = ctypes.c_int()
    has_driver_model = optional.call(
        "nvmlDeviceGetDriverModel_v2",
        handle,
        ctypes.byref(driver_model_current),
        ctypes.byref(driver_model_pending),
    )

    architecture = optional.uint("nvmlDeviceGetArchitecture", handle)
    display_active = optional.uint("nvmlDeviceGetDisplayActive", handle)

    return {
        "uuid": optional.string("nvmlDeviceGetUUID", handle),
        "vbios_version": optional.string("nvmlDeviceGetVbiosVersion", handle, size=32),
        "architecture": NVML_ARCHITECTURES.get(architecture, architecture),
        "compute_capability": f"{major.value}.{minor.value}" if has_cc else None,
        "sm_arch": f"sm_{major.value}{minor.value}" if has_cc else None,
        "cuda_cores": optional.uint("nvmlDeviceGetNumGpuCores", handle),
        "memory_bus_width_bits": optional.uint("nvmlDeviceGetMemoryBusWidth", handle),
        "memory_reserved_mb": _bytes_to_mb(memory_v2.reserved) if has_memory_v2 else None,
        "bar1_total_mb": _bytes_to_mb(bar1.total) if has_bar1 else None,
        "bar1_used_mb": _bytes_to_mb(bar1.used) if has_bar1 else None,
        "display_active": None if display_active is None else bool(display_active),
        "driver_model": (
            NVML_DRIVER_MODELS.get(driver_model_current.value, driver_model_current.value)
            if has_driver_model
            else None
        ),
        "pci": {
            "bus_id": pci.busId.decode("utf-8", errors="replace") if has_pci else None,
            "domain": pci.domain if has_pci else None,
            "bus": pci.bus if has_pci else None,
            "device": pci.device if has_pci else None,
            "device_id": f"0x{pci.pciDeviceId:08x}" if has_pci else None,
            "subsystem_id": f"0x{pci.pciSubSystemId:08x}" if has_pci else None,
            "link_gen_current": optional.uint("nvmlDeviceGetCurrPcieLinkGeneration", handle),
            "link_gen_max": optional.uint("nvmlDeviceGetMaxPcieLinkGeneration", handle),
            "link_width_current": optional.uint("nvmlDeviceGetCurrPcieLinkWidth", handle),
            "link_width_max": optional.uint("nvmlDeviceGetMaxPcieLinkWidth", handle),
            "link_speed_mbps": optional.uint("nvmlDeviceGetPcieSpeed", handle),
        },
        "clocks_mhz": {
            "graphics": optional.uint("nvmlDeviceGetClockInfo", handle, NVML_CLOCK_GRAPHICS),
            "sm": optional.uint("nvmlDeviceGetClockInfo", handle, NVML_CLOCK_SM),
            "memory": optional.uint("nvmlDeviceGetClockInfo", handle, NVML_CLOCK_MEM),
            "graphics_max": optional.uint("nvmlDeviceGetMaxClockInfo", handle, NVML_CLOCK_GRAPHICS),
            "sm_max": optional.uint("nvmlDeviceGetMaxClockInfo", handle, NVML_CLOCK_SM),
            "memory_max": optional.uint("nvmlDeviceGetMaxClockInfo", handle, NVML_CLOCK_MEM),
        },
        "power_w": {
            "usage": _mw_to_w(optional.uint("nvmlDeviceGetPowerUsage", handle)),
            "limit": _mw_to_w(optional.uint("nvmlDeviceGetEnforcedPowerLimit", handle)),
        },
        "temperature_c": optional.uint("nvmlDeviceGetTemperature", handle, NVML_TEMPERATURE_GPU),
        "fan_speed_percent": optional.uint("nvmlDeviceGetFanSpeed", handle),
        "utilization_percent": {
            "gpu": utilization.gpu if has_utilization else None,
            "memory": utilization.memory if has_utilization else None,
        },
        "unavailable": optional.unavailable,
    }


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
    system: dict[str, Any] | None = None


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

                device_dict = asdict(device)
                device_dict["details"] = _collect_device_details(nvml, handle)
                devices.append(device_dict)

            return NvmlInfo(
                available=True,
                error=None,
                device_count=len(devices),
                devices=devices,
                system=_collect_system_info(nvml),
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