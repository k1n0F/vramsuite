"""
CUDA driver API backend (nvcuda.dll / libcuda.so).

The driver library is installed with every NVIDIA driver, so allocations and
copies through it work without the CUDA Toolkit, PyTorch or a cudart DLL in
the environment. The interface mirrors CudaRuntime, so probe and transfer can
use either.

Device memory is handled through the primary context, the same one the CUDA
runtime uses, so mixing both APIs in one process is safe.
"""

from __future__ import annotations

import ctypes
import platform
from types import TracebackType
from typing import Any

from vramsuite.backends.cuda import (
    CUDA_DEVICE_ATTRIBUTES,
    CudaMemcpyKind,
    CudaRuntime,
    _format_runtime_version,
)
from vramsuite.backends.native_loader import load_native_library

CUDA_SUCCESS = 0
BYTES_PER_MB = 1024 * 1024
# CUdeviceptr is a 64-bit integer, not a host pointer.
CUdeviceptr = ctypes.c_uint64


def _candidate_library_names() -> list[str]:
    if platform.system() == "Windows":
        return ["nvcuda.dll"]
    return ["libcuda.so.1", "libcuda.so"]


def _configure_signatures(cuda: ctypes.CDLL) -> None:
    # The *_v2 names are what cuda.h maps the plain names to since CUDA 4.0/12.0;
    # the plain exports are the legacy 32-bit-size versions.
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDriverGetVersion": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDeviceGetAttribute": [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
        "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
        "cuDevicePrimaryCtxRelease_v2": [ctypes.c_int],
        "cuCtxSetCurrent": [ctypes.c_void_p],
        "cuCtxSynchronize": [],
        "cuMemGetInfo_v2": [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)],
        "cuMemAlloc_v2": [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t],
        "cuMemFree_v2": [CUdeviceptr],
        "cuMemHostAlloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint],
        "cuMemFreeHost": [ctypes.c_void_p],
        "cuMemcpyHtoD_v2": [CUdeviceptr, ctypes.c_void_p, ctypes.c_size_t],
        "cuMemcpyDtoH_v2": [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t],
        "cuMemcpyDtoD_v2": [CUdeviceptr, CUdeviceptr, ctypes.c_size_t],
        "cuGetErrorString": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
    }
    for name, argtypes in signatures.items():
        function = getattr(cuda, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int


class CudaDriver:
    def __init__(self) -> None:
        self._native_library = None
        self.cuda: ctypes.CDLL | None = None
        self.library_path: str | None = None
        self._device: int | None = None

    def __enter__(self) -> "CudaDriver":
        self._native_library = load_native_library(
            logical_name="nvcuda" if platform.system() == "Windows" else "cuda",
            candidates=_candidate_library_names(),
        )
        try:
            loaded = self._native_library.__enter__()
            self.cuda = loaded.require_library()
            self.library_path = loaded.library_path
            _configure_signatures(self.cuda)
            self._check_status(self.cuda.cuInit(0), "cuInit")
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.cuda is not None and self._device is not None:
            self.cuda.cuCtxSetCurrent(None)
            self.cuda.cuDevicePrimaryCtxRelease_v2(self._device)
        self._device = None
        self.cuda = None
        self.library_path = None

        if self._native_library is not None:
            self._native_library.__exit__(exc_type, exc, traceback)
        self._native_library = None

    def _require_cuda(self) -> ctypes.CDLL:
        if self.cuda is None:
            raise RuntimeError("CUDA driver is not loaded")
        return self.cuda

    def _check_status(self, status: int, operation: str) -> None:
        if status == CUDA_SUCCESS:
            return
        message = ctypes.c_char_p()
        if self._require_cuda().cuGetErrorString(status, ctypes.byref(message)) == CUDA_SUCCESS and message.value:
            raise RuntimeError(f"{operation} failed: {message.value.decode('utf-8', errors='replace')}")
        raise RuntimeError(f"{operation} failed: CUDA driver error code {status}")

    def driver_version(self) -> int:
        version = ctypes.c_int()
        self._check_status(self._require_cuda().cuDriverGetVersion(ctypes.byref(version)), "cuDriverGetVersion")
        return int(version.value)

    def device_count(self) -> int:
        count = ctypes.c_int()
        self._check_status(self._require_cuda().cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        return int(count.value)

    def _get_device(self, device_index: int) -> int:
        device = ctypes.c_int()
        self._check_status(self._require_cuda().cuDeviceGet(ctypes.byref(device), device_index), "cuDeviceGet")
        return int(device.value)

    def device_attributes(self, device_index: int) -> dict[str, int | None]:
        """All CUDA_DEVICE_ATTRIBUTES; CUdevice_attribute has the same values as cudaDeviceAttr."""
        cuda = self._require_cuda()
        device = self._get_device(device_index)
        result: dict[str, int | None] = {}
        for name, attr in CUDA_DEVICE_ATTRIBUTES.items():
            value = ctypes.c_int()
            status = cuda.cuDeviceGetAttribute(ctypes.byref(value), attr, device)
            result[name] = int(value.value) if status == CUDA_SUCCESS else None
        return result

    def set_device(self, device_index: int) -> None:
        """Make the primary context of this device current (what cudaSetDevice does lazily)."""
        cuda = self._require_cuda()
        device = self._get_device(device_index)
        if self._device == device:
            return

        context = ctypes.c_void_p()
        self._check_status(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device), "cuDevicePrimaryCtxRetain")
        if self._device is not None:
            cuda.cuDevicePrimaryCtxRelease_v2(self._device)
        self._device = device
        self._check_status(cuda.cuCtxSetCurrent(context), "cuCtxSetCurrent")

    def _require_context(self) -> ctypes.CDLL:
        # Unlike the runtime, the driver API has no implicit device 0 context.
        if self._device is None:
            self.set_device(0)
        return self._require_cuda()

    def memory_info(self) -> tuple[int, int]:
        free_bytes = ctypes.c_size_t()
        total_bytes = ctypes.c_size_t()
        status = self._require_context().cuMemGetInfo_v2(ctypes.byref(free_bytes), ctypes.byref(total_bytes))
        self._check_status(status, "cuMemGetInfo")
        return int(free_bytes.value), int(total_bytes.value)

    def malloc(self, size_bytes: int) -> ctypes.c_void_p:
        if size_bytes <= 0:
            raise ValueError(f"size_bytes must be > 0, got {size_bytes}")

        ptr = CUdeviceptr()
        status = self._require_context().cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(size_bytes))
        self._check_status(status, "cuMemAlloc")
        if not ptr.value:
            raise RuntimeError("cuMemAlloc returned NULL pointer")
        return ctypes.c_void_p(ptr.value)

    def free(self, ptr: ctypes.c_void_p | None) -> None:
        if ptr is None or ptr.value is None:
            return
        self._check_status(self._require_cuda().cuMemFree_v2(ptr.value), "cuMemFree")

    def host_malloc(self, size_bytes: int) -> ctypes.c_void_p:
        """Page-locked (pinned) host memory: the GPU can DMA it directly."""
        if size_bytes <= 0:
            raise ValueError(f"size_bytes must be > 0, got {size_bytes}")

        ptr = ctypes.c_void_p()
        status = self._require_context().cuMemHostAlloc(ctypes.byref(ptr), ctypes.c_size_t(size_bytes), 0)
        self._check_status(status, "cuMemHostAlloc")
        if ptr.value is None:
            raise RuntimeError("cuMemHostAlloc returned NULL pointer")
        return ptr

    def host_free(self, ptr: ctypes.c_void_p | None) -> None:
        if ptr is None or ptr.value is None:
            return
        self._check_status(self._require_cuda().cuMemFreeHost(ptr), "cuMemFreeHost")

    # Same logic as the runtime: free every pointer, raise the first error afterwards.
    free_all = CudaRuntime.free_all

    def memcpy(self, dst: ctypes.c_void_p, src: ctypes.c_void_p, size_bytes: int, kind: int) -> None:
        """cudaMemcpy-style copy; kind is a CudaMemcpyKind."""
        if size_bytes < 0:
            raise ValueError(f"size_bytes must be >= 0, got {size_bytes}")

        cuda = self._require_context()
        size = ctypes.c_size_t(size_bytes)
        if kind == CudaMemcpyKind.HOST_TO_DEVICE:
            status, operation = cuda.cuMemcpyHtoD_v2(dst.value, src, size), "cuMemcpyHtoD"
        elif kind == CudaMemcpyKind.DEVICE_TO_HOST:
            status, operation = cuda.cuMemcpyDtoH_v2(dst, src.value, size), "cuMemcpyDtoH"
        elif kind == CudaMemcpyKind.DEVICE_TO_DEVICE:
            status, operation = cuda.cuMemcpyDtoD_v2(dst.value, src.value, size), "cuMemcpyDtoD"
        else:
            raise ValueError(f"unsupported copy kind {kind}")
        self._check_status(status, operation)

    def synchronize(self) -> None:
        self._check_status(self._require_context().cuCtxSynchronize(), "cuCtxSynchronize")


def collect_cuda_driver_info_dict(device_index: int = 0) -> dict[str, Any]:
    """Driver version, memory and device attributes; needs only the NVIDIA driver."""
    info: dict[str, Any] = {
        "available": False,
        "error": None,
        "library_path": None,
        "driver_version": None,
        "driver_version_text": None,
        "device_count": None,
        "device_index": None,
        "free_memory_mb": None,
        "total_memory_mb": None,
        "attributes": None,
    }
    try:
        with CudaDriver() as cuda:
            info["library_path"] = cuda.library_path
            info["driver_version"] = cuda.driver_version()
            info["driver_version_text"] = _format_runtime_version(info["driver_version"])
            info["device_count"] = cuda.device_count()
            if not 0 <= device_index < info["device_count"]:
                info["error"] = f"Invalid device index {device_index}. Device count: {info['device_count']}"
                return info

            cuda.set_device(device_index)
            free_bytes, total_bytes = cuda.memory_info()
            info.update({
                "available": True,
                "device_index": device_index,
                "free_memory_mb": free_bytes // BYTES_PER_MB,
                "total_memory_mb": total_bytes // BYTES_PER_MB,
                "attributes": cuda.device_attributes(device_index),
            })
    except Exception as exc:
        info["error"] = f"CUDA driver API error: {exc}"
    return info
