from __future__ import annotations

import ctypes
import platform
from dataclasses import asdict, dataclass
from typing import Any
from types import TracebackType

from vramsuite.backends.native_loader import load_native_library


CUDA_SUCCESS = 0

class CudaMemcpyKind:
    HOST_TO_HOST = 0
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2
    DEVICE_TO_DEVICE = 3
    DEFAULT = 4


@dataclass(frozen=True)
class CudaRuntimeInfo:
    available: bool
    error: str | None
    library_path: str | None
    runtime_version: int | None
    runtime_version_text: str | None
    device_count: int | None
    current_device: int | None
    free_memory_mb: int | None
    total_memory_mb: int | None

def _candidate_library_names() -> list[str]:
    system = platform.system()

    if system == "Windows":
        return [
            "cudart64_13.dll",
            "cudart64_130.dll",
            "cudart64_12.dll",
            "cudart64_110.dll",
        ]
    
    if system == "Linux":
        return [
            "libcudart.so",
            "libcudart.so.13",
            "libcudart.so.12",
            "libcudart.so.11.0",
        ]
    
    return ["libcudart.so"]


def _decode_cuda_error(cudart: ctypes.CDLL, code: int) -> str:
    try:
        cudart.cudaGetErrorString.argtypes = [ctypes.c_int]
        cudart.cudaGetErrorString.restype = ctypes.c_char_p

        raw = cudart.cudaGetErrorString(code)
        if raw is None:
            return f"CUDA error code {code}"
        
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return f"CUDA error code {code}"
    

def _check(status: int, cudart: ctypes.CDLL, operation: str) -> str | None:
    if status == CUDA_SUCCESS:
        return None

    return f"{operation} failed: {_decode_cuda_error(cudart, status)}"
    

def _format_runtime_version(version: int | None) -> str | None:
    if version is None:
        return None
        
    major = version // 1000
    minor = (version % 1000) // 10

    return f"{major}.{minor}"
    
def _configure_signatures(cudart: ctypes.CDLL) -> None:
    cudart.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cudart.cudaRuntimeGetVersion.restype = ctypes.c_int

    cudart.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cudart.cudaGetDeviceCount.restype = ctypes.c_int

    cudart.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cudart.cudaGetDevice.restype = ctypes.c_int
        
    cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    cudart.cudaSetDevice.restype = ctypes.c_int

    cudart.cudaMemGetInfo.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    cudart.cudaMemGetInfo.restype = ctypes.c_int

    cudart.cudaMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
    ]
    cudart.cudaMalloc.restype = ctypes.c_int

    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaFree.restype = ctypes.c_int

    cudart.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    cudart.cudaMemcpy.restype = ctypes.c_int

    cudart.cudaMemset.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    cudart.cudaMemset.restype = ctypes.c_int

    cudart.cudaDeviceSynchronize.argtypes = []
    cudart.cudaDeviceSynchronize.restype = ctypes.c_int


class CudaRuntime:
    def __init__(self):
        self._native_library = None
        self.cudart: ctypes.CDLL | None = None
        self.library_path: str | None = None

    def __enter__(self) -> "CudaRuntime":
        venv_patterns = (
            ["cudart*.dll"]
            if platform.system() == "Windows"
            else ["libcudart.so*"]
        
        )
        
        self._native_library = load_native_library(
            logical_name="cudart",
            candidates=_candidate_library_names(),
            venv_patterns=venv_patterns,
        )

        loaded = self._native_library.__enter__()
        
        self.cudart = loaded.require_library()
        self.library_path = loaded.library_path
        
        _configure_signatures(self.cudart)

        return self
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.cudart = None
        self.library_path = None

        if self._native_library is not None:
            self._native_library.__exit__(exc_type, exc, traceback)

        self._native_library = None

    def _require_cudart(self) -> ctypes.CDLL:
        if self.cudart is None:
            raise RuntimeError("Cuda runtime is not loaded")
        
        return self.cudart
    
    def _check_status(self, status: int, operation: str) -> None:
        cudart = self._require_cudart()

        error = _check(status, cudart, operation)
        if error is not None:
            raise RuntimeError(error)
        

    def runtime_version(self) -> int:
        cudart = self._require_cudart()

        version = ctypes.c_int()
        status = cudart.cudaRuntimeGetVersion(ctypes.byref(version))
        self._check_status(status, "cudaRuntimeGetVersion")

        return int(version.value)
    
    def set_device(self, device_index: int) -> None:
        cudart = self._require_cudart()

        status = cudart.cudaSetDevice(ctypes.c_int(device_index))
        self._check_status(status, "cudaSetDevice")

    def current_device(self) -> int:
        cudart = self._require_cudart()

        device = ctypes.c_int()
        status = cudart.cudaGetDevice(ctypes.byref(device))
        self._check_status(status, "cudaGetDevice")

        return int(device.value)


    def memory_info(self) -> tuple[int, int]:
        cudart = self._require_cudart()

        free_bytes = ctypes.c_size_t()
        total_bytes = ctypes.c_size_t()

        status = cudart.cudaMemGetInfo(
            ctypes.byref(free_bytes),
            ctypes.byref(total_bytes),
        )
        self._check_status(status, "cudaMemGetInfo")

        return int(free_bytes.value), int(total_bytes.value)
    
    def malloc(self, size_bytes: int) -> ctypes.c_void_p:
        if size_bytes <= 0:
            raise ValueError(f"size_bytes must be > 0, got {size_bytes}")
        
        cudart = self._require_cudart()

        ptr = ctypes.c_void_p()
        status = cudart.cudaMalloc(
            ctypes.byref(ptr),
            ctypes.c_size_t(size_bytes),
        )
        self._check_status(status, "cudaMalloc")

        if ptr.value is None:
            raise RuntimeError("cudaMalloc returned NULL pointer")

        return ptr
    
    def free(self, ptr: ctypes.c_void_p | None) -> None:
        if ptr is None or ptr.value is None:
            return
        
        cudart = self._require_cudart()

        status = cudart.cudaFree(ptr)
        self._check_status(status, "cudaFree")


    def memcpy(
        self,
        dst: ctypes.c_void_p,
        src: ctypes.c_void_p,
        size_bytes: int,
        kind: int,
    ) -> None:
        if size_bytes < 0:
            raise ValueError(f"size_bytes must be >= 0, got {size_bytes}")
        
        cudart = self._require_cudart()

        status = cudart.cudaMemcpy(
            dst,
            src,
            ctypes.c_size_t(size_bytes),
            ctypes.c_int(kind),
        )
        self._check_status(status, "cudaMemcpy")

    def memset(
        self,
        ptr: ctypes.c_void_p,
        value: int,
        size_bytes: int,
    ) -> None:
        if size_bytes < 0:
            raise ValueError(f"size_bytes must be >= 0, got {size_bytes}")
        
        cudart = self._require_cudart()

        status = cudart.cudaMemset(
            ptr,
            ctypes.c_int(value),
            ctypes.c_size_t(size_bytes),
        )
        self._check_status(status, "cudaMemset")
        
    def synchronize(self) -> None:
        cudart = self._require_cudart()

        status = cudart.cudaDeviceSynchronize()
        self._check_status(status, "cudaDeviceSynchronize")

    def collect_info(self, device_index: int = 0) -> CudaRuntimeInfo:
        runtime_version = self.runtime_version()
        device_count = self.device_count()

        if device_count <= 0:
            return CudaRuntimeInfo(
                available=False,
                error="No CUDA devices found",
                library_path=self.library_path,
                runtime_version=runtime_version,
                runtime_version_text=_format_runtime_version(runtime_version),
                device_count=0,
                current_device=None,
                free_memory_mb=None,
                total_memory_mb=None,
            )
        
        if device_index < 0 or device_index >= device_count:
            return CudaRuntimeInfo(
                available=False,
                error=f"Invalid device index {device_index}. Device count: {device_count}",
                library_path=self.library_path,
                runtime_version=runtime_version,
                runtime_version_text=_format_runtime_version(runtime_version),
                device_count=device_count,
                current_device=None,
                free_memory_mb=None,
                total_memory_mb=None,
            )
        
        self.set_device(device_index)
        current_device = self.current_device()

        free_bytes, total_bytes = self.memory_info()

        return CudaRuntimeInfo(
            available=True,
            error=None,
            library_path=self.library_path,
            runtime_version=runtime_version,
            runtime_version_text=_format_runtime_version(runtime_version),
            device_count=device_count,
            current_device=current_device,
            free_memory_mb=int(free_bytes // 1024 // 1024),
            total_memory_mb=int(total_bytes // 1024 // 1024),
        )
    
    def device_count(self) -> int:
        cudart = self._require_cudart()

        count = ctypes.c_int()
        status = cudart.cudaGetDeviceCount(ctypes.byref(count))
        self._check_status(status, "cudaGetDeviceCount")

        return int (count.value)

def collect_cuda_runtime_info(device_index: int = 0) -> CudaRuntimeInfo:
    try:
        with CudaRuntime() as cuda:
            return cuda.collect_info(device_index=device_index)
        
    except Exception as exc:
        return CudaRuntimeInfo(
            available=False,
            error=f"Unexpected CUDA runtime backend error: {exc}",
            library_path=None,
            runtime_version=None,
            runtime_version_text=None,
            device_count=None,
            current_device=None,
            free_memory_mb=None,
            total_memory_mb=None
        )
         

def collect_cuda_runtime_info_dict(device_index: int = 0) -> dict[str, Any]:
    return asdict(collect_cuda_runtime_info(device_index=device_index))
        