from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
from dataclasses import asdict, dataclass
from typing import Any
from pathlib import Path


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


def _venv_cuda_runtime_candidate() -> list[str]:
    venv  = Path(".venv")

    if not venv.exists():
        return []

    return [str(path) for path in venv.rglob("cudart*.dll")]

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


def _load_cudart() -> tuple[ctypes.CDLL | None, str | None, str | None]:
    found = ctypes.util.find_library("cudart")

    candidates: list[str] = []
    if found:
        candidates.append(found)

    candidates.extend(_candidate_library_names())

    if platform.system() == "Windows":
        candidates.extend(_venv_cuda_runtime_candidate())

    errors: list[str] = []

    for name in candidates:
        try:
            path = Path(name)

            if path.is_absolute():
                dll_dir = str(path.parent)

                if hasattr(os, "add_dll_directory"):
                    with os.add_dll_directory(dll_dir):
                        return ctypes.CDLL(str(path)), str(path), None
                    
            return ctypes.CDLL(name), name, None
        
        except OSError as exc:
            errors.append(f"{name}: {exc}")

    return None, None, "Cuda runtime library not found. Tried: " + "; ".join(errors)


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


def _get_configured_cudart() -> tuple[ctypes.CDLL, str | None]:
    cudart, library_path, load_error = _load_cudart()

    if cudart is None:
        raise RuntimeError(load_error or "Cuda runtime library not found")
    
    _configure_signatures(cudart)

    return cudart, library_path

def cuda_malloc(size_bytes: int) -> ctypes.c_void_p:
    if size_bytes <= 0:
        raise ValueError(f"size_bytes must be > 0, got {size_bytes}")
    
    cudart, _ = _get_configured_cudart()

    ptr = ctypes.c_void_p()
    status = cudart.cudaMalloc(
        ctypes.byref(ptr),
        ctypes.c_size_t(size_bytes),
    )

    error = _check(status, cudart, "cudaMalloc")
    if error is not None:
        raise RuntimeError(error)
    
    if ptr.value is None:
        raise RuntimeError("cudaMalloc returned NULL pointer")
    
    return ptr

def cuda_free(ptr: ctypes.c_void_p | None) -> None:
    if ptr is None or ptr.value is None:
        return
    
    cudart, _ = _get_configured_cudart()

    status = cudart.cudaFree(ptr)
    error = _check(status, cudart, "cudaFree")
    if error is not None:
        raise RuntimeError(error)
    

def cuda_memcpy(
    dst: ctypes.c_void_p | None,
    src: ctypes.c_void_p | None,
    size_bytes: int,
    kind: int, 
) -> None:
    if size_bytes < 0:
        raise ValueError(f"size_bytes must be >= 0, got {size_bytes}")
    
    cudart, _ = _get_configured_cudart()

    status = cudart.cudaMemcpy(
        dst,
        src,
        ctypes.c_size_t(size_bytes),
        ctypes.c_int(kind),
    )

    error = _check(status, cudart, "cudaMemcpy")
    if error is not None:
        raise RuntimeError(error)
    

def cuda_memset(
    ptr: ctypes.c_void_p | None,
    value: int,
    size_bytes: int,
) -> None:
    if size_bytes < 0:
        raise ValueError(f"size_bytes must be >= 0, got {size_bytes}")
    
    cudart, _ = _get_configured_cudart()

    status = cudart.cudaMemset(
        ptr,
        ctypes.c_int(value),
        ctypes.c_size_t(size_bytes),
    )

    error = _check(status, cudart, "cudaMemset")
    if error is not None:
        raise RuntimeError(error)


def cuda_synchronize() -> None:
    cudart, _ = _get_configured_cudart()

    status = cudart.cudaDeviceSynchronize()
    error = _check(status, cudart, "cudaDeviceSynchronize")
    if error is not None:
        raise RuntimeError(error)

def collect_cuda_runtime_info(device_index: int = 0) -> CudaRuntimeInfo:
    cudart, library_path, load_error = _load_cudart()

    if cudart is None:
        return CudaRuntimeInfo(
            available=False,
            error=load_error,
            library_path=library_path,
            runtime_version=None,
            runtime_version_text=None,
            device_count=None,
            current_device=None,
            free_memory_mb=None,
            total_memory_mb=None
        )
        
    try:
        _configure_signatures(cudart)

        runtime_version = ctypes.c_int()
        status = cudart.cudaRuntimeGetVersion(ctypes.byref(runtime_version))
        error = _check(status, cudart, "cudaRuntimeGetVersion")
        if error is not None:
            return CudaRuntimeInfo(
                available=False,
                error=error,
                library_path=library_path,
                runtime_version=None,
                runtime_version_text=None,
                device_count=None,
                current_device=None,
                free_memory_mb=None,
                total_memory_mb=None
            )
            
        device_count = ctypes.c_int()
        status = cudart.cudaGetDeviceCount(ctypes.byref(device_count))
        error = _check(status, cudart, "cudaGetDeviceCount")
        if error is not None:
            return CudaRuntimeInfo(
                available=False,
                error=error,
                library_path=library_path,
                runtime_version=runtime_version.value,
                runtime_version_text=_format_runtime_version(runtime_version.value),
                device_count=None,
                current_device=None,
                free_memory_mb=None,
                total_memory_mb=None
            )
            
        if device_count.value <= 0:
            return CudaRuntimeInfo(
                available=False,
                error="No CUDA devices found",
                library_path=library_path,
                runtime_version=runtime_version.value,
                runtime_version_text=_format_runtime_version(runtime_version.value),
                device_count=0,
                current_device=None,
                free_memory_mb=None,
                total_memory_mb=None
            )
            
        if device_index < 0 or device_index >= device_count.value:
            return CudaRuntimeInfo(
                available=False,
                error=f"Invalid device index: {device_index}. Device count: {device_count.value}",
                library_path=library_path,
                runtime_version=runtime_version.value,
                runtime_version_text=_format_runtime_version(runtime_version.value),
                device_count=device_count.value,
                current_device=None,
                free_memory_mb=None,
                total_memory_mb=None
            )
            
        status = cudart.cudaSetDevice(device_index)
        error = _check(status, cudart, "cudaSetDevice")
        if error is not None:
            return CudaRuntimeInfo(
                available=False,
                error=error,
                library_path=library_path,
                runtime_version=runtime_version.value,
                runtime_version_text=_format_runtime_version(runtime_version.value),
                device_count=device_count.value,
                current_device=None,
                free_memory_mb=None,
                total_memory_mb=None
            )
            
        current_device = ctypes.c_int()
        status = cudart.cudaGetDevice(ctypes.byref(current_device))
        error = _check(status, cudart, "cudaGetDevice")
        if error is not None:
            return CudaRuntimeInfo(
                available=False,
                error=error,
                library_path=library_path,
                runtime_version=runtime_version.value,
                runtime_version_text=_format_runtime_version(runtime_version.value),
                device_count=device_count.value,
                current_device=None,
                free_memory_mb=None,
                total_memory_mb=None
            )
            
        free_bytes = ctypes.c_size_t()
        total_bytes = ctypes.c_size_t()

        status = cudart.cudaMemGetInfo(
            ctypes.byref(free_bytes),
            ctypes.byref(total_bytes),
        )
        error = _check(status, cudart, "cudaMemGetInfo")
        if error is not None:
            return CudaRuntimeInfo(
                available=False,
                error=error,
                library_path=library_path,
                runtime_version=runtime_version.value,
                runtime_version_text=_format_runtime_version(runtime_version.value),
                device_count=device_count.value,
                current_device=current_device.value,
                free_memory_mb=None,
                total_memory_mb=None
            )
            
        return CudaRuntimeInfo(
            available=True,
            error=None,
            library_path=library_path,
            runtime_version=runtime_version.value,
            runtime_version_text=_format_runtime_version(runtime_version.value),
            device_count=device_count.value,
            current_device=current_device.value,
            free_memory_mb=int(free_bytes.value // 1024 // 1024),
            total_memory_mb=int(total_bytes.value // 1024 // 1024),
        )
        
    except Exception as exc:
        return CudaRuntimeInfo(
            available=False,
            error=f"Unexpected CUDA runtime backend error: {exc}",
            library_path=library_path,
            runtime_version=None,
            runtime_version_text=None,
            device_count=None,
            current_device=None,
            free_memory_mb=None,
            total_memory_mb=None
        )
        

def collect_cuda_runtime_info_dict(device_index: int = 0) -> dict[str, Any]:
    return asdict(collect_cuda_runtime_info(device_index=device_index))
        