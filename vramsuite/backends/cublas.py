from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CUBLAS_STATUS_SUCCESS = 0

@dataclass(frozen=True)
class CublasInfo:
    available: bool
    error: str | None
    library_path: str | None
    handle_created: bool
    version: int | None
    version_text: str | None


def _candidate_library_names() -> list[str]:
    system = platform.system()

    if system == "Windows":
        return [
            "cublas64_13.dll",
            "cublas64_12.dll",
            "cublas64_11.dll",
        ]
    
    if system == "Linux":
        return [
            "libcublas.so",
            "libcublas.so.13",
            "libcublas.so.12",
            "libcublas.so.11",
        ]
    
    return ["libcublas.so"]

def _venv_cublas_candidates() -> list[str]:
    venv  = Path(".venv")

    if not venv.exists():
        return []

    if platform.system() == "Windows":
        return [str(path) for path in venv.rglob("cublas*.dll")]
    
    return [str(path) for path in venv.rglob("libcublas.so*")]

def _load_cublas() -> tuple[ctypes.CDLL | None, str | None, str | None]:
    found = ctypes.util.find_library("cublas")

    candidates: list[str] = []
    if found:
        candidates.append(found)

    candidates.extend(_candidate_library_names())
    candidates.extend(_venv_cublas_candidates())

    errors: list[str] = []

    for name in candidates:
        try:
            path = Path(name)

            if path.exists():
                resolve_path = path.resolve()
                dll_dir = str(resolve_path.parent)

                if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
                    with os.add_dll_directory(dll_dir):
                        return ctypes.CDLL(str(resolve_path)), str(resolve_path), None
                    
                return ctypes.CDLL(str(resolve_path)), str(resolve_path), None
            
            return ctypes.CDLL(name), name, None
        
        except OSError as exc:
            errors.append(f"{name}: {exc}")

    return None, None, "cuBLAS library not found: " + "; ".join(errors)


def _decode_cublas_status(status: int) -> str:
    known_statuses = {
        0: "CUBLAS_STATUS_SUCCESS",
        1: "CUBLAS_STATUS_NOT_INITIALIZED",
        3: "CUBLAS_STATUS_ALLOC_FAILED",
        7: "CUBLAS_STATUS_INVALID_VALUE",
        8: "CUBLAS_STATUS_ARCH_MISMATCH",
        11: "CUBLAS_STATUS_MAPPING_ERROR",
        13: "CUBLAS_STATUS_EXECUTION_FAILED",
        14: "CUBLAS_STATUS_INTERNAL_ERROR",
        15: "CUBLAS_STATUS_NOT_SUPPORTED",
        16: "CUBLAS_STATUS_LICENSE_ERROR",
    }

    return known_statuses.get(status, f"CUBLAS_STATUS_UNKNOWN({status})")


def _check(status: int, operation: str) -> str | None:
    if status == CUBLAS_STATUS_SUCCESS:
        return None

    return f"{operation} failed: {_decode_cublas_status(status)}"

def _format_cublas_version(version: int | None) -> str | None:
    if version is None:
        return None
    
    major = version // 10000
    minor = (version % 10000) // 100
    patch = version % 100

    return f"{major}.{minor}.{patch}"

def _configure_signatures(cublas: ctypes.CDLL) -> None:
    cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cublas.cublasCreate_v2.restype = ctypes.c_int

    cublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
    cublas.cublasDestroy_v2.restype = ctypes.c_int

    cublas.cublasGetVersion_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    cublas.cublasGetVersion_v2.restype = ctypes.c_int


def collect_cublas_info() -> CublasInfo:
    cublas, library_path, load_error = _load_cublas()

    if cublas is None:
        return CublasInfo(
            available=False,
            error=load_error,
            library_path=library_path,
            handle_created=False,
            version=None,
            version_text=None,
        )
    
    handle = ctypes.c_void_p()

    try:
        _configure_signatures(cublas)


        status = cublas.cublasCreate_v2(ctypes.byref(handle))
        error = _check(status, "cublasCreate_v2")
        if error is not None:
            return CublasInfo(
                available=False,
                error=error,
                library_path=library_path,
                handle_created=False,
                version=None,
                version_text=None,
            )
        
        version = ctypes.c_int()
        status = cublas.cublasGetVersion_v2(handle, ctypes.byref(version))
        error = _check(status, "cublasGetVersion_v2")
        if error is not None:
            return CublasInfo(
                available=False,
                error=error,
                library_path=library_path,
                handle_created=True,
                version=None,
                version_text=None,
            )
        
        return CublasInfo(
            available=True,
            error=None,
            library_path=library_path,
            handle_created=True,
            version=version.value,
            version_text=_format_cublas_version(version.value),
        )
    
    except Exception as exc:
        return CublasInfo(
            available=False,
            error=f"Unexpected cuBLAS backend error: {exc}",
            library_path=library_path,
            handle_created=handle.value is not None,
            version=None,
            version_text=None,
        )
    
    finally:
        if handle.value is not None:
            try:
                cublas.cublasDestroy_v2(handle)
            except Exception:
                pass


def collect_cublas_info_dict() -> dict[str, Any]:
    return asdict(collect_cublas_info())