from __future__ import annotations

import ctypes
import platform
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import Any

from vramsuite.backends.native_loader import load_native_library
from vramsuite.backends.cublaslt.errors import check_cublas_status


@dataclass(frozen=True)
class CublasLtInfo:
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
            "cublaslt64_13.dll",
            "cublaslt64_12.dll",
            "cublaslt64_11.dll",
        ]
    
    if system == "Linux":
        return [
            "libcublaslt.so",
            "libcublaslt.so.13",
            "libcublaslt.so.12",
            "libcublaslt.so.11",
        ]
    
    return ["libcublaslt.so"]


def _format_cublaslt_version(version: int | None) -> str | None:
    if version is None:
        return None
    
    major = version // 10000
    minor = (version % 10000) // 100
    patch = version % 100

    return f"{major}.{minor}.{patch}"


def _configure_signatures(cublaslt: ctypes.CDLL) -> None:
    cublaslt.cublasLtCreate.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
    ]
    cublaslt.cublasLtCreate.restype = ctypes.c_int

    cublaslt.cublasLtDestroy.argtypes = [
        ctypes.c_void_p,
    ]
    cublaslt.cublasLtDestroy.restype = ctypes.c_int

    cublaslt.cublasLtGetVersion.argtypes = []
    cublaslt.cublasLtGetVersion.restype = ctypes.c_int

    cublaslt.cublasLtMatmulDescCreate.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),   # matmulDesc
        ctypes.c_int,                      # computeType
        ctypes.c_int,                      # scaleType
    ]
    cublaslt.cublasLtMatmulDescCreate.restype = ctypes.c_int

    cublaslt.cublasLtMatmulDescDestroy.argtypes = [
        ctypes.c_void_p,                   # matmulDesc
    ]
    cublaslt.cublasLtMatmulDescDestroy.restype = ctypes.c_int

    cublaslt.cublasLtMatrixLayoutCreate.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),   # matLayout
        ctypes.c_int,                      # type
        ctypes.c_uint64,                   # rows
        ctypes.c_uint64,                   # cols
        ctypes.c_int64,                    # ld
    ]
    cublaslt.cublasLtMatrixLayoutCreate.restype = ctypes.c_int

    cublaslt.cublasLtMatrixLayoutDestroy.argtypes = [
        ctypes.c_void_p,                   # matLayout
    ]
    cublaslt.cublasLtMatrixLayoutDestroy.restype = ctypes.c_int

    cublaslt.cublasLtMatmulPreferenceCreate.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),   # pref
    ]
    cublaslt.cublasLtMatmulPreferenceCreate.restype = ctypes.c_int

    cublaslt.cublasLtMatmulPreferenceDestroy.argtypes = [
        ctypes.c_void_p,                   # pref
    ]
    cublaslt.cublasLtMatmulPreferenceDestroy.restype = ctypes.c_int

    cublaslt.cublasLtMatmulPreferenceSetAttribute.argtypes = [
        ctypes.c_void_p,                   # pref
        ctypes.c_int,                      # attr
        ctypes.c_void_p,                   # buf
        ctypes.c_size_t,                   # sizeInBytes
    ]
    cublaslt.cublasLtMatmulPreferenceSetAttribute.restype = ctypes.c_int


class CublasLtRuntime:
    def __init__(self) -> None:
        self._native_library = None
        self.cublaslt: ctypes.CDLL | None = None
        self.library_path: str | None = None
        self.handle: ctypes.c_void_p | None = None
        
    def __enter__(self) -> "CublasLtRuntime":
        venv_patterns = (
            ["cublaslt*.dll"]
            if platform.system() == "Windows"
            else ["libcublaslt.so*"]
        )

        self._native_library = load_native_library(
            logical_name="cublaslt",
            candidates=_candidate_library_names(),
            venv_patterns=venv_patterns,
        )

        try:
            loaded = self._native_library.__enter__()

            self.cublaslt = loaded.require_library()
            self.library_path = loaded.library_path

            _configure_signatures(self.cublaslt)

            self.handle = self._create_handle()

            return self
        
        except Exception:
            if self._native_library is not None:
                self._native_library.__exit__(None, None, None)
            
            self._native_library = None
            self.cublaslt = None
            self.library_path = None
            self.handle = None

            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.handle is not None:
            try:
                self._destroy_handle(self.handle)
            finally:
                self.handle = None

        self.cublaslt = None
        self.library_path = None

        if self._native_library is not None:
            self._native_library.__exit__(exc_type, exc, traceback)

        self._native_library = None

    def _require_cublaslt(self) -> ctypes.CDLL:
        if self.cublaslt is None:
            raise RuntimeError("cuBLASLt runtime is not loaded")
        
        return self.cublaslt
    
    def _require_handle(self) -> ctypes.c_void_p:
        if self.handle is None or self.handle.value is None:
            raise RuntimeError("cuBLASLt handle is not created")
        
        return self.handle
    
    def _create_handle(self) -> ctypes.c_void_p:
        cublaslt = self._require_cublaslt()

        handle = ctypes.c_void_p()
        status = cublaslt.cublasLtCreate(ctypes.byref(handle))
        check_cublas_status(status, "cublasLtCreate")

        if handle.value is None:
            raise RuntimeError("cublasLtCreate returned NULL handle")

        return handle
    
    def _destroy_handle(self, handle: ctypes.c_void_p) -> None:
        cublaslt = self._require_cublaslt()

        status = cublaslt.cublasLtDestroy(handle)
        check_cublas_status(status, "cublasLtDestroy")

    def get_version(self) -> int:
        cublaslt = self._require_cublaslt()

        version = cublaslt.cublasLtGetVersion()
        
        return int(version)
    
    def collect_info(self) -> CublasLtInfo:
        version = self.get_version()

        return CublasLtInfo(
            available=True,
            error=None,
            library_path=self.library_path,
            handle_created=self.handle is not None and self.handle.value is not None,
            version=version,
            version_text=_format_cublaslt_version(version),
        )
    
def collect_cublaslt_info() -> CublasLtInfo:
    try:
        with CublasLtRuntime() as cublaslt:
            return cublaslt.collect_info()
        
    except Exception as exc:
        return CublasLtInfo(
            available=False,
            error=f"Unexpected cuBLASLt runtime backend error: {exc}",
            library_path=None,
            handle_created=False,
            version=None,
            version_text=None
        )
    
def collect_cublaslt_info_dict() -> dict[str, Any]:
    return asdict(collect_cublaslt_info())