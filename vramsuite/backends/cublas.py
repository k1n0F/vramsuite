from __future__ import annotations

import ctypes
import platform
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import Any

from vramsuite.backends.cublaslt.errors import decode_cublas_status
from vramsuite.backends.native_loader import load_native_library


CUBLAS_STATUS_SUCCESS = 0

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2

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


def _check(status: int, operation: str) -> str | None:
    if status == CUBLAS_STATUS_SUCCESS:
        return None

    return f"{operation} failed: {decode_cublas_status(status)}"

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

    cublas.cublasSgemm_v2.argtypes = [
        ctypes.c_void_p,                    # handle
        ctypes.c_int,                       # transa
        ctypes.c_int,                       # transb
        ctypes.c_int,                       # m
        ctypes.c_int,                       # n
        ctypes.c_int,                       # k
        ctypes.POINTER(ctypes.c_float),     # alpha
        ctypes.c_void_p,                    # A
        ctypes.c_int,                       # lda
        ctypes.c_void_p,                    # B
        ctypes.c_int,                       # ldb
        ctypes.POINTER(ctypes.c_float),     # beta
        ctypes.c_void_p,                    # c
        ctypes.c_int,                       # ldc
    ]
    cublas.cublasSgemm_v2.restype = ctypes.c_int


class CublasRuntime:
    def __init__(self) -> None:
        self._native_library = None
        self.cublas: ctypes.CDLL | None = None
        self.library_path: str | None = None
        self.handle: ctypes.c_void_p | None = None

    def __enter__(self) -> "CublasRuntime":
        venv_patterns = (
            ["cublas*.dll"]
            if platform.system() == "Windows"
            else ["libcublas.so*"]
        )

        self._native_library = load_native_library(
            logical_name="cublas",
            candidates=_candidate_library_names(),
            venv_patterns=venv_patterns,
        )

        try:
            loaded = self._native_library.__enter__()

            self.cublas = loaded.require_library()
            self.library_path = loaded.library_path

            _configure_signatures(self.cublas)

            self.handle = self._create_handle()

            return self
        
        except Exception:
            if self._native_library is not None:
                self._native_library.__exit__(None, None, None)
            
            self._native_library = None
            self.cublas = None
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

        self.cublas = None
        self.library_path = None

        if self._native_library is not None:
            self._native_library.__exit__(exc_type, exc, traceback)

        self._native_library = None

    def _require_cublas(self) -> ctypes.CDLL:
        if self.cublas is None:
            raise RuntimeError("cuBLAS runtime is not loaded")
        
        return self.cublas
    
    def _require_handle(self) -> ctypes.c_void_p:
        if self.handle is None:
            raise RuntimeError("cuBLAS handle is not created")
        
        return self.handle
    
    def _check_status(self, status: int, operation: str) -> None:
        error = _check(status, operation)
        if error is not None:
            raise RuntimeError(error)
    
    
    def _create_handle(self) -> ctypes.c_void_p:
        cublas = self._require_cublas()

        handle = ctypes.c_void_p()
        status = cublas.cublasCreate_v2(ctypes.byref(handle))
        self._check_status(status, "cublasCreate_v2")

        if handle.value is None:
            raise RuntimeError("cublasCreate_v2 returned NULL handle")
        
        
        return handle
    
    def _destroy_handle(self, handle: ctypes.c_void_p) -> None:
        cublas = self._require_cublas()

        status = cublas.cublasDestroy_v2(handle)
        self._check_status(status, "cublasDestroy_v2")

    def get_version(self) -> int:
        cublas = self._require_cublas()
        handle = self._require_handle()

        version = ctypes.c_int()
        status = cublas.cublasGetVersion_v2(handle, ctypes.byref(version))
        self._check_status(status, "cublasGetVersion_v2")

        return int(version.value)
    
    def sgemm(
            self,
            *,
            m: int,
            n: int,
            k: int,
            alpha: float,
            a_ptr: ctypes.c_void_p,
            lda: int,
            b_ptr: ctypes.c_void_p,
            ldb: int,
            beta: float,
            c_ptr: ctypes.c_void_p,
            ldc: int,
            transa: int = CUBLAS_OP_N,
            transb: int = CUBLAS_OP_N,
    ) -> None:
        cublas = self._require_cublas()
        handle = self._require_handle()

        alpha_value = ctypes.c_float(alpha)
        beta_value = ctypes.c_float(beta)

        status = cublas.cublasSgemm_v2(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            ctypes.byref(alpha_value),
            a_ptr,
            lda,
            b_ptr,
            ldb,
            ctypes.byref(beta_value),
            c_ptr,
            ldc,
        )

        self._check_status(status, "cublasSgemm_v2")


    def collect_info(self) -> CublasInfo:
        version = self.get_version()

        return CublasInfo(
            available=True,
            error=None,
            library_path=self.library_path,
            handle_created=self.handle is not None and self.handle.value is not None,
            version=version,
            version_text=_format_cublas_version(version),
        )
        


def collect_cublas_info() -> CublasInfo:
    try:
        with CublasRuntime() as runtime:
            return runtime.collect_info()
        
    except Exception as exc:
        return CublasInfo(
            available=False,
            error=f"Unexpected cuBLAS runtime backend error: {exc}",
            library_path=None,
            handle_created=False,
            version=None,
            version_text=None
        )
    
def collect_cublas_info_dict() -> dict[str, Any]:
    return asdict(collect_cublas_info())