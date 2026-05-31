from __future__ import annotations

import ctypes
from types import TracebackType

from vramsuite.backends.cublaslt.enums import (
    CUBLAS_COMPUTE_32F,
    CUBLAS_MATMUL_PREF_MAX_WORKSPACE_BYTES,
    CUDA_R_32F,
)
from vramsuite.backends.cublaslt.errors import check_cublas_status
from vramsuite.backends.cublaslt.runtime import CublasLtRuntime



class CublasLtDescriptor:
    def __init__(self, runtime: CublasLtRuntime) -> None:
        self.runtime = runtime
        self.ptr: ctypes.c_void_p | None = None

    def _require_cublaslt(self) -> ctypes.CDLL:
        if self.runtime.cublaslt is None:
            raise RuntimeError("cuBLASLt runtime is not loaded")
        
        return self.runtime.cublaslt
    
    def _require_ptr(self) -> ctypes.c_void_p:
        if self.ptr is None or self.ptr.value is None:
            raise RuntimeError(f"{self.__class__.__name__} is not created")
        
        return self.ptr
    
    def __enter__(self):
        self.create()
        return self
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.destroy()


class MatmulDesc(CublasLtDescriptor):
    def __init__(
        self,
        runtime: CublasLtRuntime,
        *,
        compute_type: int = CUBLAS_COMPUTE_32F,
        scale_type: int = CUDA_R_32F,
    ) -> None:
        super().__init__(runtime)
        self.compute_type = compute_type
        self.scale_type = scale_type

    def create(self) -> None:
        if self.ptr is not None:
            return
        
        cublaslt = self._require_cublaslt()

        desc = ctypes.c_void_p()
        status = cublaslt.cublasLtMatmulDescCreate(
            ctypes.byref(desc),
            self.compute_type,
            self.scale_type,
        )
        check_cublas_status(status, "cublasLtMatmulDescCreate")

        if desc.value is None:
            raise RuntimeError("cublasLtMatmulDescCreate returned NULL descriptor")
        
        self.ptr = desc

    def destroy(self) -> None:
        if self.ptr is None:
            return
        
        cublaslt = self._require_cublaslt()

        try:
            status = cublaslt.cublasLtMatmulDescDestroy(self.ptr)
            check_cublas_status(status, "cublasLtMatmulDescDestroy")
        finally:
            self.ptr = None

class MatrixLayout(CublasLtDescriptor):
    def __init__(
        self,
        runtime: CublasLtRuntime,
        *,
        data_type: int = CUDA_R_32F,
        rows: int,
        cols: int,
        ld: int,
    ) -> None:
        super().__init__(runtime)
        self.data_type = data_type
        self.rows = rows
        self.cols = cols
        self.ld = ld

    def create(self) -> None:
        if self.ptr is not None:
            return
        
        if self.rows <= 0:
            raise ValueError("rows must be positive")
        if self.cols <= 0:
            raise ValueError("cols must be positive")
        if self.ld < self.rows:
            raise ValueError("ld must be >= rows for column-major layout")
        
        cublaslt = self._require_cublaslt()

        layout = ctypes.c_void_p()
        status = cublaslt.cublasLtMatrixLayoutCreate(
            ctypes.byref(layout),
            self.data_type,
            self.rows,
            self.cols,
            self.ld,
        )
        check_cublas_status(status, "cublasLtMatrixLayoutCreate")

        if layout.value is None:
            raise RuntimeError("cublasLtMatrixLayoutCreate returned NULL descriptor")
        
        self.ptr = layout

    def destroy(self) -> None:
        if self.ptr is None:
            return
        
        cublaslt = self._require_cublaslt()

        try:
            status = cublaslt.cublasLtMatrixLayoutDestroy(self.ptr)
            check_cublas_status(status, "cublasLtMatrixLayoutDestroy")
        finally:
            self.ptr = None


class MatmulPreference(CublasLtDescriptor):
    def __init__(
        self,
        runtime: CublasLtRuntime,
        *,
        max_workspace_bytes: int = 0,
    ) -> None:
        super().__init__(runtime)
        self.max_workspace_bytes = max_workspace_bytes


    def create(self) -> None:
        if self.ptr is not None:
            return
        
        if self.max_workspace_bytes < 0:
            raise ValueError("max_workspace_bytes must be >= 0")
        
        cublaslt = self._require_cublaslt()

        pref = ctypes.c_void_p()
        status = cublaslt.cublasLtMatmulPreferenceCreate(ctypes.byref(pref))
        check_cublas_status(status, "cublasLtMatmulPreferenceCreate")

        if pref.value is None:
            raise RuntimeError("cublasLtMatmulPreferenceCreate returned NULL descriptor")
        
        self.ptr = pref
        
        self.set_max_workspace_bytes(self.max_workspace_bytes)

    def destroy(self) -> None:
        if self.ptr is None:
            return
        
        cublaslt = self._require_cublaslt()

        try:
            status = cublaslt.cublasLtMatmulPreferenceDestroy(self.ptr)
            check_cublas_status(status, "cublasLtMatmulPreferenceDestroy")
        finally:
            self.ptr = None

    def set_max_workspace_bytes(self, value: int) -> None:
        if value < 0:
            raise ValueError("max workspace bytes must be >= 0")
        
        cublaslt = self._require_cublaslt()
        pref = self._require_ptr()

        workspace_value = ctypes.c_uint64(value)

        status = cublaslt.cublasLtMatmulPreferenceSetAttribute(
            pref,
            CUBLAS_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            ctypes.byref(workspace_value),
            ctypes.sizeof(workspace_value),
        )
        check_cublas_status(status, "cublasLtMatmulPreferenceSetAttribute")

        self.max_workspace_bytes = value