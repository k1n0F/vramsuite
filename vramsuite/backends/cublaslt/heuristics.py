from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

from vramsuite.backends.cublaslt.descriptors import (
    MatrixLayout,
    MatmulDesc,
    MatmulPreference,
)
from vramsuite.backends.cublaslt.errors import check_cublas_status
from vramsuite.backends.cublaslt.runtime import CublasLtRuntime

@dataclass
class RawMatmulHeuristic:
    requested_algo_count: int
    returned_algo_count: int
    result_stride_bytes: int
    result_buffer: Any
    check_buffer: Any
    workspace_limit_bytes: int

    @property
    def algo_ptr(self) -> ctypes.c_void_p:
        """
        cublasLtMatmulHeuristicResult_t starts with cublasLtMatmulAlgo_t algo.

        So the address of the result buffer is also the address of the first
        algorithm descriptor.
        
        """

        return ctypes.c_void_p(ctypes.addressof(self.result_buffer))
    
    @property
    def result_ptr(self) -> ctypes.c_void_p:
        return ctypes.c_void_p(ctypes.addressof(self.result_buffer))
    
    @property
    def check_result_ptr(self) -> ctypes.c_void_p:
        return ctypes.c_void_p(ctypes.addressof(self.check_buffer))
    

def _require_descriptor_ptr(descriptor: object, name: str) -> ctypes.c_void_p:
    ptr = getattr(descriptor, "ptr", None)

    if ptr is None or ptr.value is None:
        raise RuntimeError(f"{name} descriptor is not created")
    
    return ptr


def select_matmul_algorithm(
    runtime: CublasLtRuntime,
    *,
    operation_desc: MatmulDesc,
    a_layout: MatrixLayout,
    b_layout: MatrixLayout,
    c_layout: MatrixLayout,
    d_layout: MatrixLayout,
    preference: MatmulPreference,
    requested_algo_count: int = 1,
    result_stride_bytes: int = 4096,
    workspace_limit_bytes: int = 0,
) -> RawMatmulHeuristic:
    """
    Query cuBLASLt heuristic and return an opaque algorithm handle.

    We intentionally store heuristic results as raw bytes instead of declaring
    cublasLtMatmulHeuristicResult_t as a ctypes.Structure, because the embedded
    cublasLtMatmulAlgo_t is an opaque structure.
    """

    if requested_algo_count <= 0:
        raise ValueError("requested_algo_count must be positive")
    
    if result_stride_bytes <= 0:
        raise ValueError("result_stride_bytes must be positive")
    
    if workspace_limit_bytes < 0:
        raise ValueError("workspace_limit_bytes must be >= 0")
    
    cublaslt = runtime._require_cublaslt()
    handle = runtime._require_handle()

    operation_ptr = _require_descriptor_ptr(operation_desc, "operation")
    a_ptr = _require_descriptor_ptr(a_layout, "A")
    b_ptr = _require_descriptor_ptr(b_layout, "B")
    c_ptr = _require_descriptor_ptr(c_layout, "C")
    d_ptr = _require_descriptor_ptr(d_layout, "D")
    preference_ptr = _require_descriptor_ptr(preference, "preference")

    result_buffer = ctypes.create_string_buffer(
        result_stride_bytes * requested_algo_count
    )
    return_algo_count = ctypes.c_int(0)

    status = cublaslt.cublasLtMatmulAlgoGetHeuristic(
        handle,
        operation_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        d_ptr,
        preference_ptr,
        requested_algo_count,
        ctypes.cast(result_buffer, ctypes.c_void_p),
        ctypes.byref(return_algo_count),
    )
    check_cublas_status(status, "cublasLtMatmulAlgoGetHeuristic")

    if return_algo_count.value <= 0:
        raise RuntimeError("cublasLtMatmulAlgoGetHeuristic returned no algorithms")
        
    check_buffer = ctypes.create_string_buffer(result_stride_bytes)

    heuristic = RawMatmulHeuristic(
        requested_algo_count=requested_algo_count,
        returned_algo_count=int(return_algo_count.value),
        result_stride_bytes=result_stride_bytes,
        result_buffer=result_buffer,
        check_buffer=check_buffer,
        workspace_limit_bytes=workspace_limit_bytes,
    )

    status = cublaslt.cublasLtMatmulAlgoCheck(
        handle,
        operation_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        d_ptr,
        heuristic.algo_ptr,
        heuristic.check_result_ptr,
    )
    check_cublas_status(status, "cublasLtMatmulAlgoCheck")

    return heuristic