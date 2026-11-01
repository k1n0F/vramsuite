from __future__ import annotations

import ctypes

from vramsuite.backends.cublaslt.descriptors import MatmulDesc, MatrixLayout
from vramsuite.backends.cublaslt.errors import check_cublas_status
from vramsuite.backends.cublaslt.heuristics import RawMatmulHeuristic, _require_descriptor_ptr
from vramsuite.backends.cublaslt.runtime import CublasLtRuntime


def run_cublaslt_matmul(
        runtime: CublasLtRuntime,
        *,
        operation_desc: MatmulDesc,
        a_ptr: ctypes.c_void_p,
        a_layout: MatrixLayout,
        b_ptr: ctypes.c_void_p,
        b_layout: MatrixLayout,
        c_ptr: ctypes.c_void_p,
        c_layout: MatrixLayout,
        d_ptr: ctypes.c_void_p,
        d_layout: MatrixLayout,
        heuristic: RawMatmulHeuristic,
        alpha: float = 1.0,
        beta: float = 0.0,
        workspace_ptr: ctypes.c_void_p | None = None,
        workspace_size_bytes: int = 0,
        stream: ctypes.c_void_p | None = None,
) -> None:
    if workspace_size_bytes < 0:
        raise ValueError("workspace_size_bytes must be >= 0")
    
    cublaslt = runtime._require_cublaslt()
    handle = runtime._require_handle()

    operation_ptr = _require_descriptor_ptr(operation_desc, "operation")
    a_layout_ptr = _require_descriptor_ptr(a_layout, "A")
    b_layout_ptr = _require_descriptor_ptr(b_layout, "B")
    c_layout_ptr = _require_descriptor_ptr(c_layout, "C")
    d_layout_ptr = _require_descriptor_ptr(d_layout, "D")

    alpha_value = ctypes.c_float(alpha)
    beta_value = ctypes.c_float(beta)

    alpha_ptr = ctypes.cast(ctypes.byref(alpha_value), ctypes.c_void_p)
    beta_ptr = ctypes.cast(ctypes.byref(beta_value), ctypes.c_void_p)

    workspace = workspace_ptr if workspace_ptr is not None else ctypes.c_void_p()
    cuda_stream = stream if stream is not None else ctypes.c_void_p()

    status = cublaslt.cublasLtMatmul(
        handle,
        operation_ptr,
        alpha_ptr,
        a_ptr,
        a_layout_ptr,
        b_ptr,
        b_layout_ptr,
        beta_ptr,
        c_ptr,
        c_layout_ptr,
        d_ptr,
        d_layout_ptr,
        heuristic.algo_ptr,
        workspace,
        workspace_size_bytes,
        cuda_stream,
    )
    check_cublas_status(status, "cublasLtMatmul")