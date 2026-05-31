from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
from typing import Any

from vramsuite.backends.cuda import CudaMemcpyKind, CudaRuntime
from vramsuite.backends.cublaslt import (
    CublasLtRuntime,
    DeviceWorkspace,
    MatmulDesc,
    MatmulLayoutSpec,
    MatmulPreference,
    run_cublaslt_matmul_fp32,
    select_matmul_algorithm,
)


@dataclass(frozen=True)
class CublasLtIdentityGemmResult:
    passed: bool
    size: int
    max_abs_error: float
    tolerance: float
    workspace_bytes: int
    flops_estimate: int
    sample_actual: list[float]
    sample_expected: list[float]


def _matrix_element(row: int, col: int) -> float:
    return float((row % 17) + (col % 13) * 0.01 + 1.0)


def _fill_column_major_matrix_a(buffer: Any, size: int) -> None:
    for col in range(size):
        for row in range(size):
            buffer[row + col * size] = _matrix_element(row, col)


def _fill_column_major_identity(buffer: Any, size: int) -> None:
    for col in range(size):
        for row in range(size):
            buffer[row + col * size] = 1.0 if row == col else 0.0



def _max_abs_error_against_a(actual: Any, expected_a: Any, count: int) -> float:
    max_error = 0.0

    for index in range(count):
        error = abs(float(actual[index]) - float(expected_a[index]))
        if error > max_error:
            max_error = error

    return max_error


def _sample_values(buffer: Any, count: int, sample_count: int = 8) -> list[float]:
    if count <= 0:
        return []
    
    step = max(count // sample_count, 1)
    value = []

    for index in range(0, count, step):
        value.append(float(buffer[index]))
        if len(value) >= sample_count:
            break

    return value


def run_cublaslt_identity_gemm_check(
        *,
        size: int = 512,
        tolerance: float = 1e-4,
        workspace_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    """
    Runs a real cuBLASLt FP32 GEMM:

        D = A @ I

    This validates the full cuBLASLt execution path while avoiding a slow
    python-side o(N^3) reference GEMM 
    """

    if size <= 0:
        raise ValueError("size must be positive")
    if workspace_bytes < 0:
        raise ValueError("workspace_bytes must be >= 0")
    
    matrix_element = size * size
    element_size = ctypes.sizeof(ctypes.c_float)
    matrix_size_bytes = matrix_element * element_size

    FloatArray = ctypes.c_float * matrix_element

    host_a = FloatArray()
    host_b = FloatArray()
    host_c = FloatArray()
    host_d = FloatArray()

    _fill_column_major_matrix_a(host_a, size)
    _fill_column_major_identity(host_b, size)
    
    layout_spec = MatmulLayoutSpec.fp32_column_major_nn(
        m=size,
        n=size,
        k=size,
    )

    a_ptr = None
    b_ptr = None
    c_ptr = None
    d_ptr = None

    with CudaRuntime() as cuda, CublasLtRuntime() as rt:
        with (
            MatmulDesc(rt) as op,
            layout_spec.a.create(rt) as a_layout,
            layout_spec.b.create(rt) as b_layout,
            layout_spec.c.create(rt) as c_layout,
            layout_spec.d.create(rt) as d_layout,
            MatmulPreference(rt, max_workspace_bytes=workspace_bytes) as pref,
        ):
            heuristic = select_matmul_algorithm(
                rt,
                operation_desc=op,
                a_layout=a_layout,
                b_layout=b_layout,
                c_layout=c_layout,
                d_layout=d_layout,
                preference=pref,
                requested_algo_count=1,
                workspace_limit_bytes=workspace_bytes,
            )

            try:
                a_ptr = cuda.malloc(matrix_size_bytes)
                b_ptr = cuda.malloc(matrix_size_bytes)
                c_ptr = cuda.malloc(matrix_size_bytes)
                d_ptr = cuda.malloc(matrix_size_bytes)

                cuda.memcpy(
                    a_ptr,
                    ctypes.cast(host_a, ctypes.c_void_p),
                    matrix_size_bytes,
                    CudaMemcpyKind.HOST_TO_DEVICE,
                )
                cuda.memcpy(
                    b_ptr,
                    ctypes.cast(host_b, ctypes.c_void_p),
                    matrix_size_bytes,
                    CudaMemcpyKind.HOST_TO_DEVICE,
                )
                cuda.memcpy(
                    c_ptr,
                    ctypes.cast(host_c, ctypes.c_void_p),
                    matrix_size_bytes,
                    CudaMemcpyKind.HOST_TO_DEVICE,
                )

                with DeviceWorkspace(cuda, size_bytes=workspace_bytes) as workspace:
                    run_cublaslt_matmul_fp32(
                        rt,
                        operation_desc=op,
                        a_ptr=a_ptr,
                        a_layout=a_layout,
                        b_ptr=b_ptr,
                        b_layout=b_layout,
                        c_ptr=c_ptr,
                        c_layout=c_layout,
                        d_ptr=d_ptr,
                        d_layout=d_layout,
                        heuristic=heuristic,
                        alpha=1.0,
                        beta=0.0,
                        workspace_ptr=workspace.raw_ptr,
                        workspace_size_bytes=workspace.size_bytes,
                    )

                cuda.synchronize()

                cuda.memcpy(
                    ctypes.cast(host_d, ctypes.c_void_p),
                    d_ptr,
                    matrix_size_bytes,
                    CudaMemcpyKind.DEVICE_TO_HOST,
                )
                cuda.synchronize()

            finally:
                if d_ptr is not None:
                    cuda.free(d_ptr)
                if c_ptr is not None:
                    cuda.free(c_ptr)
                if b_ptr is not None:
                    cuda.free(b_ptr)
                if a_ptr is not None:
                    cuda.free(a_ptr)

    max_error = _max_abs_error_against_a(host_d, host_a, matrix_element)

    result = CublasLtIdentityGemmResult(
        passed=max_error <= tolerance,
        size=size,
        max_abs_error=max_error,
        tolerance=tolerance,
        workspace_bytes=workspace_bytes,
        flops_estimate=2 * size * size * size,
        sample_actual=_sample_values(host_d, matrix_element),
        sample_expected=_sample_values(host_a, matrix_element),
    )
    

    return asdict(result)