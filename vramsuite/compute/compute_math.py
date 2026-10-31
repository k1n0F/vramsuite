from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
from typing import Any

from vramsuite.backends.cuda import CudaMemcpyKind, CudaRuntime
from vramsuite.backends.cublas import CublasRuntime
from vramsuite.compute.validation import max_abs_error


@dataclass(frozen=True)
class SgemmSanityResult:
    passed: bool
    max_abs_error: float
    actual_column_major: list[float]
    expected_column_major: list[float]
    actual_row_major: list[list[float]]
    expected_row_major: list[list[float]]


def _column_major_2x2_to_row_major(values: list[float]) -> list[list[float]]:
    if len(values) != 4:
        raise ValueError("Expected exactly 4 values for a 2x2 matrix")

    return [
        [values[0], values[2]], 
        [values[1], values[3]],
    ]

def run_sgemm_sanity_check(tolerance: float = 1e-5) -> dict[str, Any]:
    """
    Matrices are stored in column-major order because classic cuBLAS
    follows BLAS layout conventions.

    A row-major view:
      [[1, 2],
       [3, 4]]


    B row_major view:
      [[5, 6],
       [7, 8]]

    Expected C = A @ B:
      [[19, 22],
       [43, 50]]
    """

    matrix_size = 4
    element_size = ctypes.sizeof(ctypes.c_float)
    size_bytes = matrix_size * element_size

    FloatArray4 = ctypes.c_float * matrix_size
    
    host_a = FloatArray4(
        1.0, 3.0,
        2.0, 4.0,
    )
    host_b = FloatArray4(
        5.0, 7.0,
        6.0, 8.0,
    )
    host_c = FloatArray4(
        0.0, 0.0,
        0.0, 0.0,
    )

    expected_column_major = [
        19.0, 43.0,
        22.0, 50.0,
    ]

    a_ptr = None
    b_ptr = None
    c_ptr = None

    with CudaRuntime() as cuda, CublasRuntime() as cublas:
        try:
            a_ptr = cuda.malloc(size_bytes)
            b_ptr = cuda.malloc(size_bytes)
            c_ptr = cuda.malloc(size_bytes)

            cuda.memcpy(
                a_ptr,
                ctypes.cast(host_a, ctypes.c_void_p),
                size_bytes,
                CudaMemcpyKind.HOST_TO_DEVICE,
            )
            cuda.memcpy(
                b_ptr,
                ctypes.cast(host_b, ctypes.c_void_p),
                size_bytes,
                CudaMemcpyKind.HOST_TO_DEVICE,
            )
            cuda.memcpy(
                c_ptr,
                ctypes.cast(host_c, ctypes.c_void_p),
                size_bytes,
                CudaMemcpyKind.HOST_TO_DEVICE,
            )

            cublas.sgemm(
                m=2,
                n=2,
                k=2,
                alpha=1.0,
                a_ptr=a_ptr,
                lda=2,
                b_ptr=b_ptr,
                ldb=2,
                beta=0.0,
                c_ptr=c_ptr,
                ldc=2,
            )

            cuda.synchronize()

            cuda.memcpy(
                ctypes.cast(host_c, ctypes.c_void_p),
                c_ptr,
                size_bytes,
                CudaMemcpyKind.DEVICE_TO_HOST,
            )
            cuda.synchronize()

        finally:
            cuda.free_all(c_ptr, b_ptr, a_ptr)

    actual_column_major = [float(value) for value in host_c]
    error = max_abs_error(actual_column_major, expected_column_major)

    result = SgemmSanityResult(
        passed=error <= tolerance,
        max_abs_error=error,
        actual_column_major=actual_column_major,
        expected_column_major=expected_column_major,
        actual_row_major=_column_major_2x2_to_row_major(actual_column_major),
        expected_row_major=_column_major_2x2_to_row_major(expected_column_major),
    )

    return asdict(result)