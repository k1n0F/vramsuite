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
    run_cublaslt_matmul,
    select_matmul_algorithm,
)
from vramsuite.backends.cublaslt.enums import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    CUDA_R_8F_E4M3,
    CUDA_R_8F_E5M2,
)
from vramsuite.compute.minifloat import encode_bytes
from vramsuite.compute.validation import max_abs_error


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


def _run_lt_matmul(
        *,
        layout_spec: MatmulLayoutSpec,
        operation_desc_kwargs: dict[str, int],
        host_a: Any,
        host_b: Any,
        host_d: Any,
        workspace_bytes: int,
) -> None:
    """
    D = op(A) @ op(B) on the GPU, with beta = 0 and C aliased to D
    (cuBLASLt allows in-place when C and D share pointer and layout).

    host_* are ctypes arrays already encoded in the layout's data types.
    """

    a_bytes = ctypes.sizeof(host_a)
    b_bytes = ctypes.sizeof(host_b)
    d_bytes = ctypes.sizeof(host_d)

    a_ptr = None
    b_ptr = None
    d_ptr = None

    with CudaRuntime() as cuda, CublasLtRuntime() as rt:
        with (
            MatmulDesc(rt, **operation_desc_kwargs) as op,
            layout_spec.a.create(rt) as a_layout,
            layout_spec.b.create(rt) as b_layout,
            layout_spec.d.create(rt) as d_layout,
            MatmulPreference(rt, max_workspace_bytes=workspace_bytes) as pref,
        ):
            heuristic = select_matmul_algorithm(
                rt,
                operation_desc=op,
                a_layout=a_layout,
                b_layout=b_layout,
                c_layout=d_layout,
                d_layout=d_layout,
                preference=pref,
                requested_algo_count=1,
                workspace_limit_bytes=workspace_bytes,
            )

            try:
                a_ptr = cuda.malloc(a_bytes)
                b_ptr = cuda.malloc(b_bytes)
                d_ptr = cuda.malloc(d_bytes)

                cuda.memcpy(
                    a_ptr,
                    ctypes.cast(host_a, ctypes.c_void_p),
                    a_bytes,
                    CudaMemcpyKind.HOST_TO_DEVICE,
                )
                cuda.memcpy(
                    b_ptr,
                    ctypes.cast(host_b, ctypes.c_void_p),
                    b_bytes,
                    CudaMemcpyKind.HOST_TO_DEVICE,
                )

                with DeviceWorkspace(cuda, size_bytes=workspace_bytes) as workspace:
                    run_cublaslt_matmul(
                        rt,
                        operation_desc=op,
                        a_ptr=a_ptr,
                        a_layout=a_layout,
                        b_ptr=b_ptr,
                        b_layout=b_layout,
                        c_ptr=d_ptr,
                        c_layout=d_layout,
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
                    d_bytes,
                    CudaMemcpyKind.DEVICE_TO_HOST,
                )
                cuda.synchronize()

            finally:
                cuda.free_all(d_ptr, b_ptr, a_ptr)


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

    FloatArray = ctypes.c_float * matrix_element

    host_a = FloatArray()
    host_b = FloatArray()
    host_d = FloatArray()

    _fill_column_major_matrix_a(host_a, size)
    _fill_column_major_identity(host_b, size)
    
    layout_spec = MatmulLayoutSpec.fp32_column_major_nn(
        m=size,
        n=size,
        k=size,
    )

    _run_lt_matmul(
        layout_spec=layout_spec,
        operation_desc_kwargs={},
        host_a=host_a,
        host_b=host_b,
        host_d=host_d,
        workspace_bytes=workspace_bytes,
    )

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

_FP8_FORMATS = {
    "e4m3": CUDA_R_8F_E4M3,
    "e5m2": CUDA_R_8F_E5M2,
}


def _small_int_pattern(rows: int, cols: int, row_mul: int, col_mul: int) -> list[float]:
    """Column-major ints in [-8, 8]: exact in E4M3 and E5M2, sums exact in FP32."""
    return [
        float(((row * row_mul + col * col_mul) % 17) - 8)
        for col in range(cols)
        for row in range(rows)
    ]


def run_cublaslt_fp8_gemm_check(
        *,
        size: int = 64,
        a_fmt: str = "e4m3",
        b_fmt: str = "e4m3",
        tolerance: float = 1e-3,
        workspace_bytes: int = 32 * 1024 * 1024,
) -> dict[str, Any]:
    """
    Runs a real cuBLASLt FP8 GEMM (FP8 inputs, FP32 accumulate and output):

        D = A^T @ B      (TN, the layout cuBLASLt requires for FP8)

    Inputs are small integers, so the exact FP64 host reference is reachable
    and any error means a wrong layout, type or kernel, not rounding.
    Requires sm_89+ (Ada, Hopper, Blackwell).
    """

    for name, fmt in (("a_fmt", a_fmt), ("b_fmt", b_fmt)):
        if fmt not in _FP8_FORMATS:
            raise ValueError(f"{name} must be one of {sorted(_FP8_FORMATS)}, got {fmt!r}")
    if a_fmt == b_fmt == "e5m2":
        # cuBLASLt has no E5M2 x E5M2 kernels; E5M2 is meant for gradients.
        raise ValueError("E5M2 x E5M2 is not supported by cuBLASLt; use E4M3 for A or B")
    if size <= 0 or size % 16:
        raise ValueError("size must be a positive multiple of 16 for FP8")

    m = n = k = size
    a_values = _small_int_pattern(k, m, row_mul=7, col_mul=3)   # stored k x m
    b_values = _small_int_pattern(k, n, row_mul=5, col_mul=11)  # stored k x n

    ByteArray = ctypes.c_uint8 * (k * m)
    host_a = ByteArray.from_buffer_copy(encode_bytes(a_values, a_fmt))
    host_b = (ctypes.c_uint8 * (k * n)).from_buffer_copy(encode_bytes(b_values, b_fmt))
    host_d = (ctypes.c_float * (m * n))()

    layout_spec = MatmulLayoutSpec.fp8_column_major_tn(
        m=m,
        n=n,
        k=k,
        a_type=_FP8_FORMATS[a_fmt],
        b_type=_FP8_FORMATS[b_fmt],
    )

    _run_lt_matmul(
        layout_spec=layout_spec,
        operation_desc_kwargs={"transa": CUBLAS_OP_T, "transb": CUBLAS_OP_N},
        host_a=host_a,
        host_b=host_b,
        host_d=host_d,
        workspace_bytes=workspace_bytes,
    )

    # D[i, j] = sum_p A[p, i] * B[p, j], everything column-major.
    expected = [
        sum(a_values[p + i * k] * b_values[p + j * k] for p in range(k))
        for j in range(n)
        for i in range(m)
    ]
    actual = [float(value) for value in host_d]
    max_error = max_abs_error(actual, expected)

    return {
        "passed": max_error <= tolerance,
        "a_format": a_fmt,
        "b_format": b_fmt,
        "size": size,
        "max_abs_error": max_error,
        "tolerance": tolerance,
        "workspace_bytes": workspace_bytes,
        "flops_estimate": 2 * m * n * k,
        "sample_actual": _sample_values(actual, len(actual)),
        "sample_expected": _sample_values(expected, len(expected)),
    }
