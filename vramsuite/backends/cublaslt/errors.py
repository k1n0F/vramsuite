from __future__ import annotations


CUBLAS_STATUS_SUCCESS = 0


def decode_cublas_status(status: int) -> str:
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



def check_cublas_status(status: int, operation: str) -> None:
    if status == CUBLAS_STATUS_SUCCESS:
        return
    
    raise RuntimeError(f"{operation} failed: {decode_cublas_status(status)}")