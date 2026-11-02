"""
Structured doctor diagnostics API.

This module provides the internal/public API used by the CLI and by
external Python callers. It does not print anything by itself.
"""

from __future__ import annotations

from typing import Any


from vramsuite.backends.native_loader import NativeLibraryLoadError
from vramsuite.core.fingerprint import collect_fingerprint
from vramsuite.core.vramcard import create_vramcard
from vramsuite.core.probe import probe_result_to_dict, run_cuda_probe
from vramsuite.core.risk import estimate_oom_risk, risk_estimate_to_dict


TOOLKIT_MISSING = "CUDA Toolkit libraries not found (cudart / cuBLAS)"


def run_compute_checks() -> dict[str, Any]:
    """
    Native correctness checks of the compute stack (no torch).

    Each check is isolated: one failing library (e.g. no FP8 on pre-Ada GPUs)
    is recorded as an error and does not hide the others.
    """
    from vramsuite.compute.compute_math import run_sgemm_sanity_check
    from vramsuite.compute.cublaslt_math import (
        run_cublaslt_fp8_gemm_check,
        run_cublaslt_identity_gemm_check,
    )

    checks = {
        "cublas_sgemm_fp32": run_sgemm_sanity_check,
        "cublaslt_gemm_fp32": lambda: run_cublaslt_identity_gemm_check(size=256),
        "cublaslt_gemm_fp8_e4m3": lambda: run_cublaslt_fp8_gemm_check(size=128),
    }

    results: dict[str, Any] = {}
    for name, check in checks.items():
        try:
            result = check()
            results[name] = {
                "passed": result["passed"],
                "max_abs_error": result["max_abs_error"],
                "error": None,
            }
        except NativeLibraryLoadError:
            results[name] = {"passed": False, "max_abs_error": None, "error": TOOLKIT_MISSING}
        except Exception as exc:
            results[name] = {"passed": False, "max_abs_error": None, "error": str(exc)}

    return results

def run_doctor(
        with_probe: bool = False,
        probe_max_mb: int = 1024,
        probe_step_mb: int = 128,
        probe_floor_mb: int = 2048,
        probe_max_free_ratio: float = 0.9,
        estimate_mb: int | None = None,
        with_compute_check: bool = False,
) -> dict[str, Any]:
    """
    Run VRAM Suite doctor diagnostics and return structured data.

    This function does not print terminal tables and does not save files.
    It only collects diagnostic data and returns it as dictionaries.

    Returns:
        A dictionary containing:
        - fingerprint: raw runtime/torch/nvml fingerprint
        - vramcard: generated .vramcard dictionary
        - runtime: runtime information
        - torch: optional PyTorch/CUDA information
        - nvml: NVML driver memory information
        - gpu: selected primary GPU profile
        - memory: selected memory profile
    """

    fingerprint = collect_fingerprint()
    vramcard = create_vramcard(fingerprint=fingerprint)

    runtime_info = fingerprint.get("runtime", {})
    torch_info = fingerprint.get("torch", {})
    nvml_info = fingerprint.get("nvml", {})
    gpu_info = vramcard.get("gpu", {})
    memory_info = vramcard.get("memory", {})

    probe_info: dict[str, Any] | None = None
    risk_info: dict[str, Any] | None = None

    if with_probe:
        probe_result = run_cuda_probe(
            driver_free_mb=memory_info.get("driver_free_at_scan_mb"),
            max_probe_mb=probe_max_mb,
            step_mb=probe_step_mb,
            hard_free_floor_mb=probe_floor_mb,
            max_free_ratio=probe_max_free_ratio,
        )

        probe_info = probe_result_to_dict(probe_result)

        memory_info["process_allocatable_mb"] = probe_info.get("allocated_mb")
        memory_info["safe_allocatable_mb"] = probe_info.get("safe_allocatable_mb")
        memory_info["safety_margin_mb"] = probe_info.get("safety_margin_mb")

        vramcard["memory"] = memory_info
        vramcard["probe"] = probe_info


    compute_info: dict[str, Any] | None = None

    if with_compute_check:
        compute_info = run_compute_checks()
        vramcard["compute_check"] = compute_info

    if estimate_mb is not None:
        risk_result = estimate_oom_risk(
            required_mb=estimate_mb,
            memory_info=memory_info,
        )
        risk_info = risk_estimate_to_dict(risk_result)
        vramcard["risk_estimate"] = risk_info
    
    return {
            "fingerprint": fingerprint,
            "vramcard": vramcard,
            "runtime": runtime_info,
            "torch": torch_info,
            "nvml": nvml_info,
            "gpu": gpu_info,
            "memory": memory_info,
            "probe": probe_info,
            "risk_estimate": risk_info,
            "compute_check": compute_info,
        }
