"""
Workflow memory estimation: will these models fit, and what does offloading cost?

Weights are exact (read from .safetensors headers). Activations are not
derived from architecture formulas: they are either given, or calibrated from
a real run (peak VRAM of the process from `vramsuite processes --watch` minus
the weights). Whatever does not fit is assumed to be streamed from RAM once
per step, priced with the measured PCIe bandwidth of this machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vramsuite.backends.nvml import collect_nvml_info_dict
from vramsuite.core.model_inspect import inspect_model
from vramsuite.core.risk import estimate_oom_risk, risk_estimate_to_dict
from vramsuite.core.transfer import run_transfer_benchmark

BYTES_PER_MB = 1024 * 1024
# CUDA context, cuBLAS workspaces and allocator fragmentation.
DEFAULT_RESERVE_MB = 1024


def plan_memory(
        weights_mb: float,
        activations_mb: float,
        available_mb: float,
        reserve_mb: float,
) -> dict[str, Any]:
    """
    Split the workload into what stays in VRAM and what must be offloaded.

    Only weights can be offloaded; activations of the running step must be
    resident, so if they alone exceed the budget nothing helps.
    """
    budget_mb = available_mb - reserve_mb
    required_mb = weights_mb + activations_mb
    overflow_mb = max(required_mb - budget_mb, 0.0)

    if overflow_mb == 0:
        status = "fits"
    elif activations_mb <= budget_mb:
        status = "offload"
    else:
        status = "does_not_fit"

    return {
        "required_mb": round(required_mb, 1),
        "budget_mb": round(budget_mb, 1),
        "headroom_mb": round(max(budget_mb - required_mb, 0.0), 1),
        "overflow_mb": round(overflow_mb, 1),
        "offload_mb": round(min(overflow_mb, weights_mb), 1) if status == "offload" else 0.0,
        "status": status,
    }


def offload_seconds_per_step(offload_mb: float, gb_s: float | None) -> float | None:
    """Time to stream the offloaded weights over PCIe once (GB/s is 10^9 bytes/s)."""
    if not gb_s:
        return None
    return round(offload_mb * BYTES_PER_MB / (gb_s * 1e9), 3)


def _nvml_free_mb(device_index: int) -> float | None:
    nvml = collect_nvml_info_dict()
    for device in nvml.get("devices") or []:
        if device.get("index") == device_index:
            return device.get("free_vram_mb")
    return None


def _measure_transfer(device_index: int) -> dict[str, Any]:
    try:
        bench = run_transfer_benchmark(device_index=device_index, size_mb=128, repeats=3)
    except Exception as exc:
        return {"pinned_gb_s": None, "pageable_gb_s": None, "source": "measured", "error": str(exc)}

    host_to_device = {
        row["host_memory"]: row["best_gb_s"]
        for row in bench["results"]
        if row["direction"] == "host_to_device"
    }
    return {
        "pinned_gb_s": host_to_device.get("pinned"),
        "pageable_gb_s": host_to_device.get("pageable"),
        "source": "measured",
        "error": None,
    }


def estimate_workflow(
        model_paths: list[str | Path],
        activations_mb: float = 0.0,
        observed_peak_mb: float | None = None,
        available_mb: float | None = None,
        reserve_mb: float = DEFAULT_RESERVE_MB,
        transfer_gb_s: float | None = None,
        device_index: int = 0,
) -> dict[str, Any]:
    """
    Estimate VRAM for models that are resident at the same time.

    observed_peak_mb (peak of the process during a real run) overrides
    activations_mb: activations = peak - weights.
    """
    models = [inspect_model(path) for path in model_paths]
    weights_mb = sum(model["total_mb"] for model in models)

    if observed_peak_mb is not None:
        activations_mb = max(observed_peak_mb - weights_mb, 0.0)
        activations_source = "observed_peak"
    else:
        activations_source = "given" if activations_mb else "not_set"

    if available_mb is None:
        available_mb = _nvml_free_mb(device_index)
        available_source = "nvml_free"
    else:
        available_source = "given"

    if available_mb is None:
        raise RuntimeError(f"Cannot read free VRAM of GPU {device_index} through NVML; pass --available-mb")

    plan = plan_memory(weights_mb, activations_mb, available_mb, reserve_mb)
    risk = estimate_oom_risk(
        required_mb=round(plan["required_mb"]),
        memory_info={"driver_free_at_scan_mb": plan["budget_mb"]},
    )

    offload = None
    if plan["status"] == "offload":
        if transfer_gb_s is not None:
            transfer = {"pinned_gb_s": transfer_gb_s, "pageable_gb_s": None, "source": "given", "error": None}
        else:
            # Measured only when needed: it allocates 2 x 128 MB of VRAM.
            transfer = _measure_transfer(device_index)

        offload = {
            **transfer,
            "offload_mb": plan["offload_mb"],
            "seconds_per_step_pinned": offload_seconds_per_step(plan["offload_mb"], transfer["pinned_gb_s"]),
            "seconds_per_step_pageable": offload_seconds_per_step(plan["offload_mb"], transfer["pageable_gb_s"]),
        }

    return {
        "device_index": device_index,
        "models": [
            {
                "name": Path(model["path"]).name,
                "path": model["path"],
                "weights_mb": model["total_mb"],
                "dtypes": list(model["by_dtype"]),
            }
            for model in models
        ],
        "weights_mb": round(weights_mb, 1),
        "activations_mb": round(activations_mb, 1),
        "activations_source": activations_source,
        "available_mb": round(float(available_mb), 1),
        "available_source": available_source,
        "reserve_mb": reserve_mb,
        **plan,
        "risk": risk_estimate_to_dict(risk),
        "offload": offload,
    }
