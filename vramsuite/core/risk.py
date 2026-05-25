"""
OOM risk estimation helpers.

This module estimates whether a requested VRAM amount is likely to fit
inside the currently known safe allocatable memory budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RiskEstimate:
    available: bool
    required_mb: int
    available_mb: int | None
    availability_source: str
    remaining_mb: int | None
    usage_ratio: float | None
    risk_level: str
    reason: str


def estimate_oom_risk(
        required_mb: int,
        memory_info: dict[str, Any],
) -> RiskEstimate:
    """
    Estimate OOM risk for required VRAM and know memory info.
    
    Priority:
    1. safe_allocatble_mb form probe
    2. driver_free_at_scan_mb for NVML
    """

    safe_allocatable_mb = memory_info.get("safe_allocatable_mb")
    driver_free_mb = memory_info.get("driver_free_at_scan_mb")

    if safe_allocatable_mb is not None:
        available_mb = safe_allocatable_mb
        availability_source = "probe_safe_allocatable"
    elif driver_free_mb is not None:
        available_mb = driver_free_mb
        availability_source = "driver_free_at_scan"
    else:
        available_mb = None
        availability_source = "unknown"


    if available_mb is None:
        return RiskEstimate(
            available=False,
            required_mb=required_mb,
            available_mb=None,
            availability_source=availability_source,
            remaining_mb=None,
            usage_ratio=None,
            risk_level="unknown",
            reason="No available memory value was found."
        )
    
    remaining_mb = available_mb - required_mb
    usage_ratio = required_mb / available_mb if available_mb > 0 else None

    if required_mb <= 0:
        risk_level = "unknown"
        reason = "Required memory must be greater than zero."
    elif usage_ratio is None:
        risk_level = "unknown"
        reason = "Cannot calculate usage ratio."
    elif usage_ratio < 0.70:
        risk_level = "low"
        reason = _build_reason(
            availability_source,
            "Required memory is well within available memory."
        )
    elif usage_ratio < 0.90:
        risk_level = "medium"
        reason = _build_reason(
            availability_source,
            "Required memory is close to available memory."
        )
    elif usage_ratio <= 1.0:
        risk_level = "high"
        reason = _build_reason(
            availability_source,
            "Required memory is very close to available memory."
        )
    else:
        risk_level = "critical"
        reason = _build_reason(
            availability_source,
            "Required memory exceeds available memory."
        )


    return RiskEstimate(
        available=True,
        required_mb=required_mb,
        available_mb=available_mb,
        availability_source=availability_source,
        remaining_mb=remaining_mb,
        usage_ratio=usage_ratio,
        risk_level=risk_level,
        reason=reason
    )


def _build_reason(availability_source: str, base_reason: str) -> str:
    """Build a human-readable risk reason based on the availability source."""

    if availability_source == "probe_safe_allocatable":
        return f"{base_reason} Based on confirmed safe probe budget."
    
    if availability_source == "driver_free_at_scan":
        return f"{base_reason} Based on driver free memory at scan time."

    return base_reason

def risk_estimate_to_dict(result: RiskEstimate) -> dict[str, Any]:
    "Convert RiskEstimate dataclass to a dictionary."
    return asdict(result)
        

        