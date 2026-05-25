"""
VRAMcard comparison helpers.

This module compares two .vramcard profles and returns structured
differences between important runtime, GPU, memory, probe and risk fields.
"""


from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CompareRow:
    section: str
    field: str
    left: Any
    right: Any
    delta: Any
    changed: bool


def _get_nested(data: dict[str, Any], path:str) -> Any:
    """
    Read a nested value from a dictionary using dot notation.

    Example:
        _get_nested(card, "memory.driver_free_at_scan_mb")
    """

    current: Any = data

    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        
        current = current.get(part)

        if current is None:
            return None
        
    return current


def _calculate_delta(left: Any, right: Any) -> Any:
    """
    Calculate numeric delta when possible.

    For numbers:
        delsta = right - left

    For non-numberic values:
        delta = None
    """

    if isinstance(left, bool) or isinstance(right, bool):
        return None
    
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return right - left
    
    return None



def _make_row(
        section: str,
        field: str,
        left_card: dict[str, Any],
        right_card: dict[str, Any],
        path: str,
) -> CompareRow:
    left = _get_nested(left_card, path)
    right = _get_nested(right_card, path)

    return CompareRow(
        section=section,
        field=field,
        left=left,
        right=right,
        delta=_calculate_delta(left, right),
        changed=left != right,
    )



def compare_vramcards(
        left_card: dict[str, Any],
        right_card: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare two .vramcards dictionaries.

    Returns structured comparison data.
    """

    fields: list[tuple[str, str, str]] = [
        # Runtime
        ("Runtime", "OS", "environment.os_name"),
        ("Runtime", "Platform", "environment.platform"),
        ("Runtime", "Python Version", "environment.python_version"),
        ("Runtime", "WSL", "environment.is_wsl"),
        ("Runtime", "Container", "environment.is_container"),

        # GPU
        ("GPU", "Name", "gpu.name"),
        ("GPU", "Total VRAM MB", "gpu.total_vram_mb"),
        ("GPU", "Compute Capability", "gpu.compute_capability"),
        ("GPU", "Source", "gpu.source"),


        # Memory
        ("Memory", "Driver Total MB", "memory.driver_total_mb"),
        ("Memory", "Driver Free at Scan MB", "memory.driver_free_at_scan_mb"),
        ("Memory", "Driver Used at Scan MB", "memory.driver_used_at_scan_mb"),
        ("Memory", "Process Allocatable MB", "memory.process_allocatable_mb"),
        ("Memory", "Safe Allocatable MB", "memory.safe_allocatable_mb"),
        ("Memory", "Safety Margin MB", "memory.safety_margin_mb"),
        ("Memory", "Source", "memory.source"),

        # Probe
        ("Probe", "Available", "probe.available"),
        ("Probe", "Backend", "probe.backend"),
        ("Probe", "Attempted MB", "probe.attempted_mb"),
        ("Probe", "Allocated MB", "probe.allocated_mb"),
        ("Probe", "Safe Allocatable MB", "probe.safe_allocatable_mb"),
        ("Probe", "Safety Margin MB", "probe.safety_margin_mb"),
        ("Probe", "Error", "probe.error"),

        # Risk
        ("Risk", "Available", "risk_estimate.available"),
        ("Risk", "Required MB", "risk_estimate.required_mb"),
        ("Risk", "Available MB", "risk_estimate.available_mb"),
        ("Risk", "Availability Source", "risk_estimate.availability_source"),
        ("Risk", "Remaining MB", "risk_estimate.remaining_mb"),
        ("Risk", "Usage Ratio", "risk_estimate.usage_ratio"),
        ("Risk", "Risk Level", "risk_estimate.risk_level"),
    ]


    rows = [
        _make_row(
            section=section,
            field=field,
            left_card=left_card,
            right_card=right_card,
            path=path,
        )
        for section, field, path in fields

    ]

    changed_rows = [row for row in rows if row.changed]

    return {
        "changed": bool(changed_rows),
        "total_rows": len(rows),
        "changed_rows": len(changed_rows),
        "rows": [asdict(row) for row in rows],
    }
