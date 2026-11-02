"""
.vramcard validation: structure (sections and field types) and sanity
(do the numbers agree with each other).

Errors mean the card cannot be trusted for compare / estimate (broken or
hand-edited). Warnings are values that are possible but suspicious.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from vramsuite.core.compare import _get_nested
from vramsuite.core.vramcard import VRAMCARD_SCHEMA_VERSION

KNOWN_SCHEMAS = {VRAMCARD_SCHEMA_VERSION}
RISK_LEVELS = {"low", "medium", "high", "critical", "unknown"}

SECTIONS = ("gpu", "environment", "stack", "memory")
# Checked only when present and not null: older cards and cards from machines
# without a GPU legitimately miss some of them.
FIELD_TYPES: dict[str, type | tuple[type, ...]] = {
    "gpu.name": str,
    "gpu.total_vram_mb": (int, float),
    "gpu.compute_capability": str,
    "gpu.major": int,
    "gpu.minor": int,
    "gpu.multi_processor_count": int,
    "memory.driver_total_mb": (int, float),
    "memory.driver_free_at_scan_mb": (int, float),
    "memory.driver_used_at_scan_mb": (int, float),
    "memory.process_allocatable_mb": (int, float),
    "memory.safe_allocatable_mb": (int, float),
    "memory.safety_margin_mb": (int, float),
    "memory.cuda_runtime_free_mb": (int, float),
    "probe": dict,
    "risk_estimate": dict,
    "compute_check": dict,
    "device": dict,
    "known_issues": list,
}


def _has_type(value: Any, expected: type | tuple[type, ...]) -> bool:
    # bool is a subclass of int; True is not a VRAM size.
    return isinstance(value, expected) and not isinstance(value, bool)


def _structure_issues(card: dict[str, Any]) -> list[tuple[str, str, str]]:
    issues = []

    schema = card.get("schema")
    if not isinstance(schema, str) or not schema.startswith("vramcard@"):
        issues.append(("error", "schema", f"not a .vramcard (schema = {schema!r})"))
    elif schema not in KNOWN_SCHEMAS:
        issues.append(("warning", "schema", f"{schema} is not known to this version of VRAM Suite"))

    try:
        generated = datetime.fromisoformat(card.get("generated_at"))
        if generated.tzinfo and generated > datetime.now(timezone.utc):
            issues.append(("warning", "generated_at", "is in the future"))
    except (TypeError, ValueError):
        issues.append(("error", "generated_at", "missing or not an ISO 8601 timestamp"))

    for section in SECTIONS:
        if not isinstance(card.get(section), dict):
            issues.append(("error", section, "section is missing"))

    for path, expected in FIELD_TYPES.items():
        value = _get_nested(card, path)
        if value is not None and not _has_type(value, expected):
            issues.append(("error", path, f"has type {type(value).__name__}"))

    return issues


def _sanity_issues(card: dict[str, Any]) -> list[tuple[str, str, str]]:
    issues = []
    gpu = card.get("gpu") or {}
    memory = card.get("memory") or {}
    numbers = {key: value for key, value in memory.items() if _has_type(value, (int, float))}

    if gpu.get("name") is None:
        issues.append(("warning", "gpu.name", "no GPU was detected when the card was created"))

    cc = gpu.get("compute_capability")
    if isinstance(cc, str) and cc != f"{gpu.get('major')}.{gpu.get('minor')}":
        issues.append(("error", "gpu.compute_capability", f"{cc} disagrees with major/minor"))

    for key, value in numbers.items():
        if value < 0:
            issues.append(("error", f"memory.{key}", f"is negative ({value})"))

    total = numbers.get("driver_total_mb")
    free = numbers.get("driver_free_at_scan_mb")
    used = numbers.get("driver_used_at_scan_mb")
    if total:
        for key in ("driver_free_at_scan_mb", "driver_used_at_scan_mb", "cuda_runtime_free_mb"):
            if numbers.get(key, 0) > total:
                issues.append(("error", f"memory.{key}", f"{numbers[key]} MB is more than total {total} MB"))
        # NVML: total = used + free + a small driver reservation.
        if free is not None and used is not None and abs(free + used - total) > total * 0.02:
            issues.append(("warning", "memory", f"free {free} + used {used} MB is far from total {total} MB"))
        gpu_total = gpu.get("total_vram_mb")
        if _has_type(gpu_total, (int, float)) and abs(gpu_total - total) > total * 0.01:
            issues.append(("warning", "gpu.total_vram_mb", f"{gpu_total} MB disagrees with memory.driver_total_mb {total} MB"))

    process = numbers.get("process_allocatable_mb")
    safe = numbers.get("safe_allocatable_mb")
    if process is not None and safe is not None and safe > process:
        issues.append(("error", "memory.safe_allocatable_mb", f"{safe} MB is more than process_allocatable_mb {process} MB"))
    if process is not None and free is not None and process > free:
        issues.append(("warning", "memory.process_allocatable_mb", f"{process} MB is more than driver free {free} MB at scan"))

    probe = card.get("probe")
    if isinstance(probe, dict) and probe.get("safe_allocatable_mb") != memory.get("safe_allocatable_mb"):
        issues.append(("error", "memory.safe_allocatable_mb", "disagrees with probe.safe_allocatable_mb"))

    risk = card.get("risk_estimate")
    if isinstance(risk, dict) and risk.get("risk_level") not in RISK_LEVELS:
        issues.append(("error", "risk_estimate.risk_level", f"unknown level {risk.get('risk_level')!r}"))

    return issues


def validate_vramcard(card: Any) -> dict[str, Any]:
    """Check a loaded .vramcard. valid is False when there is at least one error."""
    if not isinstance(card, dict):
        issues = [("error", "", "a .vramcard is a JSON object")]
    else:
        issues = _structure_issues(card)
        # Sanity checks assume the structure holds.
        if not any(level == "error" for level, _, _ in issues):
            issues += _sanity_issues(card)

    return {
        "schema": card.get("schema") if isinstance(card, dict) else None,
        "valid": not any(level == "error" for level, _, _ in issues),
        "issues": [{"level": level, "field": field, "message": message} for level, field, message in issues],
    }
