"""
VRAM Suite safety policies.

This module contains conservative limits used by probes and estimators.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbePolicy:
    default_max_probe_mb: int = 1024
    default_step_mb: int = 128
    default_free_floor_mb: int = 2048
    max_free_ratio: float = 0.90
    min_step_mb: int = 64
    max_step_mb: int = 1024
    max_probe_warning_mb: int = 12288


DEFAULT_PROBE_POLICY = ProbePolicy()