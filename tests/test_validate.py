import copy

from vramsuite.core.validate import validate_vramcard
from vramsuite.core.vramcard import VRAMCARD_SCHEMA_VERSION

CARD = {
    "schema": VRAMCARD_SCHEMA_VERSION,
    "generated_at": "2026-06-24T21:35:11.342200+00:00",
    "gpu": {"name": "RTX", "total_vram_mb": 16303, "compute_capability": "12.0", "major": 12, "minor": 0},
    "environment": {},
    "stack": {},
    "memory": {
        "driver_total_mb": 16303,
        "driver_free_at_scan_mb": 8406,
        "driver_used_at_scan_mb": 7896,
        "process_allocatable_mb": 6144,
        "safe_allocatable_mb": 5222,
    },
    "probe": {"safe_allocatable_mb": 5222},
    "risk_estimate": {"risk_level": "critical"},
}


def fields(card, level):
    return [issue["field"] for issue in validate_vramcard(card)["issues"] if issue["level"] == level]


def test_real_card_is_valid():
    assert validate_vramcard(CARD) == {"schema": VRAMCARD_SCHEMA_VERSION, "valid": True, "issues": []}


def test_contradicting_numbers():
    card = copy.deepcopy(CARD)
    card["memory"]["safe_allocatable_mb"] = 7000  # more than process allocatable, and != probe
    card["memory"]["driver_free_at_scan_mb"] = 17000
    card["gpu"]["compute_capability"] = "8.9"
    assert sorted(fields(card, "error")) == [
        "gpu.compute_capability",
        "memory.driver_free_at_scan_mb",
        "memory.safe_allocatable_mb",
        "memory.safe_allocatable_mb",
    ]


def test_broken_structure_skips_sanity():
    card = copy.deepcopy(CARD)
    del card["memory"]
    card["gpu"]["total_vram_mb"] = True
    card["schema"] = "vramcard@9.9"
    assert fields(card, "error") == ["memory", "gpu.total_vram_mb"]
    assert fields(card, "warning") == ["schema"]
    assert validate_vramcard([])["valid"] is False
