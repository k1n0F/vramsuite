"""
Runtime detection module.

Planned for v0.1-alpha:
- detect OS
- detect Python version
- detect WSL
- detect basic environment metadata
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class RuntimeInfo:
    os_name: str
    os_version: str
    platform: str
    python_version: str
    python_executable: str
    is_wsl: bool
    is_container: bool


def detect_wsl() -> bool:
    """Return True if the current process appears to run inside WSL."""
    if platform.system().lower() != "linux":
        return False

    try:
        release = platform.uname().release.lower()
        version = platform.uname().version.lower()
        return "microsoft" in release or "microsoft" in version or "wsl" in release
    except Exception:
        return False

def detection_container() -> bool:
    """Best-effort container detection."""
    if os.path.exists("/.dockerenv"):
        return True

    try:
        with open("/proc/1/cgroup", "r", encoding="utf-8", errors="ignore") as file:
            content = file.read().lower()
        return "docker" in content or "containered" in content or "kubepods" in content
    except Exception:
        return False

def collect_runtime_info() -> RuntimeInfo:
    """Collect basic runtime information."""
    return RuntimeInfo(
        os_name=platform.system(),
        os_version=platform.version(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        python_executable=sys.executable,
        is_wsl=detect_wsl(),
        is_container=detection_container(),
    )


def runtime_info_to_dict(info: RuntimeInfo) -> dict:
    """Convert RuntimeInfo to a JSON-serializable dictionary."""
    return asdict(info)