from __future__ import annotations

import importlib.metadata

from vramsuite.core.doctor import run_doctor
from vramsuite.core.fingerprint import collect_fingerprint
from vramsuite.core.vramcard import create_vramcard, load_vramcard, save_vramcard

__version__ = importlib.metadata.version("vramsuite")

__all__ = [
    "__version__",
    "run_doctor",
    "collect_fingerprint",
    "create_vramcard",
    "save_vramcard",
    "load_vramcard",
]