"""
`python -m vramsuite` and the entry point of the Windows exe.

Double-clicking the exe opens a console that closes as soon as the program
exits. In that case the exe runs `doctor` and waits for Enter instead of
printing the help and vanishing.
"""

from __future__ import annotations

import platform
import sys

from vramsuite.cli.main import app


def _started_by_double_click() -> bool:
    """True when this process is the only one attached to its console (Explorer created it)."""
    if platform.system() != "Windows":
        return False
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.GetConsoleProcessList.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
    kernel32.GetConsoleProcessList.restype = ctypes.c_uint32
    processes = (ctypes.c_uint32 * 4)()
    return kernel32.GetConsoleProcessList(processes, 4) == 1


def main() -> None:
    if len(sys.argv) == 1 and _started_by_double_click():
        sys.argv += ["doctor", "--no-write-card"]
        try:
            app()
        except SystemExit:
            pass
        input("\nPress Enter to close...")
        return
    app()


if __name__ == "__main__":
    main()
