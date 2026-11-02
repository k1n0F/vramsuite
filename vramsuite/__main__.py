"""
`python -m vramsuite` and the entry point of the Windows exe.

Double-clicking the exe opens a console that closes as soon as the program
exits. In that case the exe runs `doctor` and waits for Enter instead of
printing the help and vanishing.
"""

from __future__ import annotations

import os
import platform
import sys
import traceback

from vramsuite.cli.main import app


def _started_by_double_click() -> bool:
    """
    True when no other program shares this console: Explorer created it for us.

    From a terminal, the shell is attached too. The one-file exe runs as two
    processes (the PyInstaller bootloader and its Python child, this one),
    both attached to the console, so the bootloader parent does not count.
    """
    if platform.system() != "Windows":
        return False
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.GetConsoleProcessList.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
    kernel32.GetConsoleProcessList.restype = ctypes.c_uint32
    processes = (ctypes.c_uint32 * 16)()
    count = kernel32.GetConsoleProcessList(processes, 16)

    ours = {os.getpid()}
    if getattr(sys, "frozen", False):
        ours.add(os.getppid())
    return 0 < count <= 16 and not set(processes[:count]) - ours


def main() -> None:
    if len(sys.argv) == 1 and _started_by_double_click():
        sys.argv += ["doctor", "--no-write-card"]
        try:
            app()
        except SystemExit:
            pass
        except Exception:
            # Show the error instead of a window that closes before it can be read.
            traceback.print_exc()
        input("\nPress Enter to close...")
        return
    app()


if __name__ == "__main__":
    main()
