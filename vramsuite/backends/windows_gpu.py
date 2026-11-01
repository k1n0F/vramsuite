"""
Windows-only native readers for per-process GPU memory.

NVML cannot do this on Windows: under WDDM it always reports a process's
usedGpuMemory as NOT_AVAILABLE, because Windows (not the NVIDIA driver)
manages video memory. Task Manager reads the same numbers used here:

- PDH counters "GPU Process Memory(*)" / "GPU Adapter Memory(*)" (pdh.dll)
- adapter LUID -> PCI address through D3DKMT (gdi32.dll), so the counters
  can be matched to the exact NVIDIA GPU (iGPU / Basic Render are skipped)
- process image path and command line (kernel32 / ntdll)
"""

from __future__ import annotations

import ctypes
import re
from ctypes import wintypes
from functools import lru_cache

PDH_FMT_LARGE = 0x400
PDH_MORE_DATA = 0x800007D2
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_COMMAND_LINE_INFORMATION = 60
KMTQAITYPE_ADAPTERADDRESS = 6

# "pid_44968_luid_0x00000000_0x00016A7B_phys_0" / "luid_0x00000000_0x00016A7B_phys_0"
_INSTANCE = re.compile(
    r"(?:pid_(?P<pid>\d+)_)?luid_0x(?P<high>[0-9A-Fa-f]+)_0x(?P<low>[0-9A-Fa-f]+)_phys_\d+"
)


class _FmtValue(ctypes.Structure):
    class _Value(ctypes.Union):
        _fields_ = [("largeValue", ctypes.c_longlong), ("doubleValue", ctypes.c_double)]

    _fields_ = [("CStatus", wintypes.DWORD), ("value", _Value)]


class _CounterItem(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _FmtValue)]


class _Luid(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _OpenAdapterFromLuid(ctypes.Structure):
    _fields_ = [("AdapterLuid", _Luid), ("hAdapter", ctypes.c_uint)]


class _QueryAdapterInfo(ctypes.Structure):
    _fields_ = [
        ("hAdapter", ctypes.c_uint),
        ("Type", ctypes.c_int),
        ("pPrivateDriverData", ctypes.c_void_p),
        ("PrivateDriverDataSize", ctypes.c_uint),
    ]


class _AdapterAddress(ctypes.Structure):
    _fields_ = [
        ("BusNumber", ctypes.c_uint),
        ("DeviceNumber", ctypes.c_uint),
        ("FunctionNumber", ctypes.c_uint),
    ]


class _UnicodeString(ctypes.Structure):
    _fields_ = [
        ("Length", ctypes.c_ushort),
        ("MaximumLength", ctypes.c_ushort),
        ("Buffer", ctypes.c_void_p),
    ]


def read_pdh_counter(path: str) -> dict[str, int]:
    """All instances of a wildcard PDH counter as {instance_name: value}."""
    pdh = ctypes.WinDLL("pdh")
    query = ctypes.c_void_p()
    counter = ctypes.c_void_p()

    status = pdh.PdhOpenQueryW(None, None, ctypes.byref(query))
    if status != 0:
        raise OSError(f"PdhOpenQueryW failed: 0x{status & 0xFFFFFFFF:08x}")

    try:
        status = pdh.PdhAddEnglishCounterW(query, path, None, ctypes.byref(counter))
        if status != 0:
            raise OSError(f"PdhAddEnglishCounterW({path}) failed: 0x{status & 0xFFFFFFFF:08x}")

        status = pdh.PdhCollectQueryData(query)
        if status != 0:
            raise OSError(f"PdhCollectQueryData failed: 0x{status & 0xFFFFFFFF:08x}")

        size = wintypes.DWORD(0)
        count = wintypes.DWORD(0)
        status = pdh.PdhGetFormattedCounterArrayW(
            counter, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None
        )
        if status & 0xFFFFFFFF != PDH_MORE_DATA:
            return {}  # no instances (e.g. no process uses a GPU right now)

        buffer = ctypes.create_string_buffer(size.value)
        status = pdh.PdhGetFormattedCounterArrayW(
            counter, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buffer
        )
        if status != 0:
            raise OSError(f"PdhGetFormattedCounterArrayW failed: 0x{status & 0xFFFFFFFF:08x}")

        items = ctypes.cast(buffer, ctypes.POINTER(_CounterItem))
        return {items[i].szName: int(items[i].FmtValue.value.largeValue) for i in range(count.value)}

    finally:
        pdh.PdhCloseQuery(query)


@lru_cache(maxsize=None)
def luid_pci_address(high: int, low: int) -> tuple[int, int, int] | None:
    """(bus, device, function) of a display adapter, None for software adapters."""
    gdi32 = ctypes.WinDLL("gdi32")
    open_args = _OpenAdapterFromLuid(_Luid(low, high), 0)

    if gdi32.D3DKMTOpenAdapterFromLuid(ctypes.byref(open_args)) != 0:
        return None

    address = _AdapterAddress()
    query = _QueryAdapterInfo(
        open_args.hAdapter,
        KMTQAITYPE_ADAPTERADDRESS,
        ctypes.cast(ctypes.byref(address), ctypes.c_void_p),
        ctypes.sizeof(address),
    )
    status = gdi32.D3DKMTQueryAdapterInfo(ctypes.byref(query))
    gdi32.D3DKMTCloseAdapter(ctypes.byref(ctypes.c_uint(open_args.hAdapter)))

    # Software adapters (Microsoft Basic Render) report 0xFFFFFFFF.
    if status != 0 or address.BusNumber == 0xFFFFFFFF:
        return None
    return address.BusNumber, address.DeviceNumber, address.FunctionNumber


def parse_instance(name: str) -> tuple[int | None, int, int] | None:
    """Counter instance name -> (pid or None, luid_high, luid_low)."""
    match = _INSTANCE.fullmatch(name)
    if not match:
        return None
    pid = int(match["pid"]) if match["pid"] else None
    return pid, int(match["high"], 16), int(match["low"], 16)


def _open_process(pid: int) -> int | None:
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    return handle or None


def process_image(pid: int) -> str | None:
    """Full exe path, None for protected processes (dwm, csrss) or exited ones."""
    handle = _open_process(pid)
    if handle is None:
        return None

    kernel32 = ctypes.WinDLL("kernel32")
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buffer))
        ok = kernel32.QueryFullProcessImageNameW(
            wintypes.HANDLE(handle), 0, buffer, ctypes.byref(size)
        )
        return buffer.value if ok else None
    finally:
        kernel32.CloseHandle(wintypes.HANDLE(handle))


def process_command_line(pid: int) -> str | None:
    handle = _open_process(pid)
    if handle is None:
        return None

    kernel32 = ctypes.WinDLL("kernel32")
    ntdll = ctypes.WinDLL("ntdll")
    try:
        size = ctypes.c_ulong(0)
        ntdll.NtQueryInformationProcess(
            wintypes.HANDLE(handle), PROCESS_COMMAND_LINE_INFORMATION, None, 0, ctypes.byref(size)
        )
        if size.value == 0:
            return None

        buffer = ctypes.create_string_buffer(size.value)
        status = ntdll.NtQueryInformationProcess(
            wintypes.HANDLE(handle),
            PROCESS_COMMAND_LINE_INFORMATION,
            buffer,
            size,
            ctypes.byref(size),
        )
        if status != 0:
            return None

        text = ctypes.cast(buffer, ctypes.POINTER(_UnicodeString)).contents
        return ctypes.wstring_at(text.Buffer, text.Length // 2) if text.Buffer else None
    finally:
        kernel32.CloseHandle(wintypes.HANDLE(handle))
