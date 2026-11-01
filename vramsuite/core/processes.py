"""
Per-process VRAM usage of one GPU, observed from outside the process.

Works for any framework (ComfyUI, games, browsers) without code inside them.
Windows: PDH "GPU Process Memory" counters matched to the GPU by PCI address
(NVML cannot report per-process memory under WDDM). Linux: NVML.
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

from vramsuite.backends.nvml import collect_nvml_gpu_processes

BYTES_PER_MB = 1024 * 1024


def aggregate_counters(
        dedicated: dict[str, int],
        shared: dict[str, int],
        parse_instance: Callable[[str], tuple[int | None, int, int] | None],
        luid_pci: Callable[[int, int], tuple[int, int, int] | None],
        target_bus_device: tuple[int, int],
) -> dict[int, dict[str, int]]:
    """
    Sum PDH per-process instances that belong to the target GPU.

    One process can have several instances (one per adapter / physical engine);
    instances on other adapters (iGPU, Basic Render) are dropped.
    """
    per_pid: dict[int, dict[str, int]] = {}

    for kind, counters in (("dedicated", dedicated), ("shared", shared)):
        for name, value in counters.items():
            parsed = parse_instance(name)
            if parsed is None or parsed[0] is None:
                continue

            pid, high, low = parsed
            pci = luid_pci(high, low)
            if pci is None or (pci[0], pci[1]) != target_bus_device:
                continue

            entry = per_pid.setdefault(pid, {"dedicated": 0, "shared": 0})
            entry[kind] += value

    return per_pid


def _describe_process(pid: int) -> tuple[str | None, str | None]:
    """(exe path, command line); None where the OS denies access."""
    if platform.system() == "Windows":
        from vramsuite.backends.windows_gpu import process_command_line, process_image

        return process_image(pid), process_command_line(pid)

    proc = Path("/proc") / str(pid)
    try:
        exe = os.readlink(proc / "exe")
    except OSError:
        exe = None
    try:
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip() or None
    except OSError:
        command = None
    return exe, command


def _sample(device_index: int) -> dict[str, Any]:
    gpu = collect_nvml_gpu_processes(device_index)

    if platform.system() == "Windows":
        from vramsuite.backends import windows_gpu

        per_pid = aggregate_counters(
            dedicated=windows_gpu.read_pdh_counter("\\GPU Process Memory(*)\\Dedicated Usage"),
            shared=windows_gpu.read_pdh_counter("\\GPU Process Memory(*)\\Shared Usage"),
            parse_instance=windows_gpu.parse_instance,
            luid_pci=windows_gpu.luid_pci_address,
            target_bus_device=(gpu["pci"]["bus"], gpu["pci"]["device"]),
        )
        source = "windows-pdh"
    else:
        per_pid = {
            pid: {"dedicated": used or 0, "shared": 0}
            for pid, used in gpu["processes"].items()
        }
        source = "nvml"

    return {"gpu": gpu, "source": source, "per_pid": per_pid}


def _report(
        sample: dict[str, Any],
        peaks: dict[int, int],
        samples: int,
        min_mb: float,
) -> dict[str, Any]:
    gpu = sample["gpu"]
    rows = []

    for pid, usage in sample["per_pid"].items():
        peak = max(peaks.get(pid, 0), usage["dedicated"])
        if peak / BYTES_PER_MB < min_mb:
            continue

        exe, command = _describe_process(pid)
        rows.append({
            "pid": pid,
            "name": Path(exe).name if exe else None,
            "exe": exe,
            "command_line": command,
            "dedicated_mb": round(usage["dedicated"] / BYTES_PER_MB, 1),
            "peak_dedicated_mb": round(peak / BYTES_PER_MB, 1),
            "shared_mb": round(usage["shared"] / BYTES_PER_MB, 1),
        })

    rows.sort(key=lambda row: row["peak_dedicated_mb"], reverse=True)
    processes_total = sum(usage["dedicated"] for usage in sample["per_pid"].values())

    return {
        "gpu": {
            "index": gpu["index"],
            "name": gpu["name"],
            "total_mb": round(gpu["total_bytes"] / BYTES_PER_MB, 1),
            "nvml_used_mb": round(gpu["used_bytes"] / BYTES_PER_MB, 1),
        },
        "source": sample["source"],
        "samples": samples,
        # Not physical usage: WDDM counts shared surfaces (e.g. with dwm) in several
        # processes and includes allocations paged out to RAM, so the sum can
        # exceed both nvml_used_mb and the card size.
        "processes_sum_mb": round(processes_total / BYTES_PER_MB, 1),
        "processes": rows,
    }


def snapshot_gpu_processes(device_index: int = 0, min_mb: float = 1.0) -> dict[str, Any]:
    sample = _sample(device_index)
    return _report(sample, {}, samples=1, min_mb=min_mb)


def watch_gpu_processes(
        seconds: float,
        interval: float = 0.5,
        device_index: int = 0,
        min_mb: float = 1.0,
        on_sample: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """
    Sample repeatedly and keep each process's peak dedicated memory.

    Ctrl+C ends the watch early and still returns the peaks seen so far.
    """
    peaks: dict[int, int] = {}
    last: dict[str, Any] | None = None
    samples = 0
    deadline = time.monotonic() + seconds

    try:
        while True:
            last = _sample(device_index)
            samples += 1
            for pid, usage in last["per_pid"].items():
                peaks[pid] = max(peaks.get(pid, 0), usage["dedicated"])

            if on_sample is not None:
                on_sample(samples)
            if time.monotonic() >= deadline:
                break
            time.sleep(interval)

    except KeyboardInterrupt:
        pass

    if last is None:
        last = _sample(device_index)
        samples = 1

    # Processes that exited during the watch still deserve their peak row.
    for pid, peak in peaks.items():
        last["per_pid"].setdefault(pid, {"dedicated": 0, "shared": 0})

    return _report(last, peaks, samples=samples, min_mb=min_mb)
