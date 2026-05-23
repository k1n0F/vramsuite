from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel



def print_doctor_header(console: Console) -> None:
    """Print the VRAM Suite doctor header."""
    console.print(
        Panel(
            "[bold cyan]VRAM Suite Doctor[/bold cyan]\n"
            "Status: pre-alpha / v0.1-alpha foundation",
            border_style="cyan",
        )
    )


def print_runtime_table(console: Console, runtime_info: dict[str, Any]) -> None:
    """Print runtime evnvironment information."""
    table= Table(title="Runtime")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("OS", str(runtime_info.get("os_name")))
    table.add_row("Platform", str(runtime_info.get("platform")))
    table.add_row("Python", str(runtime_info.get("python_version")))
    table.add_row("WSL", str(runtime_info.get("is_wsl")))
    table.add_row("Container", str(runtime_info.get("is_container")))

    console.print(table)


def print_torch_table(console: Console, torch_info: dict[str, Any]) -> None:
    """Print optional PyTorch/CUDA information."""
    table = Table(title="PyTorch / CUDA")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Torch available", str(torch_info.get("available")))
    table.add_row("Torch version", str(torch_info.get("version")))
    table.add_row("CUDA available", str(torch_info.get("cuda_available")))
    table.add_row("Torch CUDA", str(torch_info.get("cuda_version")))
    table.add_row("CUDA devices", str(torch_info.get("device_count")))

    console.print(table)


def print_nvml_table(console: Console, nvml_info: dict[str, Any]) -> None:
    """Print NVML driver availability information."""
    table = Table(title="NVML / Driver Memory")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("NVML available", str(nvml_info.get("available")))
    table.add_row("NVML error", str(nvml_info.get("error")))
    table.add_row("NVML devices", str(nvml_info.get("device_count")))

    console.print(table)


def print_nvml_devices_table(console: Console, nvml_info: dict[str, Any]) -> None:
    """Print NVML GPU memory information"""
    devices = nvml_info.get("devices") or []

    if not devices:
        return
    
    table = Table(title="NVML Devices")
    table.add_column("Index", style="bold")
    table.add_column("Name")
    table.add_column("Total VRAM MB")
    table.add_column("Free VRAM MB")
    table.add_column("Used VRAM MB")

    for device in devices:
        table.add_row(
            str(device.get("index")),
            str(device.get("name")),
            str(device.get("total_vram_mb")),
            str(device.get("free_vram_mb")),
            str(device.get("used_vram_mb")),
        )

    console.print(table)


def print_nvml_device_info(console: Console, nvml_info: dict[str, Any]) -> None:
    """Print NVML GPU memory information"""
    devices = nvml_info.get("devices") or []

    if not devices:
        return
    
    table = Table(title="NVML Device")
    table.add_column("Index", style="bold")
    table.add_column("Name")
    table.add_column("Total VRAM MB")
    table.add_column("Free VRAM MB")
    table.add_column("Used VRAM MB")

    for device in devices:
        table.add_row(
            str(device.get("index")),
            str(device.get("name")),
            str(device.get("total_vram_mb")),
            str(device.get("free_vram_mb")),
            str(device.get("used_vram_mb")),
        )

    console.print(table)

def print_vramcard_memory_table(console: Console, memory_info: dict[str, Any]) -> None:
    """Print memory information stored in the generated .vramcard."""
    table = Table(title="VRAMCard Memory")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Driver total MB", str(memory_info.get("driver_total_mb")))
    table.add_row("Driver free at scan MB", str(memory_info.get("driver_free_at_scan_mb")))
    table.add_row("Driver used at scan MB", str(memory_info.get("driver_used_at_scan_mb")))
    table.add_row("Process allocatable MB", str(memory_info.get("process_allocatable_mb")))
    table.add_row("Safe allocatable MB", str(memory_info.get("safe_allocatable_mb")))
    table.add_row("Safety margin MB", str(memory_info.get("safety_margin_mb")))
    table.add_row("Source", str(memory_info.get("source")))

    console.print(table)


def print_cuda_devices_table(console: Console, torch_info: dict[str, Any]) -> None:
    """Print CUDA devices detected by PyTorch."""
    devices = torch_info.get("devices") or []

    if not devices:
        return
    
    table = Table(title="CUDA Devices")
    table.add_column("Index", style="bold")
    table.add_column("Name")
    table.add_column("VRAM MB")
    table.add_column("SM")

    for device in devices:
        table.add_row(
            str(device.get("index")),
            str(device.get("name")),
            str(device.get("total_vram_mb")),
            str(device.get("compute_capability")),
        )

    console.print(table)

def print_verbose_table(
    console: Console,
    runtime_info: dict[str, Any],
    vramcard_data: dict[str, Any],
) -> None:
    """Print additional diagnostic details."""
    memory_info = vramcard_data.get("memory" or {})
    gpu_info = vramcard_data.get("gpu" or {})
    
    table = Table(title="Verbose Diagnostics")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Schema", str(vramcard_data.get("schema")))
    table.add_row("Generated at", str(vramcard_data.get("generated_at")))
    table.add_row("GPU Source", str(gpu_info.get("source")))
    table.add_row("Memory Source", str(memory_info.get("source")))
    table.add_row("Python executable", str(runtime_info.get("python_executable")))

    console.print(table)

def print_probe_table(console: Console, probe_info: dict[str, Any] | None) -> None:
    """Print safe allocation probe results."""
    if probe_info is None:
        return
    
    table = Table(title="Safe Probe")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Available", str(probe_info.get("available")))
    table.add_row("Backend", str(probe_info.get("backend")))
    table.add_row("Attempted MB", str(probe_info.get("attempted_mb")))
    table.add_row("Allocated MB", str(probe_info.get("allocated_mb")))
    table.add_row("Safe Allocatable MB", str(probe_info.get("safe_allocatable_mb")))
    table.add_row("Safety Margin MB", str(probe_info.get("safety_margin_mb")))
    table.add_row("Error", str(probe_info.get("error")))

    notes = probe_info.get("notes") or []
    if notes:
        table.add_row("Notes", " | ".join(str(note) for note in notes))

    console.print(table)
