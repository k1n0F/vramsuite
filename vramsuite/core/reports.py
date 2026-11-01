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


def print_compute_check_table(console: Console, compute_info: dict[str, Any] | None) -> None:
    """Print native compute stack check results."""
    if compute_info is None:
        return

    table = Table(title="Compute Check")
    table.add_column("Check", style="bold")
    table.add_column("Passed")
    table.add_column("Max abs error")
    table.add_column("Error")

    for name, result in compute_info.items():
        passed = result.get("passed")
        table.add_row(
            name,
            "[green]True[/green]" if passed else "[red]False[/red]",
            str(result.get("max_abs_error")),
            str(result.get("error") or ""),
        )

    console.print(table)


def print_model_table(console: Console, model_info: dict[str, Any]) -> None:
    """Print weight sizes of an inspected model file."""
    summary = Table(title="Model")
    summary.add_column("Field", style="bold")
    summary.add_column("Value")

    files = model_info.get("files") or []
    summary.add_row("Path", str(model_info.get("path")))
    summary.add_row("Files", str(len(files)))
    summary.add_row("Tensors", str(model_info.get("tensor_count")))
    summary.add_row("Parameters", f"{model_info.get('total_params', 0) / 1e9:.3f} B")
    summary.add_row("Weights MB", str(model_info.get("total_mb")))
    console.print(summary)

    dtypes = Table(title="Weights by dtype")
    dtypes.add_column("Dtype", style="bold")
    dtypes.add_column("Tensors")
    dtypes.add_column("Params")
    dtypes.add_column("MB")
    for name, item in (model_info.get("by_dtype") or {}).items():
        dtypes.add_row(name, str(item["tensors"]), str(item["params"]), str(item["mb"]))
    console.print(dtypes)

    components = Table(title="Weights by component (first name segment)")
    components.add_column("Component", style="bold")
    components.add_column("MB")
    for name, size_mb in (model_info.get("components_mb") or {}).items():
        components.add_row(name, str(size_mb))
    other = model_info.get("other_components") or 0
    if other:
        components.add_row(f"... {other} more", "")
    console.print(components)


def print_processes_table(console: Console, info: dict[str, Any], watched: bool) -> None:
    """Print per-process VRAM usage of one GPU."""
    gpu = info["gpu"]
    title = f"GPU {gpu['index']} {gpu['name']}: processes ({info['source']}, {info['samples']} samples)"

    table = Table(title=title)
    table.add_column("PID", style="bold")
    table.add_column("Process")
    table.add_column("Dedicated MB", justify="right")
    if watched:
        table.add_column("Peak MB", justify="right")
    table.add_column("Shared MB", justify="right")
    table.add_column("Command", overflow="fold")

    for row in info["processes"]:
        cells = [
            str(row["pid"]),
            row["name"] or "[dim]<protected>[/dim]",
            str(row["dedicated_mb"]),
        ]
        if watched:
            cells.append(str(row["peak_dedicated_mb"]))
        cells += [str(row["shared_mb"]), (row["command_line"] or "")[:120]]
        table.add_row(*cells)

    console.print(table)
    console.print(
        f"NVML used: {gpu['nvml_used_mb']} MB of {gpu['total_mb']} MB. "
        f"Sum over processes: {info['processes_sum_mb']} MB "
        "[dim](can exceed physical usage on Windows: shared surfaces are counted "
        "in several processes, paged-out allocations are included)[/dim]"
    )


def print_risk_table(console: Console, risk_info: dict[str, Any] | None) -> None:
    """Print OOM risk estimate results."""
    if risk_info is None:
        return
    
    table = Table(title="OOM Risk Estimate")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Available", str(risk_info.get("available")))
    table.add_row("Required MB", str(risk_info.get("required_mb")))
    table.add_row("Available MB", str(risk_info.get("available_mb")))
    table.add_row("Availability Source", str(risk_info.get("availability_source")))
    table.add_row("Remaining MB", str(risk_info.get("remaining_mb")))
    
    usage_ratio = risk_info.get("usage_ratio")
    if usage_ratio is not None:
        usage_ratio_text = f"{usage_ratio:.2%}"
    else:
        usage_ratio_text = "None"

    table.add_row("Usage Ratio", usage_ratio_text)
    table.add_row("Risk Level", str(risk_info.get("risk_level")))
    table.add_row("Reason", str(risk_info.get("reason")))

    console.print(table)


def print_vramcard_compare_table(console, comparison: dict) -> None:
    """Print .vramcard comparison table."""

    table = Table(title="VRAMCard Compare")

    table.add_column("Section")
    table.add_column("Field")
    table.add_column("Left")
    table.add_column("Right")
    table.add_column("Delta")
    table.add_column("Changed")

    for row in comparison.get("rows", []):
        changed = row.get("changed")

        if not changed:
            continue

        delta = row.get("delta")

        if isinstance(delta, (int, float)):
            if delta > 0:
                delta_text = f"+{delta}"
            else:
                delta_text = str(delta)
        else:
            delta_text = "-"

        table.add_row(
            str(row.get("section")),
            str(row.get("field")),
            str(row.get("left")),
            str(row.get("right")),
            delta_text,
            "yes" if changed else "no"
        )

    console.print(table)

    console.print(
        f"[green]Changed rows:[/green] "
        f"{comparison.get('changed_rows')} / {comparison.get('total_rows')}"
    )