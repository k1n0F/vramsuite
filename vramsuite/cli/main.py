"""
VRAM Suite CLI entrypoint.

Status:
    pre-alpha / project skeleton
"""

from __future__ import annotations

import typer

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vramsuite.core.fingerprint import collect_fingerprint
from vramsuite.core.vramcard import save_vramcard

app = typer.Typer(
    help="VRAM Suite CLI",
    no_args_is_help=True,
    )

console = Console()


@app.callback()
def main() -> None:
    """Predictive GPU memory framework for AI inference workflows."""
    pass


@app.command()
def doctor(
    write_card: bool = typer.Option (
        True,
        "--write-card/--no-write-card",
        help="Write system.vramcard.json to the current directory.",
     ),
) -> None:
    """Show basic VRAM Suite diagnostic information."""
    fingerprint = collect_fingerprint()
    runtime = fingerprint["runtime"]
    torch_info = fingerprint["torch"]
    nvml_info = fingerprint["nvml"]

    console.print(
        Panel.fit(
            "[bold cyan]VRAM Suite Doctor[/bold cyan]\n"
            "Status: pre-alpha / v0.1-alpha foundation",
            border_style="cyan",
            )
        )

    table = Table(title="Runtime")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("OS", str(runtime["os_name"]))
    table.add_row("Platform", str(runtime["platform"]))
    table.add_row("Python", str(runtime["python_version"]))
    table.add_row("WSL", str(runtime["is_wsl"]))
    table.add_row("Container", str(runtime["is_container"]))

    console.print(table)

    torch_table = Table(title="PyTorch / CUDA")
    torch_table.add_column("Field", style="bold")
    torch_table.add_column("Value")

    torch_table.add_row("Torch available", str(torch_info["available"]))
    torch_table.add_row("Torch version", str(torch_info["version"]))
    torch_table.add_row("CUDA available", str(torch_info["cuda_available"]))
    torch_table.add_row("Torch CUDA", str(torch_info["cuda_version"]))
    torch_table.add_row("CUDA devices", str(torch_info["device_count"]))

    console.print(torch_table)

    nvml_table = Table(title="NVML / Driver Memory")
    nvml_table.add_column("Field")
    nvml_table.add_column("Value")

    nvml_table.add_row("NVML available", str(nvml_info.get("available")))
    nvml_table.add_row("NVML error", str(nvml_info.get("error")))
    nvml_table.add_row("NVML devices", str(nvml_info.get("device_count")))

    console.print(nvml_table)

    if nvml_info.get("devices"):
        gpu_table = Table(title="NVML Device")
        gpu_table.add_column("Index")
        gpu_table.add_column("Name")
        gpu_table.add_column("Total VRAM MB")
        gpu_table.add_column("Free VRAM MB")
        gpu_table.add_column("Used VRAM MB")

        for device in nvml_info["devices"]:
            gpu_table.add_row(
                str(device.get("index")),
                str(device.get("name")),
                str(device.get("total_vram_mb")),
                str(device.get("free_vram_mb")),
                str(device.get("used_vram_mb")),
            )

        console.print(gpu_table)


    if torch_info.get["devices"]:
        gpu_table = Table(title="CUDA Devices")
        gpu_table.add_column("Index", style="bold")
        gpu_table.add_column("Name")
        gpu_table.add_column("VRAM MB")
        gpu_table.add_column("SM")

        for device in torch_info.get("devices", []):
            gpu_table.add_row(
                str(device["index"]),
                str(device["name"]),
                str(device["total_vram_mb"]),
                str(device["compute_capability"])
             )

        console.print(gpu_table)

    if write_card:
        path = save_vramcard("system.vramcard.jsson")
        console.print(f"[green]Saved:[/green] {path}")

    console.print("[yellow]Next:[/yellow] v0.2-alpha safe allocation probe")

if __name__ == "__main__":
    app()