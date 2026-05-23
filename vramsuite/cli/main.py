"""
VRAM Suite CLI entrypoint.

Status:
    pre-alpha / project skeleton
"""

from __future__ import annotations
import json
from pathlib import Path

import typer
from rich.console import Console

from vramsuite.core.fingerprint import collect_fingerprint
from vramsuite.core.vramcard import create_vramcard, save_vramcard


from vramsuite.core.reports import (
    print_cuda_devices_table,
    print_doctor_header,
    print_nvml_devices_table,
    print_nvml_table,
    print_runtime_table,
    print_torch_table,
    print_verbose_table,
    print_vramcard_memory_table,
)

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
    write_card: bool = typer.Option(
        True,
        "--write-card/--no-write-card",
        help="Write a .vramcard JSON file.",
    ),
    output: Path = typer.Option(
        Path("system.vramcard.json"),
        "--output",
        "-o",
        help="Output path for .vramcard JSON file.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the generated .vramcard JSON to stdout.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed diagnostic information.",
    ),
) -> None:
    """Show basic VRAM Suite diagnostic information."""
    fingerprint = collect_fingerprint()
    runtime = fingerprint["runtime"]
    torch_info = fingerprint["torch"]
    nvml_info = fingerprint["nvml"]

    vramcard = create_vramcard(fingerprint=fingerprint)
    memory_info = vramcard["memory"]

    print_doctor_header(console)
    print_runtime_table(console, runtime)
    print_torch_table(console, torch_info)
    print_nvml_table(console, nvml_info)
    print_nvml_devices_table(console, nvml_info)
    print_vramcard_memory_table(console, memory_info)
    print_cuda_devices_table(console, torch_info)

    if verbose:
        print_verbose_table(console, runtime, vramcard)

    if write_card:
        path = save_vramcard(output, card=vramcard)
        console.print(f"[green]Saved:[/green] {path}")

    if json_output:
        console.print_json(
            json.dumps(
                vramcard,
                indent=2,
                ensure_ascii=False,
            )
        )

    console.print("[yellow]Next:[/yellow] v0.2-alpha safe allocation probe")

if __name__ == "__main__":
    app()