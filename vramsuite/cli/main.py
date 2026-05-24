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

from vramsuite.core.doctor import run_doctor
from vramsuite.core.vramcard import save_vramcard


from vramsuite.core.reports import (
    print_cuda_devices_table,
    print_doctor_header,
    print_nvml_devices_table,
    print_nvml_table,
    print_runtime_table,
    print_torch_table,
    print_verbose_table,
    print_vramcard_memory_table,
    print_probe_table,
    print_risk_table,
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
    probe: bool = typer.Option(
        False,
        "--probe",
        help="Run a bounded CUDA allocation probe.",
    ),
    probe_max_mb: int = typer.Option(
        1024,
        "--probe-max-mb",
        help="Maximum amount of VRAM to probe.",
    ),
    probe_step_mb: int = typer.Option(
        128,
        "--probe-step-mb",
        help="Allocation step size for the probe.",
    ),
    probe_free_floor_mb: int = typer.Option(
        2048,
        "--probe-free-floor-mb",
        help="Minimum free driver VRAM to leave untouched.",

    ),
    estimate_mb: int | None = typer.Option(
        None,
        "--estimate-mb",
        help="Estimate OOM risk for a required VRAM amount in MB.",
    ),
    probe_max_free_ratio: float = typer.Option(
        0.9,
        "--probe-max-free-ratio",
        help="Maximum fraction of currently free VRAM that the probe may attempt to allocate.",
    ),
) -> None:
    """Show basic VRAM Suite diagnostic information."""
    doctor_results = run_doctor(
        with_probe=probe,
        probe_max_mb=probe_max_mb,
        probe_step_mb=probe_step_mb,
        probe_floor_mb=probe_free_floor_mb,
        estimate_mb=estimate_mb,
        probe_max_free_ratio=probe_max_free_ratio,
    )

    runtime = doctor_results["runtime"]
    torch_info = doctor_results["torch"]
    nvml_info = doctor_results["nvml"]
    vramcard = doctor_results["vramcard"]
    memory_info = doctor_results["memory"]
    probe_info = doctor_results.get("probe")
    risk_info = doctor_results.get("risk_estimate")
    
    
    print_doctor_header(console)
    print_runtime_table(console, runtime)
    print_torch_table(console, torch_info)
    print_nvml_table(console, nvml_info)
    print_nvml_devices_table(console, nvml_info)
    print_vramcard_memory_table(console, memory_info)
    print_probe_table(console, probe_info)
    print_cuda_devices_table(console, torch_info)
    print_risk_table(console, risk_info)
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

    console.print("[yellow]Next:[/yellow] workflow profiling and OOM risk estimation")

if __name__ == "__main__":
    app()