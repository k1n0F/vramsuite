"""
VRAM Suite CLI entrypoint.

Status:
    alpha
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from vramsuite.core.doctor import run_doctor
from vramsuite.core.vramcard import load_vramcard, save_vramcard
from vramsuite.core.compare import compare_vramcards
from vramsuite.core.model_inspect import ModelInspectError, inspect_model
from vramsuite.core.processes import snapshot_gpu_processes, watch_gpu_processes
from vramsuite.core.transfer import run_transfer_benchmark
from vramsuite.core.estimate import DEFAULT_RESERVE_MB, estimate_workflow
from vramsuite.core.workflow import estimate_comfy_workflow
from vramsuite.core.validate import validate_vramcard



from vramsuite.core.reports import (
    print_doctor_header,
    print_nvml_devices_table,
    print_nvml_table,
    print_runtime_table,
    print_torch_table,
    print_verbose_table,
    print_vramcard_memory_table,
    print_probe_table,
    print_risk_table,
    print_compute_check_table,
    print_model_table,
    print_processes_table,
    print_transfer_table,
    print_estimate_table,
    print_validation,
    print_vramcard_compare_table,
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
    compute_check: bool = typer.Option(
        False,
        "--compute-check",
        help="Run native cuBLAS / cuBLASLt (FP32, FP8) correctness checks.",
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
        with_compute_check=compute_check,
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
    print_risk_table(console, risk_info)
    print_compute_check_table(console, doctor_results.get("compute_check"))
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

@app.command()
def compare(
    left: Path = typer.Argument(
        ...,
        help="Path to the first .vramcard JSON file.",
    ),
    right: Path = typer.Argument(
        ...,
        help="Path to the second .vramcard JSON file."
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print comparison result as JSON.",
    ),
) -> None:
    """Compare two .vramcard JSON files."""

    left_card = load_vramcard(left)
    right_card = load_vramcard(right)

    comparison = compare_vramcards(left_card, right_card)

    if json_output:
        console.print_json(
            json.dumps(
                comparison,
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    
    print_vramcard_compare_table(console, comparison)

@app.command()
def validate(
    path: Path = typer.Argument(
        ...,
        help="Path to a .vramcard JSON file.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the validation result as JSON.",
    ),
) -> None:
    """Check a .vramcard for structure errors and numbers that contradict each other."""
    try:
        card = load_vramcard(path)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Cannot read card:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    info = validate_vramcard(card)
    if json_output:
        console.print_json(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        print_validation(console, str(path), info)

    if not info["valid"]:
        raise typer.Exit(code=1)


@app.command()
def inspect(
    path: Path = typer.Argument(
        ...,
        help="A .safetensors file or a directory of .safetensors shards.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the inspection result as JSON.",
    ),
) -> None:
    """Show exact weight sizes of a model file without loading it."""
    try:
        model_info = inspect_model(path)
    except (ModelInspectError, OSError) as exc:
        console.print(f"[red]Cannot inspect model:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        console.print_json(json.dumps(model_info, indent=2, ensure_ascii=False))
        return

    print_model_table(console, model_info)


@app.command()
def processes(
    watch: float = typer.Option(
        0,
        "--watch",
        "-w",
        help="Sample for this many seconds and report each process's peak (Ctrl+C stops early).",
    ),
    interval: float = typer.Option(
        0.5,
        "--interval",
        help="Seconds between samples in --watch mode.",
    ),
    gpu: int = typer.Option(
        0,
        "--gpu",
        help="NVML device index.",
    ),
    min_mb: float = typer.Option(
        1.0,
        "--min-mb",
        help="Hide processes whose (peak) dedicated memory is below this.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the result as JSON.",
    ),
) -> None:
    """Show VRAM used by each process on a GPU (observed from outside)."""
    if watch > 0:
        with console.status(f"Watching GPU {gpu} for {watch:g}s (Ctrl+C to stop early)..."):
            info = watch_gpu_processes(
                seconds=watch, interval=interval, device_index=gpu, min_mb=min_mb
            )
    else:
        info = snapshot_gpu_processes(device_index=gpu, min_mb=min_mb)

    if json_output:
        console.print_json(json.dumps(info, indent=2, ensure_ascii=False))
        return

    print_processes_table(console, info, watched=watch > 0)


@app.command()
def transfer(
    size_mb: int = typer.Option(
        256,
        "--size-mb",
        help="Bytes per copy. Needs this much free VRAM (x2) and pinned RAM.",
    ),
    repeats: int = typer.Option(
        5,
        "--repeats",
        help="Timed copies per direction (after one warm-up copy).",
    ),
    gpu: int = typer.Option(
        0,
        "--gpu",
        help="CUDA device index.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the result as JSON.",
    ),
) -> None:
    """Measure host <-> device copy bandwidth over the real PCIe link."""
    with console.status(f"Copying {size_mb} MB blocks..."):
        info = run_transfer_benchmark(device_index=gpu, size_mb=size_mb, repeats=repeats)

    if json_output:
        console.print_json(json.dumps(info, indent=2, ensure_ascii=False))
        return

    print_transfer_table(console, info)


@app.command()
def estimate(
    models: list[Path] | None = typer.Argument(
        None,
        help="Models resident in VRAM at the same time (.safetensors, .gguf, .pth or shard directories).",
    ),
    workflow: Path | None = typer.Option(
        None,
        "--workflow",
        help="ComfyUI workflow JSON (UI or API format) instead of a list of models.",
    ),
    models_dir: Path | None = typer.Option(
        None,
        "--models-dir",
        help="ComfyUI models folder. Default: found above the workflow file.",
    ),
    activations_mb: float = typer.Option(
        0,
        "--activations-mb",
        help="VRAM for activations / latents on top of the weights.",
    ),
    observed_peak_mb: float | None = typer.Option(
        None,
        "--observed-peak-mb",
        help="Peak VRAM of a real run (from `processes --watch`); activations = peak - weights.",
    ),
    available_mb: float | None = typer.Option(
        None,
        "--available-mb",
        help="VRAM available to the workflow. Default: free VRAM right now (NVML).",
    ),
    reserve_mb: float = typer.Option(
        DEFAULT_RESERVE_MB,
        "--reserve-mb",
        help="Kept free for CUDA context, workspaces and fragmentation.",
    ),
    transfer_gb_s: float | None = typer.Option(
        None,
        "--transfer-gb-s",
        help="Host -> device bandwidth for offloading. Default: measured when offloading is needed.",
    ),
    gpu: int = typer.Option(
        0,
        "--gpu",
        help="GPU index.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the result as JSON.",
    ),
) -> None:
    """Estimate whether models fit in VRAM and what offloading the rest costs."""
    if not models and workflow is None:
        console.print("[red]Cannot estimate:[/red] pass model files or --workflow")
        raise typer.Exit(code=1)

    options = dict(
        activations_mb=activations_mb,
        observed_peak_mb=observed_peak_mb,
        available_mb=available_mb,
        reserve_mb=reserve_mb,
        transfer_gb_s=transfer_gb_s,
        device_index=gpu,
    )
    try:
        if workflow is not None:
            info = estimate_comfy_workflow(workflow, models_dir=models_dir, **options)
        else:
            info = estimate_workflow(models, **options)
    except (ModelInspectError, OSError, ValueError, RuntimeError) as exc:
        console.print(f"[red]Cannot estimate:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        console.print_json(json.dumps(info, indent=2, ensure_ascii=False))
        return

    print_estimate_table(console, info)


if __name__ == "__main__":
    app()