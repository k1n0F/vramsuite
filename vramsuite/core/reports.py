from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from vramsuite import __version__
from vramsuite.core.doctor import TOOLKIT_MISSING


def print_doctor_header(console: Console) -> None:
    """Print the VRAM Suite doctor header."""
    console.print(
        Panel(
            f"[bold cyan]VRAM Suite Doctor[/bold cyan] v{__version__}",
            border_style="cyan",
        )
    )


def print_capabilities_table(console: Console, vramcard: dict[str, Any]) -> None:
    """Which native libraries were found, and what each one enables."""
    stack = vramcard.get("stack") or {}
    rows = (
        ("NVML", stack.get("nvml_available"), "NVIDIA driver", "memory, device details, PCIe, processes"),
        ("CUDA driver API", (stack.get("cuda_driver") or {}).get("available"), "NVIDIA driver", "device attributes, --probe, transfer, estimate"),
        ("CUDA runtime", (stack.get("cuda_runtime") or {}).get("available"), "CUDA Toolkit", "--compute-check"),
        ("cuBLAS", (stack.get("cublas") or {}).get("available"), "CUDA Toolkit", "--compute-check"),
    )

    table = Table(title="Capabilities")
    table.add_column("Library", style="bold")
    table.add_column("Found")
    table.add_column("Comes with")
    table.add_column("Enables")
    for name, found, source, enables in rows:
        table.add_row(name, "[green]yes[/green]" if found else "[yellow]no[/yellow]", source, enables)
    console.print(table)


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
    if any(result.get("error") == TOOLKIT_MISSING for result in compute_info.values()):
        console.print(
            "[yellow]Compute checks need cuBLAS from the CUDA Toolkit[/yellow] "
            "(https://developer.nvidia.com/cuda-downloads). Everything else works with only the NVIDIA driver."
        )


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


def print_transfer_table(console: Console, info: dict[str, Any]) -> None:
    """Print host <-> device bandwidth with the PCIe link it ran on."""
    pcie = info["pcie"]
    link = pcie["theoretical_gb_s"]

    table = Table(title=f"GPU {info['device_index']} {info['name']}: transfers ({info['size_mb']} MB x {info['repeats']})")
    table.add_column("Direction", style="bold")
    table.add_column("Host memory")
    table.add_column("Best GB/s", justify="right")
    table.add_column("Median GB/s", justify="right")
    table.add_column("% of link", justify="right")
    table.add_column("ms per GB", justify="right")
    table.add_column("Error")

    for row in info["results"]:
        best = row["best_gb_s"]
        over_pcie = row["direction"] != "device_to_device"
        table.add_row(
            row["direction"],
            row["host_memory"],
            "" if best is None else str(best),
            "" if row["median_gb_s"] is None else str(row["median_gb_s"]),
            f"{100 * best / link:.0f}%" if best and link and over_pcie else "",
            f"{1000 / best:.1f}" if best else "",
            row["error"] or "",
        )

    console.print(table)
    console.print(
        f"PCIe Gen{pcie['link_gen_max']} x{pcie['link_width_current']} "
        f"(card supports x{pcie['link_width_max']}): theoretical {link} GB/s per direction"
        + (
            f", {pcie['theoretical_full_width_gb_s']} GB/s at full width"
            if pcie["link_width_current"] != pcie["link_width_max"]
            else ""
        )
    )
    console.print("[dim]device_to_device counts read + write, i.e. VRAM bandwidth.[/dim]")


ESTIMATE_STATUS = {
    "fits": "[green]fits[/green]",
    "offload": "[yellow]fits with offloading[/yellow]",
    "does_not_fit": "[red]does not fit[/red]",
}


def _number(value: Any) -> str:
    """Settings values: numbers as is, linked widgets as '?'."""
    return str(value) if isinstance(value, (int, float)) else "?"


def _latent_size(latent: dict[str, Any]) -> str:
    size = f"{_number(latent.get('width'))}x{_number(latent.get('height'))}"
    frames = latent.get("length", latent.get("num_frames"))
    return f"{size}, {_number(frames)} frames" if frames not in (None, 1) else size


def print_workflow_analysis(console: Console, info: dict[str, Any]) -> None:
    """Print the offline analysis of one ComfyUI workflow."""
    summary = Table(title=f"{info['name']} ({info['format']} format, {info['node_count']} nodes)")
    summary.add_column("Field", style="bold")
    summary.add_column("Value")
    summary.add_row("Graph hash", info["graph_hash"])
    summary.add_row("Config hash", info["config_hash"])
    summary.add_row("Weights MB", f"{info['weights_mb']['largest']} largest, {info['weights_mb']['all']} all")
    summary.add_row("Models folder", str(info["models_dir"]))
    console.print(summary)

    models = Table(title="Models")
    models.add_column("File", style="bold")
    models.add_column("Node")
    models.add_column("MB", justify="right")
    models.add_column("Dtypes")
    for model in info["models"]:
        if model["path"] is None:
            models.add_row(model["name"], model["node_type"], "", "[yellow]not found[/yellow]")
        else:
            models.add_row(model["name"], model["node_type"], str(model["weights_mb"]), ", ".join(model["dtypes"]))
    console.print(models)

    settings = Table(title="Settings")
    settings.add_column("Node", style="bold")
    settings.add_column("Values")
    for latent in info["latents"]:
        linked = {value for value in latent.values() if isinstance(value, str) and value.startswith("linked from")}
        note = f" [dim]({'; '.join(sorted(linked))})[/dim]" if linked else ""
        settings.add_row(latent["node_type"], f"{_latent_size(latent)}, batch {_number(latent.get('batch_size', 1))}{note}")
    for sampler in info["samplers"]:
        values = [f"{_number(sampler['steps'])} steps", f"cfg {_number(sampler.get('cfg'))}"]
        values += [str(sampler[key]) for key in ("sampler_name", "scheduler") if key in sampler]
        settings.add_row(sampler["node_type"], ", ".join(values))
    console.print(settings)

    for warning in info["warnings"]:
        console.print(f"[yellow]Warning:[/yellow] {warning['message']}")


def print_folder_analysis(console: Console, info: dict[str, Any]) -> None:
    """Print one summary row per workflow in a folder."""
    table = Table(title=f"Workflows in {info['folder']}")
    table.add_column("Workflow", style="bold")
    table.add_column("Models", justify="right")
    table.add_column("Largest MB", justify="right")
    table.add_column("All MB", justify="right")
    table.add_column("Size")
    table.add_column("Steps", justify="right")
    table.add_column("Warnings", justify="right")
    for workflow in info["workflows"]:
        found = sum(1 for model in workflow["models"] if model["path"])
        steps = [_number(sampler["steps"]) for sampler in workflow["samplers"]]
        warnings = len(workflow["warnings"])
        table.add_row(
            workflow["name"],
            f"{found}/{len(workflow['models'])}",
            str(workflow["weights_mb"]["largest"]),
            str(workflow["weights_mb"]["all"]),
            "; ".join(_latent_size(latent) for latent in workflow["latents"]),
            "+".join(steps),
            f"[yellow]{warnings}[/yellow]" if warnings else "0",
        )
    console.print(table)
    if info["skipped"]:
        console.print(f"[dim]Skipped {len(info['skipped'])} JSON files that are not workflows.[/dim]")


def print_profile_run(console: Console, captured: dict[str, Any], path: Any) -> None:
    """Print one recorded ComfyUI run."""
    analysis = captured["analysis"]
    run = captured["run"]
    table = Table(title=f"Prompt {run['prompt_id'][:8]}: {run['status']}")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Config hash", analysis["config_hash"])
    table.add_row("Models", ", ".join(model["name"] for model in analysis["models"]))
    for latent in analysis["latents"]:
        table.add_row("Size", f"{_latent_size(latent)} ({latent['node_type']})")
    table.add_row("Run seconds", str(run["run_seconds"]))
    table.add_row("Cached nodes", str(run["cached_nodes"]))
    table.add_row("Process", f"{run['process'] or 'unknown name'} (pid {run['pid']})")
    table.add_row("Started cold", "yes (models unloaded first)" if run["freed_before"] else "no")
    table.add_row("VRAM at start MB", str(run["start_mb"]))
    table.add_row("Peak VRAM MB", str(run["peak_mb"]))
    table.add_row("GPU", f"{run['gpu']} ({run['gpu_total_mb']} MB)")
    console.print(table)
    console.print(f"[green]Saved:[/green] {path}")


def print_estimate_table(console: Console, info: dict[str, Any]) -> None:
    """Print the VRAM plan for a set of models and its offloading cost."""
    workflow = info.get("workflow")
    title = f"GPU {info['device_index']}: memory estimate"
    table = Table(title=f"{workflow['name']} on {title}" if workflow else title)
    table.add_column("Field", style="bold")
    table.add_column("MB", justify="right")
    table.add_column("Note")

    for model in info["models"]:
        note = ", ".join(model["dtypes"])
        if "node_type" in model:
            note = f"{model['node_type']}: {note}"
        table.add_row(model["name"], str(model["weights_mb"]), note)
    if info["weights_basis"] == "largest":
        table.add_row("Weights", str(info["weights_mb"]), "largest model, loaded stage by stage")
    table.add_row("Activations", str(info["activations_mb"]), info["activations_source"])
    table.add_row("Required", str(info["required_mb"]), "weights + activations")
    table.add_row("Available", str(info["available_mb"]), info["available_source"])
    table.add_row("Reserve", str(info["reserve_mb"]), "CUDA context, workspaces, fragmentation")
    table.add_row("Budget", str(info["budget_mb"]), "available - reserve")
    if info["status"] == "fits":
        table.add_row("Headroom", str(info["headroom_mb"]), "")
    else:
        table.add_row("Overflow", str(info["overflow_mb"]), "required - budget")
    table.add_row("Status", "", ESTIMATE_STATUS[info["status"]])
    table.add_row("Risk level", "", f"{info['risk']['risk_level']} ({info['risk']['usage_ratio']:.0%} of budget if fully resident)")
    console.print(table)

    if info["activations_source"] == "not_set":
        hint = "record a run with `vramsuite profile`, " if workflow else ""
        console.print(
            f"[yellow]Activations not set:[/yellow] only weights are counted. To include them, {hint}"
            "pass --activations-mb, or --observed-peak-mb with the peak from `vramsuite processes --watch`."
        )

    profile = (workflow or {}).get("profile")
    if profile:
        timing = (
            f", run took {profile['fastest_seconds']}-{profile['slowest_seconds']} s"
            if profile["fastest_seconds"] is not None else ""
        )
        console.print(
            f"[green]Profile {profile['config_hash']}:[/green] {profile['runs']} recorded runs, "
            f"peak {profile['peak_mb']} MB{timing}"
        )

    if workflow:
        for name in workflow["missing"]:
            console.print(f"[yellow]Not found in {workflow['models_dir']}:[/yellow] {name}")
        together = info["all_weights_mb"] + info["activations_mb"]
        if together <= info["budget_mb"]:
            console.print(f"[dim]All models together: {together:.1f} MB, they stay loaded between runs.[/dim]")
        else:
            console.print(
                f"[dim]All models together: {together:.1f} MB, more than the budget: "
                "ComfyUI unloads and reloads models between stages on every run.[/dim]"
            )

    advice = info.get("advice")
    if advice:
        for suggestion in advice["suggestions"]:
            console.print(f"[cyan]Suggestion:[/cyan] {suggestion['message']}")
        console.print(f"[dim]Confidence: {advice['confidence']} (activations: {info['activations_source']}).[/dim]")

    offload = info["offload"]
    if offload is None:
        return

    if offload["error"]:
        console.print(f"[red]Transfer measurement failed:[/red] {offload['error']}")
        return

    cost = Table(title=f"Offloading {offload['offload_mb']} MB of weights per step ({offload['source']} PCIe)")
    cost.add_column("Host memory", style="bold")
    cost.add_column("GB/s", justify="right")
    cost.add_column("Extra seconds per step", justify="right")
    for memory in ("pinned", "pageable"):
        gb_s = offload[f"{memory}_gb_s"]
        if gb_s is not None:
            cost.add_row(memory, str(gb_s), str(offload[f"seconds_per_step_{memory}"]))
    console.print(cost)
    console.print(
        "[dim]Lower bound: assumes each offloaded weight crosses PCIe once per step. "
        "Multiply by the number of steps for the whole run.[/dim]"
    )


def print_validation(console: Console, path: str, info: dict[str, Any]) -> None:
    """Print .vramcard validation issues."""
    if not info["issues"]:
        console.print(f"[green]Valid:[/green] {path} ({info['schema']})")
        return

    table = Table(title=f"{path} ({info['schema']})")
    table.add_column("Level", style="bold")
    table.add_column("Field")
    table.add_column("Problem")
    for issue in info["issues"]:
        color = "red" if issue["level"] == "error" else "yellow"
        table.add_row(f"[{color}]{issue['level']}[/{color}]", issue["field"], issue["message"])
    console.print(table)
    console.print("[green]Valid[/green] (warnings only)" if info["valid"] else "[red]Invalid[/red]")


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