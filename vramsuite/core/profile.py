"""
Workflow profiles: real peak VRAM and run time of ComfyUI runs, per config_hash.

Passive capture: vramsuite submits nothing. It polls ComfyUI's queue over the
HTTP API, samples per-process VRAM from outside while a prompt runs, then
reads the exact prompt that ran and its timestamps from /history. The prompt
gives the config_hash, so a profile is matched to a workflow file by its
models and settings, not by file name.

Profiles are JSON files, one per config_hash, in a profiles folder
(./profiles by default); each capture appends a run.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from vramsuite.core.processes import snapshot_gpu_processes, watch_gpu_processes
from vramsuite.core.workflow import analyze_workflow_data

PROFILE_SCHEMA = "workflow_profile@0.3"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_PROFILES_DIR = Path("profiles")


def _get_json(comfy_url: str, path: str) -> Any:
    with urllib.request.urlopen(comfy_url.rstrip("/") + path, timeout=10) as response:
        return json.load(response)


def free_comfy_memory(comfy_url: str) -> None:
    """Ask ComfyUI to unload models and empty its allocator cache (the "Unload models" button)."""
    request = urllib.request.Request(
        comfy_url.rstrip("/") + "/free",
        data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10):
        pass
    # The flags are applied by ComfyUI's queue worker within a second or two.
    time.sleep(3)


def _running_prompt_id(comfy_url: str) -> str | None:
    # queue_running entries: [number, prompt_id, prompt, extra_data, outputs]
    running = _get_json(comfy_url, "/queue").get("queue_running") or []
    return running[0][1] if running else None


def _comfy_process(processes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    The ComfyUI process among VRAM users (sorted by peak, biggest first).

    The biggest user is not a safe guess on Windows: WDDM counters of dwm and
    overlays include shared surfaces and can exceed ComfyUI's own usage.
    """
    # Not this process: its own command line may contain --comfy-url.
    processes = [process for process in processes if process["pid"] != os.getpid()]
    # Name heuristics. If they ever pick the wrong process, match the one
    # listening on ComfyUI's port instead.
    for matches in (
        lambda process: "comfy" in (process.get("command_line") or "").lower(),
        lambda process: (process.get("name") or "").lower().startswith("python"),
        lambda process: True,
    ):
        for process in processes:
            if matches(process):
                return process
    return None


def capture_run(
        comfy_url: str = DEFAULT_COMFY_URL,
        device_index: int = 0,
        interval: float = 0.5,
        wait_seconds: float = 3600,
        free_first: bool = True,
        on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Wait for the next prompt ComfyUI runs, watch it to the end, return one run record.

    free_first unloads ComfyUI's models before waiting. Its caching allocator
    keeps the high-water mark of earlier prompts, so without it the peak may
    belong to a previous, heavier workflow.
    """
    status = on_status or (lambda message: None)
    if free_first:
        status("Unloading models in ComfyUI so the peak belongs to this prompt...")
        free_comfy_memory(comfy_url)

    deadline = time.monotonic() + wait_seconds
    prompt_id = _running_prompt_id(comfy_url)
    if prompt_id is None:
        status("Waiting for a prompt: press Queue in ComfyUI...")
    while prompt_id is None:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"No prompt started within {wait_seconds:g} s")
        time.sleep(interval)
        prompt_id = _running_prompt_id(comfy_url)

    status(f"Watching prompt {prompt_id[:8]}...")
    # Taken right after the prompt started: close to what the process held before it.
    start = {row["pid"]: row["dedicated_mb"] for row in snapshot_gpu_processes(device_index)["processes"]}
    watch = watch_gpu_processes(
        seconds=24 * 3600,
        interval=interval,
        device_index=device_index,
        until=lambda: _running_prompt_id(comfy_url) != prompt_id,
    )

    # History is written right after the prompt leaves the queue.
    entry = None
    for _ in range(20):
        entry = _get_json(comfy_url, f"/history/{prompt_id}").get(prompt_id)
        if entry:
            break
        time.sleep(interval)
    if entry is None:
        raise RuntimeError(f"Prompt {prompt_id} finished but is not in ComfyUI history")

    timestamps = {message[0]: message[1] for message in entry["status"].get("messages") or []}
    started = timestamps.get("execution_start", {}).get("timestamp")
    finished = (timestamps.get("execution_success") or timestamps.get("execution_error") or {}).get("timestamp")
    cached = timestamps.get("execution_cached", {}).get("nodes") or []
    process = _comfy_process(watch["processes"])
    analysis = analyze_workflow_data(entry["prompt"][2], name=f"prompt {prompt_id[:8]}")

    return {
        "analysis": analysis,
        "run": {
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "prompt_id": prompt_id,
            "status": entry["status"].get("status_str"),
            "run_seconds": round((finished - started) / 1000, 2) if started and finished else None,
            # Nodes ComfyUI did not re-run (same inputs as the previous prompt), e.g. loaders.
            "cached_nodes": len(cached),
            "freed_before": free_first,
            "process": process and process["name"],
            "pid": process and process["pid"],
            "start_mb": process and start.get(process["pid"]),
            "peak_mb": process and process["peak_dedicated_mb"],
            "gpu": watch["gpu"]["name"],
            "gpu_total_mb": watch["gpu"]["total_mb"],
            "samples": watch["samples"],
        },
    }


def profile_path(profiles_dir: Path, config: str) -> Path:
    return Path(profiles_dir) / f"{config}.json"


def load_profile(profiles_dir: str | Path, config: str) -> dict[str, Any] | None:
    path = profile_path(Path(profiles_dir), config)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def save_run(profiles_dir: str | Path, captured: dict[str, Any]) -> Path:
    """Append a captured run to the profile of its config_hash."""
    analysis = captured["analysis"]
    path = profile_path(Path(profiles_dir), analysis["config_hash"])
    profile = load_profile(profiles_dir, analysis["config_hash"]) or {
        "schema": PROFILE_SCHEMA,
        "config_hash": analysis["config_hash"],
        "graph_hash": analysis["graph_hash"],
        "models": [{"name": model["name"], "node_type": model["node_type"]} for model in analysis["models"]],
        "latents": analysis["latents"],
        "samplers": analysis["samplers"],
        "runs": [],
    }
    profile["runs"].append(captured["run"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def summarize_profile(profile: dict[str, Any]) -> dict[str, Any] | None:
    """Peak and timing over successful runs: what estimate uses."""
    runs = [run for run in profile["runs"] if run["status"] == "success" and run["peak_mb"]]
    if not runs:
        return None
    seconds = [run["run_seconds"] for run in runs if run["run_seconds"]]
    return {
        "config_hash": profile["config_hash"],
        "runs": len(runs),
        "peak_mb": max(run["peak_mb"] for run in runs),
        "fastest_seconds": min(seconds) if seconds else None,
        "slowest_seconds": max(seconds) if seconds else None,
    }
