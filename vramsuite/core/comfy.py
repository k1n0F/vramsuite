"""
VRAM estimate for a ComfyUI workflow file: the layer that combines the others.

workflow (what the file loads and its settings), profile (recorded runs),
estimate (fits / offload) and advise (what to change) do not know about each
other's callers; this module puts them together. Imports only go downwards:

    estimate  <-  workflow  <-  profile, advise  <-  comfy
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vramsuite.core.advise import advise
from vramsuite.core.estimate import estimate_workflow, measure_transfer
from vramsuite.core.profile import load_profile, summarize_profile
from vramsuite.core.workflow import analyze_workflow


def estimate_comfy_workflow(
        workflow_path: str | Path,
        models_dir: str | Path | None = None,
        profiles_dir: str | Path | None = None,
        **estimate_kwargs: Any,
) -> dict[str, Any]:
    """
    Estimate VRAM for a ComfyUI workflow file.

    ComfyUI loads models stage by stage (text encoder, then the diffusion
    model, then the VAE) and unloads what does not fit, so the plan is made
    for the largest model. The sum of all models is reported as well.

    With profiles_dir, a recorded profile of the same config_hash supplies the
    observed peak unless activations or a peak were given explicitly.
    """
    analysis = analyze_workflow(workflow_path, models_dir)
    if analysis["models_dir"] is None:
        raise RuntimeError(f"Cannot find the ComfyUI models folder above {analysis['name']}; pass --models-dir")

    found = [model for model in analysis["models"] if model["path"]]
    if not found:
        raise RuntimeError(
            f"None of the {len(analysis['models'])} models of {analysis['name']} were found in {analysis['models_dir']}"
        )

    profile = None
    if profiles_dir is not None:
        recorded = load_profile(profiles_dir, analysis["config_hash"])
        profile = summarize_profile(recorded) if recorded else None
    use_profile = bool(
        profile and estimate_kwargs.get("observed_peak_mb") is None and not estimate_kwargs.get("activations_mb")
    )
    if use_profile:
        estimate_kwargs["observed_peak_mb"] = profile["peak_mb"]

    info = estimate_workflow([model["path"] for model in found], largest_only=True, **estimate_kwargs)
    if use_profile:
        info["activations_source"] = "profile"
    for model, node in zip(info["models"], found):
        model["node_type"] = node["node_type"]
    info["workflow"] = {
        "name": analysis["name"],
        "models_dir": analysis["models_dir"],
        "missing": [model["name"] for model in analysis["models"] if not model["path"]],
        "config_hash": analysis["config_hash"],
        "profile": profile,
    }

    given_gb_s = estimate_kwargs.get("transfer_gb_s")
    info["advice"] = advise(
        info,
        analysis,
        measure_transfer=lambda: (
            {"pageable_gb_s": given_gb_s} if given_gb_s else measure_transfer(estimate_kwargs.get("device_index", 0))
        ),
    )
    return info
