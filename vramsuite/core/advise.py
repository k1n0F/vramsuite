"""
Advice for a ComfyUI workflow estimate: what to change, with numbers.

Every suggestion is computed from things VRAM Suite measured or read:
weights from file headers, activations from a profile (or given), PCIe
bandwidth from the transfer benchmark, RAM from the OS. Nothing is applied;
the user decides.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from vramsuite.core.estimate import BYTES_PER_MB, model_weights
from vramsuite.core.runtime import system_memory_mb
from vramsuite.core.workflow import TEXT_ENCODER_NODE, steps_run

CONFIDENCE = {"profile": "high", "observed_peak": "high", "given": "medium", "not_set": "low"}
# Tokens that only describe precision or packing, not which model a file is.
PRECISION_TOKENS = {"fp32", "fp16", "bf16", "fp8", "e4m3fn", "e5m2", "e4m3", "scaled", "fp4", "nvfp4", "mixed"}
SIXTEEN_BIT = ("F16", "BF16", "F32")


def _model_key(name: str) -> str:
    """'umt5_xxl_fp16.safetensors' and 'umt5_xxl_fp8_e4m3fn_scaled.safetensors' -> 'umt5_xxl'."""
    tokens = re.split(r"[_\-.]", Path(name.replace("\\", "/")).stem.lower())
    return "_".join(token for token in tokens if token not in PRECISION_TOKENS)


def find_fp8_variant(model: dict[str, Any], models_dir: Path) -> dict[str, Any] | None:
    """An FP8 file of the same model in the same folder, or in the text encoder folders."""
    folders = {Path(model["path"]).parent, models_dir / "text_encoders", models_dir / "clip"}
    key = _model_key(model["name"])
    for folder in folders:
        for path in folder.glob("*.safetensors") if folder.is_dir() else []:
            if path.name != Path(model["path"]).name and _model_key(path.name) == key:
                variant = model_weights(path)
                if variant["dtypes"][0].startswith("F8"):
                    return variant
    return None


def _round_size(value: float, multiple: int = 16) -> int:
    return max(int(value // multiple) * multiple, multiple)


def _round_frames(value: float, frames: int) -> int:
    """Keep the n*k+1 form video models need (WAN 4n+1, LTX 8n+1)."""
    step = next((k for k in (8, 4) if (frames - 1) % k == 0), 1)
    return max(int((value - 1) // step) * step + 1, 1)


def _size_suggestion(info: dict[str, Any], latent: dict[str, Any]) -> dict[str, Any] | None:
    width, height = latent.get("width"), latent.get("height")
    frames = latent.get("length", latent.get("num_frames", 1))
    if not all(isinstance(value, int) for value in (width, height, frames)) or not info["activations_mb"]:
        return None

    # Activations are assumed linear in pixels x frames, which holds for flash /
    # sage attention. Quadratic attention would need a profile per size.
    room_mb = info["budget_mb"] - info["weights_mb"]
    if room_mb <= 0:
        return None
    scale = room_mb / info["activations_mb"]

    side = math.sqrt(scale)
    size = f"{_round_size(width * side)}x{_round_size(height * side)}"
    options = [f"{size} at {frames} frames" if frames > 1 else size]
    if frames > 1 and scale < 1:
        options.append(f"{_round_frames(frames * scale, frames)} frames at {width}x{height}")

    if scale >= 1:
        message = f"Largest size that still fits: about {options[0]} (now {width}x{height})."
    else:
        message = f"To fit without offloading: about {' or '.join(options)} (now {width}x{height}, {frames} frames)."
    return {"code": "max_size" if scale >= 1 else "reduce_size", "message": message, "scale": round(scale, 2)}


def _transfer_gb_s(info: dict[str, Any]) -> float | None:
    offload = info.get("offload") or {}
    return offload.get("pageable_gb_s") or offload.get("pinned_gb_s")


def advise(info: dict[str, Any], analysis: dict[str, Any], measure_transfer=None) -> dict[str, Any]:
    """
    Suggestions for an estimate_comfy_workflow result.

    measure_transfer() -> {"pageable_gb_s", ...} is called only when a reload
    cost has to be priced and the estimate did not measure PCIe already.
    """
    suggestions = []
    models_dir = Path(analysis["models_dir"]) if analysis["models_dir"] else None

    for model in analysis["models"]:
        # Same rule as the text_encoder_16bit warning of analyze-workflow.
        if (
            not model["path"]
            or not TEXT_ENCODER_NODE.search(model["node_type"])
            or model["dtypes"][0] not in SIXTEEN_BIT
            or model["weights_mb"] <= 2048
        ):
            continue
        variant = find_fp8_variant(model, models_dir) if models_dir else None
        largest = model["weights_mb"] == info["weights_mb"]
        effect = "lowers the planned peak" if largest else "less to reload between stages"
        if variant:
            saves = round(model["weights_mb"] - variant["weights_mb"], 1)
            message = f"Use {variant['name']} ({variant['weights_mb']} MB) instead of {model['name']}: -{saves} MB, {effect}."
        else:
            saves = round(model["weights_mb"] / 2, 1)
            message = f"An FP8 version of {model['name']} would be about {saves} MB smaller ({effect})."
        suggestions.append({"code": "fp8_text_encoder", "message": message, "saves_mb": saves})

    if analysis["latents"]:
        size = _size_suggestion(info, analysis["latents"][0])
        if size:
            suggestions.append(size)

    steps = [steps_run(sampler) for sampler in analysis["samplers"]]
    total_steps = sum(step for step in steps if step) or None
    offload = info.get("offload")
    if offload and total_steps:
        costs = [
            f"+{offload[f'seconds_per_step_{memory}'] * total_steps:.1f} s with {memory} memory"
            for memory in ("pinned", "pageable") if offload.get(f"seconds_per_step_{memory}") is not None
        ]
        if costs:
            suggestions.append({
                "code": "offload_cost",
                "message": f"Offloading {offload['offload_mb']} MB over {total_steps} steps costs at least {', '.join(costs)} per run.",
                "steps": total_steps,
            })

    together = info["all_weights_mb"] + info["activations_mb"]
    if together > info["budget_mb"] and len(analysis["models"]) > 1:
        reload_mb = round(min(info["all_weights_mb"], together - info["budget_mb"]), 1)
        gb_s = _transfer_gb_s(info)
        if gb_s is None and measure_transfer is not None:
            measured = measure_transfer()
            gb_s = measured.get("pageable_gb_s") or measured.get("pinned_gb_s")
        # PCIe time only: loading also casts and prepares the weights.
        cost = f" (at least {reload_mb * BYTES_PER_MB / (gb_s * 1e9):.1f} s of PCIe transfer at {gb_s} GB/s)" if gb_s else ""
        message = f"All models together need {together:.0f} MB: ComfyUI reloads about {reload_mb} MB from RAM on every run{cost}."

        memory = system_memory_mb()
        if memory and info["all_weights_mb"] > memory[0] * 0.75:
            message += (
                f" They are {info['all_weights_mb'] / 1024:.1f} GB against {memory[0] / 1024:.0f} GB of RAM:"
                " expect models to be read from disk, which is much slower."
            )
        suggestions.append({"code": "reload_between_stages", "message": message, "reload_mb": reload_mb})

    for warning in analysis["warnings"]:
        if warning["code"] in ("vae_decode_untiled", "vae_tiling_off"):
            suggestions.append({"code": warning["code"], "message": warning["message"]})

    return {"confidence": CONFIDENCE[info["activations_source"]], "suggestions": suggestions}
