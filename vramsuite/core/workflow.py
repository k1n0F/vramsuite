"""
ComfyUI workflow analysis: which model files a workflow loads, its generation
settings and risky patterns. Whether it fits is decided in core/comfy.py.

Both formats ComfyUI saves are read:
- UI format (workflows folder, "Save"): {"nodes": [{"id", "type", "mode", "inputs", "widgets_values"}], "links": [...], "definitions": {"subgraphs": [...]}}
- API format ("Export (API)"): {"<id>": {"class_type", "inputs"}}

Nothing is matched against a table of node names: models are found by file
extension, settings by widget name (width, length, steps, ...), so custom
nodes (GGUF loaders, WanVideoWrapper, ...) are covered too.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from vramsuite.core.estimate import model_weights

MODEL_SUFFIXES = (".safetensors", ".sft", ".gguf", ".pth", ".pt", ".ckpt", ".bin")
# LiteGraph node modes: 2 = muted, 4 = bypassed. Such nodes never run.
SKIPPED_MODES = {2, 4}
# Frontend-only nodes: they never reach the backend (absent from API format).
UI_ONLY_TYPES = {"Reroute", "Note", "MarkdownNote", "PrimitiveNode", "SetNode", "GetNode"}
# The value after a seed widget is the frontend's control_after_generate.
SEED_CONTROLS = {"fixed", "increment", "decrement", "randomize"}

LATENT_KEYS = ("width", "height", "length", "num_frames", "batch_size")
SAMPLER_KEYS = ("steps", "cfg", "sampler_name", "scheduler", "denoise")
# Which of the steps a sampler runs (WAN 2.2 splits them between two models).
# Not part of config_hash: they change the time, not the memory.
STEP_RANGE_KEYS = ("start_at_step", "end_at_step", "start_step", "end_step")
TEXT_ENCODER_NODE = re.compile(r"clip|text.?encoder|t5", re.IGNORECASE)
LATENT_NODE = re.compile(r"latent|video", re.IGNORECASE)


def _linked(source_type: str | None) -> str:
    """A widget fed by another node whose value is only known at run time (e.g. GetImageSize)."""
    return f"linked from {source_type or '?'}"


def _ui_widgets(node: dict[str, Any], resolve) -> dict[str, Any]:
    values = node.get("widgets_values") or []
    if isinstance(values, dict):
        return dict(values)

    widgets: dict[str, Any] = {}
    position = 0
    for item in node.get("inputs") or []:
        name = (item.get("widget") or {}).get("name")
        if name is None:
            continue
        value = values[position] if position < len(values) else None
        position += 1
        if "seed" in name and position < len(values) and values[position] in SEED_CONTROLS:
            position += 1
        # A linked widget keeps a stale value in widgets_values.
        widgets[name] = resolve(item["link"]) if item.get("link") is not None else value
    return widgets


def _ui_nodes(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    subgraphs = (workflow.get("definitions") or {}).get("subgraphs") or []
    raw = [*workflow["nodes"], *(node for graph in subgraphs for node in graph.get("nodes") or [])]
    by_id = {node.get("id"): node for node in workflow["nodes"]}
    # [id, from_node, from_slot, to_node, to_slot, type]
    sources = {link[0]: link[1] for link in workflow.get("links") or [] if isinstance(link, list)}

    def resolve(link_id: int) -> Any:
        source = by_id.get(sources.get(link_id)) or {}
        if source.get("type", "").startswith("Primitive") and source.get("widgets_values"):
            return source["widgets_values"][0]
        return _linked(source.get("type"))

    subgraph_ids = {graph.get("id") for graph in subgraphs}
    nodes = []
    for node in raw:
        node_type = node.get("type", "?")
        if node.get("mode") in SKIPPED_MODES or node_type in UI_ONLY_TYPES or node_type in subgraph_ids:
            continue
        values = node.get("widgets_values") or []
        nodes.append({
            "type": node_type,
            "widgets": _ui_widgets(node, resolve),
            "values": list(values.values()) if isinstance(values, dict) else values,
        })
    return nodes


def _api_nodes(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    def resolve(link: list[Any]) -> Any:
        source = workflow.get(str(link[0])) or {}
        if source.get("class_type", "").startswith("Primitive") and "value" in (source.get("inputs") or {}):
            return source["inputs"]["value"]
        return _linked(source.get("class_type"))

    nodes = []
    for node in workflow.values():
        if not (isinstance(node, dict) and "class_type" in node):
            continue
        inputs = node.get("inputs") or {}
        nodes.append({
            "type": node["class_type"],
            "widgets": {key: resolve(value) if isinstance(value, list) else value for key, value in inputs.items()},
            "values": list(inputs.values()),
        })
    return nodes


def workflow_nodes(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Nodes that run: [{"type", "widgets": {name: value}, "values": [...]}], either format."""
    if "nodes" in workflow:
        return _ui_nodes(workflow)
    if any(isinstance(node, dict) and "class_type" in node for node in workflow.values()):
        return _api_nodes(workflow)
    raise ValueError("not a ComfyUI workflow (no nodes)")


def find_models(nodes: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Model file names the nodes reference, once each, in node order."""
    found: dict[str, str] = {}
    for node in nodes:
        for value in node["values"]:
            if isinstance(value, str) and value.lower().endswith(MODEL_SUFFIXES):
                found.setdefault(value, node["type"])
    return [{"name": name, "node_type": node_type} for name, node_type in found.items()]


def find_settings(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Latent sizes and sampler settings, read by widget name."""
    latents = []
    samplers = []
    for node in nodes:
        widgets = node["widgets"]
        if "width" in widgets and "height" in widgets and LATENT_NODE.search(node["type"]):
            latents.append({"node_type": node["type"], **{key: widgets[key] for key in LATENT_KEYS if key in widgets}})
        if "steps" in widgets:
            keys = SAMPLER_KEYS + STEP_RANGE_KEYS
            samplers.append({"node_type": node["type"], **{key: widgets[key] for key in keys if key in widgets}})
    return {"latents": latents, "samplers": samplers}


def steps_run(sampler: dict[str, Any]) -> int | None:
    """Steps a sampler actually runs: steps, cut to its start/end range."""
    steps = sampler.get("steps")
    if not isinstance(steps, int):
        return None
    start = sampler.get("start_at_step", sampler.get("start_step", 0))
    end = sampler.get("end_at_step", sampler.get("end_step", steps))
    if not isinstance(start, int) or not isinstance(end, int):
        return steps
    # -1 (WanVideoSampler) and 10000 (KSamplerAdvanced) both mean "to the last step".
    end = steps if end < 0 else min(end, steps)
    return max(end - start, 0)


def _frames(latent: dict[str, Any]) -> Any:
    return latent.get("length", latent.get("num_frames", 1))


def _short_hash(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


def find_models_dir(workflow_path: Path) -> Path | None:
    """ComfyUI/models above a workflow saved in ComfyUI/user/default/workflows."""
    for parent in workflow_path.resolve().parents:
        if (parent / "models" / "checkpoints").is_dir():
            return parent / "models"
    return None


def resolve_model(name: str, models_dir: Path) -> Path | None:
    # Names are relative to a model folder and may include subfolders ("wan\\x.safetensors").
    relative = Path(name.replace("\\", "/"))
    for folder in models_dir.iterdir():
        if (folder / relative).is_file():
            return folder / relative
    # extra_model_paths.yaml is not read; the recursive search covers models
    # moved to subfolders of models_dir.
    return next((path for path in models_dir.rglob(relative.name) if path.is_file()), None)


def _warnings(nodes: list[dict[str, Any]], models: list[dict[str, Any]], latents: list[dict[str, Any]]) -> list[dict[str, str]]:
    warnings = []

    for model in models:
        if model.get("path") is None:
            warnings.append({"code": "model_missing", "message": f"{model['name']} ({model['node_type']}) was not found in the models folder"})
        elif (
            TEXT_ENCODER_NODE.search(model["node_type"])
            and model["dtypes"][0] in ("F16", "BF16", "F32")
            and model["weights_mb"] > 2048
        ):
            warnings.append({
                "code": "text_encoder_16bit",
                "message": f"{model['name']} is a {model['weights_mb']:.0f} MB {model['dtypes'][0]} text encoder; an FP8 version, if one exists, is about half the size",
            })

    video = [latent for latent in latents if isinstance(_frames(latent), int) and _frames(latent) > 1]
    if video:
        frames = max(_frames(latent) for latent in video)
        if any(node["type"] == "VAEDecode" for node in nodes):
            warnings.append({
                "code": "vae_decode_untiled",
                "message": f"VAEDecode decodes {frames} frames at once; VAE decode is a common VRAM peak for video, VAEDecodeTiled lowers it",
            })
        for node in nodes:
            for key in ("enable_vae_tiling", "tiled_vae"):
                if node["widgets"].get(key) is False:
                    warnings.append({
                        "code": "vae_tiling_off",
                        "message": f"{node['type']}: {key} is off for {frames} frames; enabling it lowers the VAE peak",
                    })
    return warnings


def _canonical(value: Any) -> Any:
    # The UI format saves cfg 1, the API format 1.0: the same setting.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def config_hash(models: list[dict[str, Any]], settings: dict[str, list[dict[str, Any]]]) -> str:
    """Models + settings, independent of node order and of UI vs API format."""
    entries = settings["latents"] + [
        {key: value for key, value in sampler.items() if key not in STEP_RANGE_KEYS} for sampler in settings["samplers"]
    ]
    return _short_hash([
        sorted(model["name"] for model in models),
        sorted(json.dumps(_canonical(entry), sort_keys=True) for entry in entries),
    ])


def analyze_workflow(workflow_path: str | Path, models_dir: str | Path | None = None) -> dict[str, Any]:
    """Offline analysis of a workflow file: models, settings, hashes, warnings. Nothing runs on the GPU."""
    path = Path(workflow_path)
    workflow = json.loads(path.read_text(encoding="utf-8"))
    return analyze_workflow_data(workflow, path.name, Path(models_dir) if models_dir else find_models_dir(path))


def analyze_workflow_data(workflow: dict[str, Any], name: str, models_dir: Path | None = None) -> dict[str, Any]:
    """analyze_workflow for an already loaded workflow, e.g. a prompt from ComfyUI's /history."""
    nodes = workflow_nodes(workflow)
    models = find_models(nodes)
    for model in models:
        resolved = resolve_model(model["name"], models_dir) if models_dir else None
        model["path"] = None
        if resolved:
            model.update(model_weights(resolved), name=model["name"])

    settings = find_settings(nodes)
    found = [model for model in models if model["path"]]
    # Without a models folder there is nothing to look models up in.
    warnings = [
        warning for warning in _warnings(nodes, models, settings["latents"])
        if models_dir or warning["code"] != "model_missing"
    ]
    return {
        "name": name,
        "format": "ui" if "nodes" in workflow else "api",
        "models_dir": str(models_dir) if models_dir else None,
        "node_count": len(nodes),
        # Same graph_hash = same workflow (node types, values ignored).
        # Built from node types, not wiring: two graphs with the same nodes
        # wired differently get the same graph_hash.
        "graph_hash": _short_hash(sorted(node["type"] for node in nodes)),
        # Same config_hash = same models and settings: what a VRAM profile is valid for.
        "config_hash": config_hash(models, settings),
        "models": models,
        "weights_mb": {
            "largest": max((model["weights_mb"] for model in found), default=0.0),
            "all": round(float(sum(model["weights_mb"] for model in found)), 1),
        },
        **settings,
        "warnings": warnings,
    }


def analyze_folder(folder: str | Path, models_dir: str | Path | None = None) -> dict[str, Any]:
    """analyze_workflow for every .json under a folder; other JSON files are listed as skipped."""
    workflows = []
    skipped = []
    for path in sorted(Path(folder).rglob("*.json")):
        try:
            workflows.append({**analyze_workflow(path, models_dir), "name": str(path.relative_to(folder))})
        except (ValueError, OSError, AttributeError, TypeError) as exc:
            skipped.append({"name": str(path.relative_to(folder)), "reason": str(exc)})
    return {"folder": str(folder), "workflows": workflows, "skipped": skipped}
