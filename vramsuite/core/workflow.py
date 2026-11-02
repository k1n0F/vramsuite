"""
ComfyUI workflow analysis: which model files does a workflow load, and do they fit?

Both formats ComfyUI saves are read:
- UI format (workflows folder, "Save"): {"nodes": [{"type", "mode", "widgets_values"}], "definitions": {"subgraphs": [...]}}
- API format ("Export (API)"): {"<id>": {"class_type", "inputs"}}

Models are recognised by file extension in node values, not by a table of
loader node names, so custom loaders (GGUF, frame interpolation, ...) are found too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from vramsuite.core.estimate import estimate_workflow

MODEL_SUFFIXES = (".safetensors", ".sft", ".gguf", ".pth", ".pt", ".ckpt", ".bin")
# LiteGraph node modes: 2 = muted, 4 = bypassed. Such nodes never load anything.
SKIPPED_MODES = {2, 4}


def _ui_nodes(workflow: dict[str, Any]) -> Iterator[tuple[str, list[Any]]]:
    subgraphs = (workflow.get("definitions") or {}).get("subgraphs") or []
    nodes = [*workflow["nodes"], *(node for graph in subgraphs for node in graph.get("nodes") or [])]
    for node in nodes:
        if node.get("mode") in SKIPPED_MODES:
            continue
        values = node.get("widgets_values") or []
        # Some custom nodes save widgets as a dict.
        yield node.get("type", "?"), list(values.values()) if isinstance(values, dict) else values


def _api_nodes(workflow: dict[str, Any]) -> Iterator[tuple[str, list[Any]]]:
    for node in workflow.values():
        if isinstance(node, dict) and "class_type" in node:
            yield node["class_type"], list((node.get("inputs") or {}).values())


def find_models(workflow: dict[str, Any]) -> list[dict[str, str]]:
    """Model file names a workflow references, once each, in node order."""
    nodes = _ui_nodes(workflow) if "nodes" in workflow else _api_nodes(workflow)
    found: dict[str, str] = {}
    for node_type, values in nodes:
        for value in values:
            if isinstance(value, str) and value.lower().endswith(MODEL_SUFFIXES):
                found.setdefault(value, node_type)
    return [{"name": name, "node_type": node_type} for name, node_type in found.items()]


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
    # ponytail: no extra_model_paths.yaml support, the recursive search covers models moved inside models_dir.
    return next((path for path in models_dir.rglob(relative.name) if path.is_file()), None)


def estimate_comfy_workflow(
        workflow_path: str | Path,
        models_dir: str | Path | None = None,
        **estimate_kwargs: Any,
) -> dict[str, Any]:
    """
    Estimate VRAM for a ComfyUI workflow file.

    ComfyUI loads models stage by stage (text encoder, then the diffusion
    model, then the VAE) and unloads what does not fit, so the plan is made
    for the largest model. The sum of all models is reported as well.
    """
    path = Path(workflow_path)
    workflow = json.loads(path.read_text(encoding="utf-8"))
    models_dir = Path(models_dir) if models_dir else find_models_dir(path)
    if models_dir is None:
        raise RuntimeError(f"Cannot find the ComfyUI models folder above {path.name}; pass --models-dir")

    models = find_models(workflow)
    for model in models:
        resolved = resolve_model(model["name"], models_dir)
        model["path"] = str(resolved) if resolved else None

    found = [model for model in models if model["path"]]
    if not found:
        raise RuntimeError(f"None of the {len(models)} models of {path.name} were found in {models_dir}")

    info = estimate_workflow([model["path"] for model in found], largest_only=True, **estimate_kwargs)
    for model, node in zip(info["models"], found):
        model["node_type"] = node["node_type"]
    info["workflow"] = {
        "name": path.name,
        "models_dir": str(models_dir),
        "missing": [model["name"] for model in models if not model["path"]],
    }
    return info
