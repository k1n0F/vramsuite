"""
Model file inspection without loading weights.

`.safetensors` layout: 8-byte little-endian header length, a JSON header
({name: {"dtype", "shape", "data_offsets": [start, end]}, "__metadata__": {...}}),
then raw tensor bytes. Reading only the header gives exact weight sizes per
dtype in milliseconds, with no torch and no code execution (unlike pickled
.ckpt/.pt files, which are never opened here).
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

# Real headers are a few MB at most; a bigger value means a corrupt or hostile file.
MAX_HEADER_BYTES = 100 * 1024 * 1024
BYTES_PER_MB = 1024 * 1024
TOP_COMPONENTS = 12


class ModelInspectError(ValueError):
    pass


def read_safetensors_header(path: Path) -> dict[str, Any]:
    file_size = path.stat().st_size

    with path.open("rb") as file:
        raw_length = file.read(8)
        if len(raw_length) != 8:
            raise ModelInspectError(f"{path.name}: too small to be a safetensors file")

        (header_length,) = struct.unpack("<Q", raw_length)
        if header_length > MAX_HEADER_BYTES or 8 + header_length > file_size:
            raise ModelInspectError(f"{path.name}: invalid header length {header_length}")

        try:
            header = json.loads(file.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelInspectError(f"{path.name}: header is not valid JSON: {exc}") from exc

    if not isinstance(header, dict):
        raise ModelInspectError(f"{path.name}: header is not a JSON object")

    data_size = file_size - 8 - header_length
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        if not 0 <= start <= end <= data_size:
            raise ModelInspectError(f"{path.name}: tensor {name!r} points outside the file")

    return header


def _safetensors_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
        if not files:
            raise ModelInspectError(f"{path}: no .safetensors files in this directory")
        return files

    if path.suffix.lower() != ".safetensors":
        raise ModelInspectError(
            f"{path.name}: only .safetensors is supported "
            "(.ckpt/.pt/.bin are pickles and can run code when loaded)"
        )
    return [path]


def inspect_model(path: str | Path) -> dict[str, Any]:
    """
    Weight sizes of a .safetensors file or a directory of shards.

    Sizes come from data_offsets, so they are exact on-disk bytes for any dtype
    (including FP8 / packed FP4) without a dtype size table.
    """
    root = Path(path)
    files = _safetensors_files(root)

    total_bytes = 0
    tensor_count = 0
    by_dtype: dict[str, dict[str, int]] = {}
    by_component: dict[str, int] = {}
    metadata: dict[str, Any] = {}

    for file in files:
        header = read_safetensors_header(file)
        metadata.update(header.get("__metadata__") or {})

        for name, info in header.items():
            if name == "__metadata__":
                continue

            start, end = info["data_offsets"]
            size = end - start
            params = math.prod(info["shape"])

            dtype = by_dtype.setdefault(info["dtype"], {"tensors": 0, "params": 0, "bytes": 0})
            dtype["tensors"] += 1
            dtype["params"] += params
            dtype["bytes"] += size

            # First name segment groups e.g. SDXL checkpoints into
            # model (UNet) / first_stage_model (VAE) / conditioner (text encoders).
            component = name.split(".", 1)[0]
            by_component[component] = by_component.get(component, 0) + size

            total_bytes += size
            tensor_count += 1

    top_components = sorted(by_component.items(), key=lambda item: item[1], reverse=True)

    return {
        "path": str(root),
        "format": "safetensors",
        "files": [file.name for file in files],
        "tensor_count": tensor_count,
        "total_params": sum(item["params"] for item in by_dtype.values()),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / BYTES_PER_MB, 1),
        "by_dtype": {
            name: {**item, "mb": round(item["bytes"] / BYTES_PER_MB, 1)}
            for name, item in sorted(by_dtype.items(), key=lambda kv: kv[1]["bytes"], reverse=True)
        },
        "components_mb": {
            name: round(size / BYTES_PER_MB, 1) for name, size in top_components[:TOP_COMPONENTS]
        },
        "other_components": max(len(top_components) - TOP_COMPONENTS, 0),
        "metadata": metadata,
    }
