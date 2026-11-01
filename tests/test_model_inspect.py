import json
import struct

import pytest

from vramsuite.core.model_inspect import ModelInspectError, inspect_model


def _write_safetensors(path, tensors, metadata=None):
    """tensors: {name: (dtype, shape, nbytes)} -> minimal valid safetensors file."""
    header = {"__metadata__": metadata or {}}
    offset = 0
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes

    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
    return path


def test_sizes_by_dtype_and_component(tmp_path):
    model = _write_safetensors(
        tmp_path / "m.safetensors",
        {
            "model.block.weight": ("F16", [4, 8], 64),
            "model.block.bias": ("F32", [8], 32),
            "vae.conv.weight": ("F8_E4M3", [16], 16),
        },
        metadata={"format": "pt"},
    )

    info = inspect_model(model)

    assert info["tensor_count"] == 3
    assert info["total_params"] == 32 + 8 + 16
    assert info["total_bytes"] == 112
    assert info["by_dtype"]["F16"] == {"tensors": 1, "params": 32, "bytes": 64, "mb": 0.0}
    assert list(info["components_mb"]) == ["model", "vae"]  # sorted by size
    assert info["metadata"] == {"format": "pt"}


def test_directory_of_shards_is_summed(tmp_path):
    _write_safetensors(tmp_path / "a.safetensors", {"x": ("BF16", [10], 20)})
    _write_safetensors(tmp_path / "b.safetensors", {"y": ("BF16", [5], 10)})

    info = inspect_model(tmp_path)

    assert info["files"] == ["a.safetensors", "b.safetensors"]
    assert info["total_bytes"] == 30


def test_rejects_pickles_and_corrupt_headers(tmp_path):
    with pytest.raises(ModelInspectError, match="pickles"):
        inspect_model(tmp_path / "model.ckpt")

    corrupt = tmp_path / "bad.safetensors"
    corrupt.write_bytes(struct.pack("<Q", 10**12) + b"{}")
    with pytest.raises(ModelInspectError, match="invalid header length"):
        inspect_model(corrupt)

    outside = _write_safetensors(tmp_path / "out.safetensors", {"x": ("F16", [4], 8)})
    raw = outside.read_bytes()
    outside.write_bytes(raw[:-4])  # tensor data now runs past end of file
    with pytest.raises(ModelInspectError, match="outside the file"):
        inspect_model(outside)
