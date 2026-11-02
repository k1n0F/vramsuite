from vramsuite.core.workflow import find_models


def test_ui_format_skips_muted_and_reads_subgraphs():
    workflow = {
        "nodes": [
            {"type": "UNETLoader", "mode": 0, "widgets_values": ["wan.safetensors", "default"]},
            {"type": "VAELoader", "mode": 4, "widgets_values": ["bypassed_vae.pth"]},
            {"type": "KSampler", "mode": 0, "widgets_values": [42, "fixed", 20]},
            {"type": "UNETLoader", "mode": 0, "widgets_values": ["wan.safetensors", "default"]},
        ],
        "definitions": {"subgraphs": [{"nodes": [{"type": "UnetLoaderGGUF", "widgets_values": ["q4.gguf"]}]}]},
    }
    assert find_models(workflow) == [
        {"name": "wan.safetensors", "node_type": "UNETLoader"},
        {"name": "q4.gguf", "node_type": "UnetLoaderGGUF"},
    ]


def test_api_format():
    workflow = {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5.safetensors", "type": "wan"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "sub\\lora.safetensors"}},
    }
    assert [model["name"] for model in find_models(workflow)] == ["umt5.safetensors", "sub\\lora.safetensors"]
