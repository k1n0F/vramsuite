from vramsuite.core.workflow import find_models, find_settings, workflow_nodes

UI_WORKFLOW = {
    "nodes": [
        {"id": 1, "type": "UNETLoader", "mode": 0, "widgets_values": ["wan.safetensors", "default"]},
        {"id": 2, "type": "VAELoader", "mode": 4, "widgets_values": ["bypassed_vae.pth"]},
        {"id": 3, "type": "Note", "mode": 0, "widgets_values": ["see model.safetensors"]},
        {"id": 4, "type": "PrimitiveNode", "mode": 0, "widgets_values": [832]},
        {
            "id": 5,
            "type": "WanImageToVideo",
            "mode": 0,
            "inputs": [
                {"name": "vae", "link": None},
                {"name": "width", "link": 10, "widget": {"name": "width"}},
                {"name": "height", "link": 11, "widget": {"name": "height"}},
                {"name": "length", "link": None, "widget": {"name": "length"}},
            ],
            # width/height are linked: their values here are stale.
            "widgets_values": [480, 720, 81],
        },
        {
            "id": 6,
            "type": "KSamplerAdvanced",
            "mode": 0,
            "inputs": [{"name": key, "widget": {"name": key}} for key in ("add_noise", "noise_seed", "steps", "cfg")],
            # "randomize" is control_after_generate, not a widget of its own.
            "widgets_values": ["enable", 42, "randomize", 6, 1.0],
        },
    ],
    "links": [[10, 4, 0, 5, 1, "INT"], [11, 99, 0, 5, 2, "INT"]],
    "definitions": {"subgraphs": [{"id": "sub", "nodes": [{"type": "UnetLoaderGGUF", "widgets_values": ["q4.gguf"]}]}]},
}


def test_ui_format():
    nodes = workflow_nodes(UI_WORKFLOW)
    assert find_models(nodes) == [
        {"name": "wan.safetensors", "node_type": "UNETLoader"},
        {"name": "q4.gguf", "node_type": "UnetLoaderGGUF"},
    ]
    settings = find_settings(nodes)
    # width from the primitive, height linked to an unknown node, length from the widget.
    assert settings["latents"] == [{"node_type": "WanImageToVideo", "width": 832, "height": "linked from ?", "length": 81}]
    assert settings["samplers"] == [{"node_type": "KSamplerAdvanced", "steps": 6, "cfg": 1.0}]


def test_api_format():
    workflow = {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5.safetensors", "type": "wan"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "sub\\lora.safetensors"}},
        "3": {"class_type": "PrimitiveInt", "inputs": {"value": 1280}},
        "4": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": ["3", 0], "height": ["5", 1], "length": 97}},
        "5": {"class_type": "GetImageSize", "inputs": {"image": ["9", 0]}},
    }
    nodes = workflow_nodes(workflow)
    assert [model["name"] for model in find_models(nodes)] == ["umt5.safetensors", "sub\\lora.safetensors"]
    assert find_settings(nodes)["latents"] == [
        {"node_type": "EmptyLTXVLatentVideo", "width": 1280, "height": "linked from GetImageSize", "length": 97}
    ]


def test_warnings():
    from vramsuite.core.workflow import _warnings

    nodes = [{"type": "VAEDecode", "widgets": {}, "values": []}]
    models = [
        {"name": "umt5_fp16.safetensors", "node_type": "CLIPLoader", "path": "x", "weights_mb": 10840.0, "dtypes": ["F16"]},
        {"name": "umt5_fp8.safetensors", "node_type": "CLIPLoader", "path": "x", "weights_mb": 6424.0, "dtypes": ["F8_E4M3"]},
        {"name": "rife.pth", "node_type": "RIFE VFI", "path": None},
    ]
    video = [{"node_type": "WanImageToVideo", "width": 832, "height": 480, "length": 81}]
    assert [w["code"] for w in _warnings(nodes, models, video)] == [
        "text_encoder_16bit", "model_missing", "vae_decode_untiled",
    ]
    image = [{"node_type": "EmptyLatentImage", "width": 1024, "height": 1024}]
    assert [w["code"] for w in _warnings(nodes, models[1:2], image)] == []
