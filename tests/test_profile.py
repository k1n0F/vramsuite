from vramsuite.core.profile import _comfy_process, load_profile, save_run, summarize_profile
from vramsuite.core.workflow import analyze_workflow_data

# The same workflow as the UI saves it and as ComfyUI runs it (API, from /history).
UI = {
    "nodes": [
        {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["sdxl.safetensors"]},
        {"id": 2, "type": "PrimitiveNode", "widgets_values": [1024]},
        {
            "id": 3, "type": "EmptyLatentImage",
            "inputs": [
                {"name": "width", "link": 7, "widget": {"name": "width"}},
                {"name": "height", "link": None, "widget": {"name": "height"}},
                {"name": "batch_size", "link": None, "widget": {"name": "batch_size"}},
            ],
            "widgets_values": [512, 768, 1],
        },
        {
            "id": 4, "type": "KSampler",
            "inputs": [{"name": key, "widget": {"name": key}} for key in ("seed", "steps", "cfg")],
            "widgets_values": [5, "randomize", 30, 6],
        },
    ],
    "links": [[7, 2, 0, 3, 0, "INT"]],
}
API = {
    "4": {"class_type": "KSampler", "inputs": {"seed": 99, "steps": 30, "cfg": 6.0, "model": ["1", 0]}},
    "3": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 768, "batch_size": 1}},
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl.safetensors"}},
}


def test_config_hash_is_format_independent():
    ui = analyze_workflow_data(UI, "ui")
    api = analyze_workflow_data(API, "api")
    assert ui["config_hash"] == api["config_hash"]
    # A different resolution is a different config.
    API["3"]["inputs"]["height"] = 1024
    assert analyze_workflow_data(API, "api")["config_hash"] != ui["config_hash"]
    API["3"]["inputs"]["height"] = 768


def test_save_and_summarize(tmp_path):
    analysis = analyze_workflow_data(API, "api")
    for status, peak, seconds in (("success", 9000.0, 12.5), ("success", 9400.0, 8.1), ("error", 15000.0, 3.0)):
        run = {"status": status, "peak_mb": peak, "run_seconds": seconds}
        save_run(tmp_path, {"analysis": analysis, "run": run})

    profile = load_profile(tmp_path, analysis["config_hash"])
    assert len(profile["runs"]) == 3
    # Failed runs do not count.
    assert summarize_profile(profile) == {
        "config_hash": analysis["config_hash"],
        "runs": 2,
        "peak_mb": 9400.0,
        "fastest_seconds": 8.1,
        "slowest_seconds": 12.5,
    }


def test_comfy_process_is_not_the_biggest_wddm_counter():
    # Real Windows snapshot: a protected process and an overlay outweigh ComfyUI.
    processes = [
        {"pid": 2460, "name": None, "command_line": None},
        {"pid": 21908, "name": "NVIDIA Overlay.exe", "command_line": None},
        {"pid": 24500, "name": "python.exe", "command_line": "python.exe main.py --listen"},
    ]
    assert _comfy_process(processes)["pid"] == 24500
    processes[1]["command_line"] = "C:/ComfyUI/python_embeded/python.exe -s ComfyUI/main.py"
    assert _comfy_process(processes)["pid"] == 21908
