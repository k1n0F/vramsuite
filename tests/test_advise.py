from vramsuite.core.advise import _model_key, _round_frames, _size_suggestion, advise
from vramsuite.core.workflow import steps_run


def test_model_key_ignores_precision():
    assert _model_key("umt5_xxl_fp16.safetensors") == _model_key("umt5_xxl_fp8_e4m3fn_scaled.safetensors")
    assert _model_key("umt5_xxl_fp16.safetensors") != _model_key("qwen_3_4b.safetensors")


def test_video_frames_keep_their_form():
    # 81 fits both 4n+1 (WAN) and 8n+1 (LTX): the stricter 8n+1 is valid for both.
    assert _round_frames(80.5, 81) == 73
    assert _round_frames(80.5, 77) == 77  # 4n+1 only
    assert _round_frames(80.5, 30) == 80  # no n*k+1 form


def test_steps_run_with_ranges():
    assert steps_run({"steps": 6, "start_at_step": 0, "end_at_step": 3}) == 3
    assert steps_run({"steps": 6, "start_at_step": 3, "end_at_step": 10000}) == 3
    assert steps_run({"steps": 20, "start_step": 0, "end_step": -1}) == 20
    assert steps_run({"steps": "linked from X"}) is None


def test_size_suggestion():
    info = {"budget_mb": 15000, "weights_mb": 9000, "activations_mb": 8000}
    latent = {"width": 1280, "height": 720, "length": 81}
    # 6000 MB of room for 8000 MB of activations: 0.75 of the pixels x frames.
    size = _size_suggestion(info, latent)
    assert size["code"] == "reduce_size" and size["scale"] == 0.75
    assert "1104x608 at 81 frames or 57 frames at 1280x720" in size["message"]
    info["activations_mb"] = 3000
    assert _size_suggestion(info, latent)["code"] == "max_size"


def test_confidence_follows_activation_source():
    info = {
        "activations_source": "not_set", "activations_mb": 0, "budget_mb": 15000, "weights_mb": 6000,
        "all_weights_mb": 6000, "offload": None,
    }
    analysis = {"models_dir": None, "models": [], "latents": [], "samplers": [], "warnings": []}
    assert advise(info, analysis) == {"confidence": "low", "suggestions": []}
