import json
import os
import zipfile

from vramsuite.core.bundle import REDACTED, anonymize, create_bundle


def test_anonymize_masks_identifying_values(monkeypatch):
    monkeypatch.setenv("USERNAME", "alice")
    monkeypatch.setenv("USER", "alice")
    home = os.path.expanduser("~")
    data = {
        "device": {"nvml": {"uuid": "GPU-1234", "name": "RTX"}},
        "environment": {"python_executable": r"D:\projects\x\python.exe"},
        "stack": {"cublas": {"library_path": home + r"\venv\cublas64_13.dll"}},
        "note": "ComfyUI on 192.168.1.20 by alice",
        "runs": [{"peak_mb": 8318.0}],
    }
    counter = {}
    clean = anonymize(data, counter)

    assert clean["device"]["nvml"] == {"uuid": REDACTED, "name": "RTX"}
    assert clean["environment"]["python_executable"] == REDACTED
    assert clean["stack"]["cublas"]["library_path"] == r"~\venv\cublas64_13.dll"
    assert clean["note"] == "ComfyUI on <ip> by <user>"
    assert clean["runs"] == [{"peak_mb": 8318.0}]
    assert counter == {"uuid": 1, "python_executable": 1, "masked_strings": 2}


def test_bundle_zip(tmp_path):
    card = {"schema": "vramcard@0.1.0-alpha", "device": {"nvml": {"uuid": "GPU-1"}}}
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "abc.json").write_text(json.dumps({"config_hash": "abc", "runs": []}))

    manifest = create_bundle(tmp_path / "b.zip", card, tmp_path / "profiles")
    with zipfile.ZipFile(tmp_path / "b.zip") as archive:
        names = sorted(archive.namelist())
        packed = json.loads(archive.read("system.vramcard.json"))
    assert names == ["manifest.json", "profiles/abc.json", "system.vramcard.json", "validation.json"]
    assert packed["device"]["nvml"]["uuid"] == REDACTED
    # The card above is incomplete on purpose: the bundle still records that it is invalid.
    assert manifest["card_valid"] is False
