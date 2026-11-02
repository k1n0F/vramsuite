# Bundle

`vramsuite bundle` packs what someone else needs to understand this machine's results into one zip, without anything that identifies you or your card. Use it to attach to a bug report or to share a benchmark setup.

## Usage

```bash
uv run vramsuite bundle
```

This writes `vramsuite-bundle-<date>.zip` in the current folder.

Options:

- `--output`, `-o` — the zip file to write
- `--card` — use an existing `.vramcard`, for example one made with `doctor --probe`, instead of creating a fresh one (without the probe)
- `--profiles-dir` — workflow profiles to include (default `profiles`, see [Workflow profiles](profiles.md))

## What is inside

```text
system.vramcard.json   the .vramcard of this machine (see .vramcard format)
validation.json        that card checked by `vramsuite validate`
profiles/*.json        recorded ComfyUI runs: models, settings, peak VRAM, run time
manifest.json          VRAM Suite version, date, file list, what was removed
```

Profiles contain model file names and settings (resolution, frames, steps), but no prompts, seeds or images.

## What is removed

```text
uuid, serial          identify one physical GPU          -> <redacted>
python_executable     shows folder names on this machine -> <redacted>
home folder           in any path                        -> ~
user name             anywhere in a string               -> <user>
IPv4 addresses        anywhere in a string               -> <ip>
```

The paths of CUDA Toolkit libraries (`C:\Program Files\NVIDIA GPU Computing Toolkit\...`) are kept: they show which CUDA version was used and are the same on every machine.

The command prints what it removed, and `manifest.json` records it:

```text
Removed: python_executable x1, uuid x1 (home folder -> ~, user name -> <user>, IPv4 addresses -> <ip>).
```

If the card does not pass validation, the bundle is still written, and `validation.json` says why.

## Python API

```python
from vramsuite.core.bundle import anonymize, create_bundle
from vramsuite.core.vramcard import load_vramcard

manifest = create_bundle("bundle.zip", load_vramcard("system.vramcard.json"), profiles_dir="profiles")
print(manifest["files"], manifest["redacted"]["counts"])
```
