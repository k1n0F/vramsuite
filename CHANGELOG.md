# Changelog

All notable changes to VRAM Suite. Versions follow [Semantic Versioning](https://semver.org/); while in 0.x, minor versions may change the CLI and file formats.

## [0.5.0] - 2026-11-02

The first public release: installable from PyPI and as a Windows exe.

### Added

- `pip install vramsuite`. Only the NVIDIA driver is needed: no PyTorch, no CUDA Toolkit.
- `pip install "vramsuite[compute]"` installs cuBLAS / cuBLASLt from NVIDIA's wheels for `doctor --compute-check`.
- `vramsuite.exe` for Windows, built by GitHub Actions. Double-clicking it runs `doctor` and waits for Enter.
- `vramsuite bundle` writes an anonymized zip to share: the card, its validation, workflow profiles and a manifest. GPU UUID and the Python path are removed; the home folder, user name and IP addresses are masked.
- `vramsuite --version`, and `python -m vramsuite`.

### Changed

- `pydantic` is no longer a dependency; the runtime dependencies are `typer` and `rich`.
- README links and images use absolute URLs, so they also work on PyPI.

## [0.4.0] - 2026-11-02

### Added

- Suggestions with numbers at the end of `estimate --workflow`:
  - an FP8 text encoder found in the models folder, with the MB it saves
  - the largest resolution or frame count that fits
  - the offloading cost over the steps the samplers actually run
  - the cost of reloading models between stages, with a RAM check
- Confidence of the estimate (`high`, `medium` or `low`), based on where the activations came from.
- Native system RAM reading (`GlobalMemoryStatusEx`, `/proc/meminfo`).

### Changed

- `estimate_comfy_workflow` moved to `vramsuite.core.comfy`, so imports between modules only go one way.

## [0.3.0] - 2026-11-02

### Added

- `vramsuite profile` records the peak VRAM and run time of real ComfyUI runs, passively: it reads ComfyUI's queue and history and samples VRAM per process. Profiles are saved per `config_hash`.
- `estimate --workflow` uses a matching profile for the activations automatically.

## [0.2.0] - 2026-11-02

### Added

- `vramsuite analyze-workflow` for a ComfyUI workflow file or a folder of them:
  - models, resolution, frames, batch, steps and sampler
  - `graph_hash` and `config_hash`
  - risky patterns: 16-bit text encoders, untiled VAE decode of video, missing models
- CUDA driver API backend (`nvcuda.dll` / `libcuda.so`). The probe, the transfer benchmark and device attributes no longer need the CUDA Toolkit.
- A Capabilities table in `doctor`.

## [0.1.0] - 2026-11-02

### Added

- `doctor`: the native GPU profile and the `.vramcard` file.
- `doctor --probe`, with OOM risk.
- `doctor --compute-check`: cuBLAS / cuBLASLt FP32 and FP8 checks.
- `inspect`: exact weight sizes from `.safetensors` headers.
- `processes`: VRAM per process, observed from outside.
- `transfer`: real PCIe bandwidth, pinned and pageable.
- `estimate`: does it fit, and what offloading costs per step. It works for a list of models or a ComfyUI workflow.
- `compare` and `validate` for `.vramcard` files.
