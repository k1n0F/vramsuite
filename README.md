# VRAM Suite

VRAM Suite is an experimental Python framework for GPU memory profiling in local AI workflows.

The goal is simple:

```text
less guessing
less random CUDA OOM
more real data about VRAM usage
```

Right now the project is in early pre-alpha.

## What is this?

Local AI tools like ComfyUI, Diffusers and custom PyTorch scripts often fail because VRAM usage is hard to predict.

A GPU can have 16 GB of VRAM, but the real usable amount depends on many things:

* PyTorch
* CUDA
* drivers
* allocator settings
* attention backend
* model size
* resolution
* batch size
* video frames
* VAE decode
* offload settings
* custom nodes

VRAM Suite is my attempt to make this behavior easier to measure, profile and understand.

## Main idea

VRAM Suite will use a few profile files:

```text
system.vramcard.json
  describes the GPU and runtime memory capability

workflow\_profile.json
  describes how much memory a workflow used

model\_profile.json
  future profile for model files like .safetensors
```

Together they should help answer:

```text
Can this GPU run this model with this workflow?
```

## Current status

This repository currently contains the first project skeleton.

Working command:

```bash
uv run vramsuite doctor
```

Current output:

```text
VRAM Suite Doctor
Status: pre-alpha / project skeleton
Next target: v0.1-alpha - system.vramcard generation
```

## Planned features

* GPU and runtime fingerprint
* `.vramcard` generation
* safe VRAM probe
* VRAM reader
* workflow profiler
* workflow analyzer
* profile hashes
* memory risk advisor
* ComfyUI adapter
* Diffusers adapter
* `.safetensors` model inspector
* future profile database

## Project structure

```text
vramsuite/
  core/
  backends/
  workflow/
  adapters/
  cli/
  schemas/

docs/
examples/
tests/
```

## Philosophy

VRAM Suite should help, not block.

The default behavior should be:

```text
observe
profile
warn
suggest
```

Not every workflow can be predicted perfectly. The point is to reduce chaos and give better warnings before something crashes.

## Development

This project uses `uv`.

Install dependencies:

```bash
uv sync
```

Run the CLI:

```bash
uv run vramsuite doctor
```

Run tests:

```bash
uv run pytest
```

## Roadmap

```text
v0.0.1  project skeleton
v0.1    doctor + system.vramcard
v0.2    safe VRAM probe
v0.3    VRAM reader
v0.4    workflow analyzer
v0.5    workflow profiler
v0.6    profile hashes
v0.7    advisor
v1.0    first usable CUDA MVP
v1.1+   model inspector
v2.0    profile database
```

## Status

Very early project.

Most files are placeholders right now. The first real milestone is `v0.1-alpha`.

## Author

Created by **k1n0F**.

