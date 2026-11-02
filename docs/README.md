# VRAM Suite Documentation

This folder documents each VRAM Suite feature in detail.

The main [README](../README.md) is the overview. These pages go deeper: what every command does, where its numbers come from, and how to read them.

## Pages

- [Device profile](device-profile.md) — `doctor`, the `device` section of `.vramcard`, native compute check
- [Allocation probe and OOM risk](probe-and-risk.md) — `doctor --probe`, `--estimate-mb`, Windows vs WSL
- [Model inspection](model-inspection.md) — `inspect`, exact weight sizes from `.safetensors` headers
- [Process tracking](process-tracking.md) — `processes`, per-process VRAM observed from outside
- [Workflow memory estimate](estimate.md) — `estimate`, will models fit and what offloading costs
- [Transfer benchmark](transfer.md) — `transfer`, real RAM <-> VRAM bandwidth over PCIe
- [`.vramcard` format](vramcard.md) — every section of the profile, and `compare`
- [Native backends](native-backends.md) — how VRAM Suite talks to NVIDIA libraries without PyTorch

## Commands at a glance

```bash
uv run vramsuite doctor                  # diagnostics + system.vramcard.json
uv run vramsuite doctor --compute-check  # + cuBLAS / cuBLASLt FP32, FP8 GEMM checks
uv run vramsuite inspect model.safetensors
uv run vramsuite processes               # VRAM per process, right now
uv run vramsuite processes --watch 60    # peak VRAM per process over 60 seconds
uv run vramsuite transfer                # RAM <-> VRAM bandwidth, pinned and pageable
uv run vramsuite estimate model.safetensors --activations-mb 3000  # fits? offload cost?
uv run vramsuite compare a.vramcard.json b.vramcard.json
```

Every command supports `--json` for machine-readable output.

## Design rules

These rules apply to every page:

- Everything is native: NVML, CUDA runtime, cuBLAS and cuBLASLt are called through `ctypes`.
- PyTorch and Triton are never imported. PyTorch is only detected from package metadata.
- Nothing heavy runs by default. Probes and compute checks need an explicit flag.
- Every resource is released, even when a step fails.
- A value that cannot be read is reported as unavailable with a reason, not silently dropped.
