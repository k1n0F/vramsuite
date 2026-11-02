# Workflow memory estimate

`vramsuite estimate` answers one question before a workflow runs: will these models fit in VRAM, and if not, how much slower will offloading make each step?

It combines what VRAM Suite already measures:

- **weights** — exact, from `.safetensors` headers ([Model inspection](model-inspection.md))
- **available VRAM** — free memory right now, from NVML
- **offloading cost** — real PCIe bandwidth of this machine ([Transfer benchmark](transfer.md))

## Usage

Will this model fit in the VRAM that is free right now?

```bash
uv run vramsuite estimate models\diffusion_models\wan2.2_i2v_low_noise_14B_nvfp4_mixed.safetensors --activations-mb 3000
```

Several models that are loaded at the same time (their weights are summed):

```bash
uv run vramsuite estimate model.safetensors text_encoder.safetensors vae.safetensors --activations-mb 3000
```

Plan for the whole card, as if nothing else was running:

```bash
uv run vramsuite estimate model.safetensors --available-mb 16303 --activations-mb 3000
```

Print the result as JSON:

```bash
uv run vramsuite estimate model.safetensors --activations-mb 3000 --json
```

Only list models that are in VRAM **at the same time**. For example, WAN 2.2 runs its high-noise and low-noise models one after another, so estimate them separately, not together.

`.safetensors` sizes are exact. `.gguf`, `.pth` and `.ckpt` are not parsed (pickles can run code when loaded); their file size is used instead.

## ComfyUI workflows

Pass a workflow file instead of a list of models:

```bash
uv run vramsuite estimate --workflow "ComfyUI\user\default\workflows\my workflow.json" --observed-peak-mb 14745
```

- Both formats are read: the UI format from the workflows folder and the API format from "Export (API)".
- Models are found by file extension in node values (`.safetensors`, `.gguf`, `.pth`, ...), so custom loader nodes work too. Muted and bypassed nodes are skipped.
- Files are looked up in `ComfyUI/models`, found automatically above the workflow file. Pass `--models-dir` otherwise. Models that are not found are listed as warnings; some custom nodes keep weights in their own folder under `custom_nodes`.

ComfyUI loads models stage by stage (text encoder, diffusion model, VAE) and unloads what does not fit, so the plan is made for the **largest model** plus activations. It is conservative when the largest model is the text encoder, whose activations are small. The sum of all models is shown below the table: if it exceeds the budget, ComfyUI reloads models between stages on every run.

```text
                wan22_i2v_nvfp4.json on GPU 0: memory estimate
│ wan2.2_i2v_high_noise_14B_nvfp4_mixed.safetensors  │ 10047.7 │ UNETLoader: F8_E4M3, U8, F16, F32 │
│ wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise...   │  1170.0 │ LoraLoaderModelOnly: F32, I64     │
│ wan2.2_i2v_low_noise_14B_nvfp4_mixed.safetensors   │ 10047.7 │ UNETLoader: F8_E4M3, U8, F16, F32 │
│ wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise...    │  1170.0 │ LoraLoaderModelOnly: F32, I64     │
│ umt5_xxl_fp8_e4m3fn_scaled.safetensors             │  6423.8 │ CLIPLoader: F8_E4M3, F16, U8, F32 │
│ Wan2.1_VAE.pth                                     │   484.1 │ VAELoader: file size              │
│ Weights                                            │ 10047.7 │ largest model, loaded stage by stage │
│ Activations                                        │  4697.3 │ observed_peak                     │
│ Status                                             │         │ fits                              │
Not found in ...\ComfyUI\models: rife49.pth
All models together: 34040.6 MB, more than the budget: ComfyUI unloads and reloads models between stages on every run.
```

## Activations

Weights are exact. Activations (latents, attention buffers, intermediate tensors) depend on resolution, frame count, batch and the attention implementation, so VRAM Suite does not guess them from formulas.

There are two ways to set them.

**Given** — you know or assume a number:

```bash
uv run vramsuite estimate model.safetensors --activations-mb 3000
```

**Calibrated from a real run** — the most accurate way:

1. Run `uv run vramsuite processes --watch 600` in a terminal.
2. Run the workflow once in ComfyUI at the resolution and length you care about.
3. Take `Peak MB` of the ComfyUI worker process.
4. Pass it as `--observed-peak-mb`. VRAM Suite computes `activations = peak - weights`.

```bash
uv run vramsuite estimate model.safetensors --observed-peak-mb 14745
```

The calibrated activations can then be reused: change the model (FP8 instead of FP4, another checkpoint of the same architecture) and see whether it still fits.

If the observed run was already offloading, its peak is capped by the card and underestimates activations. Calibrate on a run that fits.

**From a profile** — for ComfyUI workflows, the easiest way. Record runs with `vramsuite profile` ([Workflow profiles](profiles.md)); `estimate --workflow` finds the profile with the same config hash in `./profiles` (or `--profiles-dir`) and uses its peak. `activations_source` is then `profile`.

Without any of these only weights are counted, and the report warns about it.

## How the estimate works

```text
required = weights + activations
budget   = available - reserve
overflow = required - budget
```

`reserve` (`--reserve-mb`, default 1024 MB) is memory that is always taken: the CUDA context, cuBLAS workspaces, and allocator fragmentation.

The result is one of three statuses:

- `fits` — everything stays in VRAM; `Headroom` is what is left
- `fits with offloading` — the overflow is taken from the weights and streamed from RAM
- `does not fit` — activations alone exceed the budget; offloading weights cannot help. Lower the resolution, frame count or batch.

`Risk level` reuses the OOM risk levels from `doctor` (`low`, `medium`, `high`, `critical`) for the case where everything is kept in VRAM.

## Offloading cost

When the status is `fits with offloading`, VRAM Suite measures host -> device bandwidth (a short 128 MB transfer test) and prices the overflow:

```text
extra seconds per step = offloaded bytes / PCIe bandwidth
```

Both pinned and pageable host memory are shown: pinned is the best case, pageable the typical case for frameworks that do not pin offloaded weights.

This is a lower bound. It assumes every offloaded weight crosses PCIe exactly once per step. Real frameworks may move some weights more than once, or overlap transfers with compute.

To skip the measurement, pass a known bandwidth:

```bash
uv run vramsuite estimate model.safetensors --activations-mb 3000 --transfer-gb-s 28.5
```

## Example output

WAN 2.2 14B FP8 with ComfyUI already holding most of the card (5 GB free):

```text
                                        GPU 0: memory estimate
┌─────────────────────────────────────────────────┬─────────┬─────────────────────────────────────────┐
│ Field                                           │      MB │ Note                                    │
├─────────────────────────────────────────────────┼─────────┼─────────────────────────────────────────┤
│ wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors │ 13632.4 │ F8_E4M3, F16, F32                       │
│ Activations                                     │  3000.0 │ given                                   │
│ Required                                        │ 16632.4 │ weights + activations                   │
│ Available                                       │  5249.0 │ nvml_free                               │
│ Reserve                                         │  1024.0 │ CUDA context, workspaces, fragmentation │
│ Budget                                          │  4225.0 │ available - reserve                     │
│ Overflow                                        │ 12407.4 │ required - budget                       │
│ Status                                          │         │ fits with offloading                    │
│ Risk level                                      │         │ critical (394% of budget if fully       │
│                                                 │         │ resident)                               │
└─────────────────────────────────────────────────┴─────────┴─────────────────────────────────────────┘
   Offloading 12407.4 MB of weights per step
                (measured PCIe)
┌─────────────┬───────┬────────────────────────┐
│ Host memory │  GB/s │ Extra seconds per step │
├─────────────┼───────┼────────────────────────┤
│ pinned      │ 28.49 │                  0.457 │
│ pageable    │ 19.22 │                  0.677 │
└─────────────┴───────┴────────────────────────┘
```

The same family in NVFP4, calibrated from a real run and planned for the whole card:

```text
│ wan2.2_i2v_low_noise_14B_nvfp4_mixed.safetensors │ 10047.7 │ F8_E4M3, U8, F16, F32                   │
│ Activations                                      │  4697.3 │ observed_peak                           │
│ Required                                         │ 14745.0 │ weights + activations                   │
│ Available                                        │ 16303.0 │ given                                   │
│ Budget                                           │ 15279.0 │ available - reserve                     │
│ Headroom                                         │   534.0 │                                         │
│ Status                                           │         │ fits                                    │
│ Risk level                                       │         │ high (97% of budget if fully resident)  │
```

It fits, but with only about 0.5 GB of headroom: a browser or an overlay that takes VRAM during the run can push it into offloading.

## JSON fields

```text
device_index
models               list: name, path, weights_mb, dtypes (+ node_type for workflows)
weights_mb           weights the plan is made for
weights_basis        all (sum of models) | largest (workflows)
all_weights_mb       sum of all models
activations_mb
activations_source   given | observed_peak | profile | not_set
available_mb
available_source     nvml_free | given
reserve_mb
required_mb          weights + activations
budget_mb            available - reserve
headroom_mb          budget - required, if it fits
overflow_mb          required - budget, if it does not
offload_mb           weights streamed from RAM per step
status               fits | offload | does_not_fit
risk                 OOM risk object, same as doctor --estimate-mb
offload              null unless status is offload:
  pinned_gb_s, pageable_gb_s, source (measured | given), error,
  offload_mb, seconds_per_step_pinned, seconds_per_step_pageable
workflow             only with --workflow: name, models_dir, missing (file names), config_hash, profile (runs, peak_mb, fastest / slowest seconds) or null
```

## Python API

```python
from vramsuite.core.estimate import estimate_workflow

info = estimate_workflow(
    ["model.safetensors"],
    observed_peak_mb=14745,
    available_mb=16303,
)

print(info["status"], info["headroom_mb"])

from vramsuite.core.workflow import estimate_comfy_workflow

info = estimate_comfy_workflow("workflow.json", observed_peak_mb=14745, available_mb=16303)
print(info["status"], info["workflow"]["missing"])
```
