# Workflow analysis

`vramsuite analyze-workflow` reads a ComfyUI workflow file and reports what it will do, without running anything on the GPU:

- which model files it loads, and how big they are
- its generation settings: resolution, frame count, batch, steps, sampler
- two hashes that identify the workflow and its exact configuration
- risky patterns that commonly cost VRAM

To check whether the workflow fits on this GPU, use [`estimate --workflow`](estimate.md#comfyui-workflows).

## Usage

One workflow:

```bash
uv run vramsuite analyze-workflow "ComfyUI\user\default\workflows\my workflow.json"
```

A whole folder (every `.json` inside, recursively), one summary row per workflow:

```bash
uv run vramsuite analyze-workflow ComfyUI\user\default\workflows
```

Options:

- `--models-dir` — the ComfyUI models folder. By default it is found above the workflow file.
- `--json` — print the analysis as JSON.

## Example

```text
   WAN stable work.json (ui format, 17 nodes)
| Graph hash    | e843a8b1edd8                 |
| Config hash   | 6b47bef20853                 |
| Weights MB    | 10839.8 largest, 28009.7 all |

| File                                  | Node           |      MB | Dtypes    |
| wan2.2_i2v_low_noise_14B_Q4_K_S.gguf  | UnetLoaderGGUF |  8342.9 | file size |
| wan2.2_i2v_high_noise_14B_Q4_K_S.gguf | UnetLoaderGGUF |  8342.9 | file size |
| Wan2.1_VAE.pth                        | VAELoader      |   484.1 | file size |
| umt5_xxl_fp16.safetensors             | CLIPLoader     | 10839.8 | F16, U8   |
| rife47.pth                            | RIFE VFI       |         | not found |

| WanImageToVideo  | 768x1344, 101 frames, batch 1 |
| KSamplerAdvanced | 6 steps, cfg 1, euler, simple |
| KSamplerAdvanced | 6 steps, cfg 1, euler, simple |

Warning: umt5_xxl_fp16.safetensors is a 10840 MB F16 text encoder; an FP8 version, if one exists, is about half the size
Warning: rife47.pth (RIFE VFI) was not found in the models folder
```

## How it reads a workflow

Both formats ComfyUI saves are supported: the UI format from the workflows folder and the API format from "Export (API)". Subgraphs are included.

Skipped nodes:

- muted and bypassed nodes, because they never run
- frontend-only nodes (`Note`, `Reroute`, `PrimitiveNode`, `SetNode` / `GetNode`), because they never reach the backend

Nothing is matched against a list of known node names:

- **Models** are found by file extension in node values: `.safetensors`, `.gguf`, `.pth`, `.ckpt`, ...
- **Settings** are found by widget name:
  - a node with `width` and `height` whose type mentions latent or video is a latent (with `length` or `num_frames` for video)
  - a node with `steps` is a sampler

  Custom nodes such as WanVideoWrapper are covered the same way as core nodes.

In the UI format, widget values are stored by position and widget names come from the node's inputs. The extra value after a seed (`fixed`, `randomize`, ...) is the frontend's "control after generate" and is skipped.

### Linked values

A setting can come from another node instead of the widget. Example: width from `GetImageSize` of the input image.

- If that node is a primitive (`PrimitiveInt`, `PrimitiveNode`), its value is used.
- Otherwise the value is only known at run time. It is shown as `?` with its source (`linked from GetImageSize`) instead of guessing.

## Hashes

```text
graph_hash    node types of the workflow, values ignored: "the same workflow"
config_hash   model files + settings: "the same workflow with the same settings"
```

A VRAM profile measured for one `config_hash` is valid for every run with that hash. Changing the resolution, frame count or a model changes `config_hash` but keeps `graph_hash`.

The graph hash is built from the node types, not the wiring. Two workflows with the same nodes wired differently get the same `graph_hash`.

## Warnings

```text
model_missing         a model file is not in the models folder (custom nodes like RIFE keep weights in custom_nodes)
text_encoder_16bit    a text encoder over 2 GB in F16 / BF16 / F32; FP8 versions are about half the size
vae_decode_untiled    VAEDecode on a video latent: decoding all frames at once is a common VRAM peak
vae_tiling_off        a node's enable_vae_tiling / tiled_vae is off for a video
```

## JSON fields

```text
name, format (ui | api), models_dir, node_count
graph_hash, config_hash
models        list: name, node_type, path (null if not found), weights_mb, dtypes
weights_mb    largest (one model), all (sum of found models)
latents       list: node_type, width, height, length / num_frames, batch_size
samplers      list: node_type, steps, cfg, sampler_name, scheduler, denoise
warnings      list: code, message
```

A folder prints `folder`, `workflows` (one analysis each, `name` relative to the folder) and `skipped` (JSON files that are not workflows, with the reason).

## Python API

```python
from vramsuite.core.workflow import analyze_folder, analyze_workflow

info = analyze_workflow("workflow.json")
print(info["config_hash"], info["latents"], [w["code"] for w in info["warnings"]])
```
