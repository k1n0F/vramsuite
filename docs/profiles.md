# Workflow profiles

`vramsuite profile` records what a ComfyUI workflow really uses: the peak VRAM of the ComfyUI process and how long the run took. [`estimate --workflow`](estimate.md#comfyui-workflows) then uses the recorded peak automatically, with no `--observed-peak-mb` typed by hand.

## Usage

Start the profiler, then press **Queue** in ComfyUI:

```bash
uv run vramsuite profile
```

Record a cold and a warm run of the same workflow (press Queue twice):

```bash
uv run vramsuite profile --runs 2
```

Then estimate the workflow file. A profile with the same config hash is picked up from `./profiles`:

```bash
uv run vramsuite estimate --workflow "my workflow.json" --available-mb 16303
```

Options of `profile`:

- `--comfy-url` — the ComfyUI address you open in the browser (default `http://127.0.0.1:8188`)
- `--runs` — how many prompts to record, one after another
- `--profiles-dir` — where profiles are saved (default `profiles` in the current folder; `estimate` has the same option)
- `--no-free` — do not unload ComfyUI's models before the first run (see below)
- `--wait` — seconds to wait for each prompt to start
- `--json` — print each recorded run as JSON

## How it works

The profiler is passive: it never submits anything to ComfyUI.

1. It polls ComfyUI's `/queue` until a prompt is running.
2. While the prompt runs, it samples VRAM per process from outside, like [`processes --watch`](process-tracking.md).
3. When the prompt leaves the queue, it reads `/history`: the exact prompt that ran, whether it succeeded, and its start and end timestamps.
4. The prompt is analyzed like a workflow file ([Workflow analysis](workflow-analysis.md)). Its `config_hash` names the profile file.

The workflow is identified by the prompt that actually ran, not by a file name. Change the resolution, frame count or a model, and you get a different `config_hash` and a separate profile. Seeds and prompts do not change the hash.

`config_hash` is the same for the UI format and the API format of one workflow: node order is ignored, and `cfg 1` equals `cfg 1.0`. A profile recorded from a run therefore matches the saved workflow file, as long as the file was saved with the same settings.

## Why models are unloaded first

ComfyUI's caching allocator (`cudaMallocAsync`) keeps memory after a prompt ends. Seen from outside, the process holds the high-water mark of every earlier prompt, so the peak of a light workflow after a heavy one would be wrong.

Before the first run, the profiler calls ComfyUI's `/free`: the same as the "Unload models" button. The first run then starts cold, and its peak belongs to this workflow. Later runs (`--runs 2` and more) start warm and show how fast the workflow is with models already loaded.

Pass `--no-free` to keep ComfyUI's models loaded. The recorded peak is then an upper bound.

## Which process is ComfyUI

The process is chosen in this order:

1. a process whose command line mentions `comfy`
2. a `python*` process
3. the biggest VRAM user

The biggest user is the last resort on purpose. On Windows, the counters of protected system processes and overlays include shared surfaces, and they can be larger than ComfyUI's own usage. The chosen process and its PID are shown with every run.

## Example

```text
                     Prompt 25037713: success
| Config hash      | 53b8a62ad828                                  |
| Models           | anime\raehoshiIllustXL_v120.safetensors, ...  |
| Size             | 720x1280 (EmptyLatentImage)                   |
| Run seconds      | 6.81                                          |
| Cached nodes     | 0                                             |
| Process          | python.exe (pid 24500)                        |
| Started cold     | yes (models unloaded first)                   |
| VRAM at start MB | 2141.9                                        |
| Peak VRAM MB     | 8318.0                                        |

                     Prompt 1c2701bf: success
| Run seconds      | 5.35                                          |
| Cached nodes     | 5                                             |
| Started cold     | no                                            |
| VRAM at start MB | 8318.0                                        |
| Peak VRAM MB     | 10398.0                                       |
```

The warm run peaks higher than the cold one: the checkpoint stays loaded while the upscaler and the VAE run on top of it. `estimate` uses the highest successful peak:

```text
| Activations | 3781.7 | profile |
Profile 53b8a62ad828: 2 recorded runs, peak 10398.0 MB, run took 5.35-6.81 s
```

## Profile file

`profiles/<config_hash>.json`, one per configuration. Every capture appends a run:

```text
schema        workflow_profile@0.3
config_hash, graph_hash
models        name, node_type
latents, samplers
runs          list:
  measured_at, prompt_id, status (success | error)
  run_seconds     execution_start -> execution_success, from ComfyUI
  cached_nodes    nodes ComfyUI did not re-run (same inputs as before)
  freed_before    models were unloaded before this run
  process, pid    the ComfyUI process
  start_mb        its VRAM when the prompt started
  peak_mb         its peak VRAM during the prompt
  gpu, gpu_total_mb, samples
```

Failed runs are kept in the file but not used by `estimate`.

## Limits

- The run time covers the whole prompt: model loading, sampling, VAE decode and saving. Per-step timing is not recorded yet.
- If the run had to offload weights to RAM, its peak is capped by the card and underestimates the workflow. Record profiles on runs that fit.
- Peak VRAM is sampled every 0.5 s. A spike shorter than that can be missed.
