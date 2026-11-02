# Model inspection

`vramsuite inspect` reads a model file and reports its exact weight sizes without loading it.

It answers questions like:

- How many MB of weights will this model put on the GPU?
- Which dtypes does it use: BF16, FP16, FP8, packed FP4?
- Which part of a checkpoint is the largest: UNet, VAE, text encoder?

This is the first input for workflow memory estimation: weights are the part of VRAM usage that can be known exactly before anything runs.

## Usage

Inspect one file:

```bash
uv run vramsuite inspect models\vae\flux2-vae_small.safetensors
```

Inspect a directory of shards (all `*.safetensors` in it are summed):

```bash
uv run vramsuite inspect path\to\model-dir
```

Print the result as JSON:

```bash
uv run vramsuite inspect model.safetensors --json
```

## How it works

A `.safetensors` file has a simple layout:

```text
[8 bytes]   header length (little-endian u64)
[N bytes]   JSON header: { tensor name: { dtype, shape, data_offsets }, __metadata__ }
[rest]      raw tensor bytes
```

VRAM Suite reads only the first two parts. Even for a 20 GB model this takes milliseconds.

Sizes are taken from `data_offsets` (`end - start`), not from `dtype × shape`. That makes them exact on-disk bytes for any dtype, including FP8 and packed FP4, without a dtype size table.

## Safety

Model files come from the internet, so the reader is defensive:

- Only `.safetensors` is supported.
- `.ckpt`, `.pt` and `.bin` are rejected: they are Python pickles and can run code when loaded.
- A header larger than 100 MB is rejected as corrupt or hostile.
- Every tensor's `data_offsets` must point inside the file.
- PyTorch is not used and no code from the file is executed.

Example error:

```text
Cannot inspect model: model.ckpt: only .safetensors is supported (.ckpt/.pt/.bin are pickles and can run code when loaded)
```

## Example output

```text
                                 Model
┌────────────┬───────────────────────────────────────────┐
│ Field      │ Value                                     │
├────────────┼───────────────────────────────────────────┤
│ Path       │ models\vae\flux2-vae_small.safetensors    │
│ Files      │ 1                                         │
│ Tensors    │ 251                                       │
│ Parameters │ 0.062 B                                   │
│ Weights MB │ 237.9                                     │
└────────────┴───────────────────────────────────────────┘
           Weights by dtype
┌───────┬─────────┬──────────┬───────┐
│ Dtype │ Tensors │ Params   │ MB    │
├───────┼─────────┼──────────┼───────┤
│ F32   │ 250     │ 62373347 │ 237.9 │
│ I64   │ 1       │ 1        │ 0.0   │
└───────┴─────────┴──────────┴───────┘
Weights by component (first
       name segment)
┌─────────────────┬───────┐
│ Component       │ MB    │
├─────────────────┼───────┤
│ encoder         │ 131.3 │
│ decoder         │ 106.6 │
│ quant_conv      │ 0.0   │
│ post_quant_conv │ 0.0   │
│ bn              │ 0.0   │
└─────────────────┴───────┘
```

### Components

Components are grouped by the first segment of the tensor name. For example, an SDXL checkpoint splits into:

- `model` — UNet
- `first_stage_model` — VAE
- `conditioner` — text encoders

The 12 largest components are shown. The rest are counted in `other_components`.

### Metadata

`__metadata__` from the header (training info, quantization config, etc.) is returned in the `metadata` field of the JSON output. With shards, metadata from all files is merged.

## JSON fields

```text
path              input path
format            "safetensors"
files             file names that were read
tensor_count      number of tensors
total_params      sum of shape products
total_bytes       exact weight bytes
total_mb          total_bytes in MB
by_dtype          { dtype: { tensors, params, bytes, mb } }, largest first
components_mb     { component: mb }, top 12
other_components  number of components not shown
metadata          merged __metadata__
```

## Python API

```python
from vramsuite.core.model_inspect import inspect_model

info = inspect_model("model.safetensors")

print(info["total_mb"])
print(info["by_dtype"])
```

`inspect_model` raises `ModelInspectError` for unsupported or invalid files.
