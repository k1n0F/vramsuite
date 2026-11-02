# Transfer benchmark

`vramsuite transfer` measures how fast data moves between system RAM and VRAM on this machine.

It answers questions like:

- What does the PCIe link actually deliver, not what the spec says?
- How much does offloading cost: how long does it take to move 1 GB of weights?
- How much faster is pinned memory than ordinary (pageable) memory?
- Does a narrowed slot (x8 instead of x16) matter for this workload?

This is the input for offloading estimates: when a model does not fit in VRAM, frameworks like ComfyUI move weights between RAM and VRAM every step, and this speed decides how much slower the step becomes.

## Usage

Run with defaults (256 MB blocks, 5 timed copies per direction):

```bash
uv run vramsuite transfer
```

Bigger blocks and more repeats for steadier numbers:

```bash
uv run vramsuite transfer --size-mb 1024 --repeats 10
```

Pick another GPU (CUDA index):

```bash
uv run vramsuite transfer --gpu 1
```

Print the result as JSON:

```bash
uv run vramsuite transfer --json
```

The benchmark needs `2 × size-mb` of free VRAM and `size-mb` of free RAM for pinned memory. Close heavy GPU apps first for clean numbers: a running generation competes for the same link.

## What is measured

Five copies, each through native `cudaMemcpy`:

- `host_to_device` / `device_to_host` with **pinned** host memory (`cuMemHostAlloc`)
- `host_to_device` / `device_to_host` with **pageable** host memory (an ordinary buffer)
- `device_to_device` — a copy inside VRAM

Each direction runs one warm-up copy, then `--repeats` timed copies. `Best GB/s` is the fastest copy, `Median GB/s` the typical one. GB/s here is 10^9 bytes per second, the same unit as the PCIe spec.

### Pinned vs pageable

Pinned (page-locked) memory cannot be moved by the OS, so the GPU reads it directly over PCIe (DMA).

Pageable memory is ordinary memory. The driver first copies it into its own pinned buffer and only then sends it to the GPU. That extra copy is why pageable transfers are slower.

In PyTorch terms: `tensor.pin_memory()` gives pinned memory, a normal CPU tensor is pageable.

### Device to device

`device_to_device` reads and writes every byte in VRAM, so it is counted twice. The result is close to the real VRAM bandwidth and can be compared with `derived.theoretical_bandwidth_gb_s` from the [device profile](device-profile.md).

## Example output

Real run on an RTX 5080 in a slot that runs at x8:

```text
                   GPU 0 NVIDIA GeForce RTX 5080: transfers (256 MB x 5)
┌──────────────────┬─────────────┬───────────┬─────────────┬───────────┬───────────┬───────┐
│ Direction        │ Host memory │ Best GB/s │ Median GB/s │ % of link │ ms per GB │ Error │
├──────────────────┼─────────────┼───────────┼─────────────┼───────────┼───────────┼───────┤
│ host_to_device   │ pinned      │     28.74 │       28.51 │       91% │      34.8 │       │
│ device_to_host   │ pinned      │     28.46 │       28.37 │       90% │      35.1 │       │
│ host_to_device   │ pageable    │     20.67 │       19.98 │       66% │      48.4 │       │
│ device_to_host   │ pageable    │     19.36 │        18.7 │       61% │      51.7 │       │
│ device_to_device │ -           │    800.58 │      669.67 │           │       1.2 │       │
└──────────────────┴─────────────┴───────────┴─────────────┴───────────┴───────────┴───────┘
PCIe Gen5 x8 (card supports x16): theoretical 31.5 GB/s per direction, 63.0 GB/s at full width
device_to_device counts read + write, i.e. VRAM bandwidth.
```

### Reading the result

- `% of link` — share of the theoretical PCIe bandwidth. 85-92% for pinned memory is normal: the rest is PCIe protocol overhead (packet headers, acknowledgements).
- `ms per GB` — time to move 1 GB. For offloading: a step that streams 8 GB of weights from RAM spends at least `8 × ms per GB` just on transfers.
- `device_to_device` at 800 GB/s against 960 GB/s theoretical shows VRAM is about 30 times faster than the PCIe link. This is why offloading slows inference down so much.

### PCIe theoretical bandwidth

The theoretical value is computed from the link generation and the current width:

```text
Gen  GT/s per lane  encoding   x8        x16
3    8              128/130    7.9 GB/s  15.8 GB/s
4    16             128/130    15.8      31.5
5    32             128/130    31.5      63.0
```

The generation is the maximum the card supports. Idle GPUs drop to a lower generation to save power and switch back up under load, so the generation at rest says nothing about transfer speed. The width does not change with load. `link_gen_after_run` in JSON shows the generation read right after the copies.

## JSON fields

```text
device_index, name, size_mb, repeats
pcie
  link_gen_max
  link_gen_after_run
  link_width_current
  link_width_max
  theoretical_gb_s              at the current width
  theoretical_full_width_gb_s   if the slot ran at the card's full width
results                         list:
  direction                     host_to_device | device_to_host | device_to_device
  host_memory                   pinned | pageable | -
  size_mb
  best_gb_s
  median_gb_s
  error                         e.g. pinned allocation failed
```

If pinned memory cannot be allocated (not enough free RAM), the pinned rows show the error and the other rows still run.

## Python API

```python
from vramsuite.core.transfer import run_transfer_benchmark

info = run_transfer_benchmark(device_index=0, size_mb=256, repeats=5)

for row in info["results"]:
    print(row["direction"], row["host_memory"], row["best_gb_s"])
```
