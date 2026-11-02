# Allocation probe and OOM risk

`vramsuite doctor` can confirm how much VRAM the current process can really allocate, and estimate whether a workflow of a given size will fit.

Both are explicit:

- the probe runs only with `--probe`;
- the risk estimate runs only with `--estimate-mb`.

## Bounded CUDA allocation probe

The probe is designed to be conservative:

- It allocates through the CUDA driver API (`cuMemAlloc`); only the NVIDIA driver is needed, no PyTorch or CUDA Toolkit.
- It allocates memory in configurable steps.
- It never attempts more than `--probe-max-mb`.
- It keeps a configurable free VRAM floor through `--probe-free-floor-mb`.
- It can clamp the requested probe size through `--probe-max-free-ratio`.
- It releases every allocation before returning.
- It is not a full VRAM exhaustion test.

Run the probe:

```bash
uv run vramsuite doctor --probe
```

With custom safety limits:

```bash
uv run vramsuite doctor --probe --probe-max-mb 8192 --probe-step-mb 256 --probe-free-floor-mb 2048
```

With a maximum free-memory ratio:

```bash
uv run vramsuite doctor --probe --probe-max-mb 17000 --probe-max-free-ratio 0.90
```

Example result:

```text
Safe Probe
Available             True
Backend               cuda-driver
Attempted MB          8192
Allocated MB          8192
Safe Allocatable MB   6963
Safety Margin MB      1228
Error                 None
```

This confirms that the current process could allocate the configured amount within the probe limits.

It does **not** mean the whole GPU was stress-tested, and it does **not** intentionally push the GPU to OOM.

## Probe safety clamp

The probe clamps unsafe or unrealistic requests. Asking for more VRAM than is currently free does not force VRAM Suite to try that exact value:

```bash
uv run vramsuite doctor --probe --probe-max-mb 17000
```

```text
Requested probe_max_mb=17000 MB exceeds current driver_free_mb=14052 MB.
Probe will be clamped to available free VRAM safety bounds.
Requested probe limit was clamped from 17000 MB to 11904 MB.
```

The final probe limit is the strictest active safety bound:

```text
min(
  probe_max_mb,
  driver_free_mb - probe_free_floor_mb,
  driver_free_mb * probe_max_free_ratio
)
```

The result is then rounded down to the configured allocation step.

## OOM risk estimation

The risk estimator checks whether a requested VRAM amount is likely to fit inside the currently known memory budget.

Without probe, it uses driver-level free VRAM from NVML:

```bash
uv run vramsuite doctor --estimate-mb 8000
```

With probe, it uses the safer probed budget:

```bash
uv run vramsuite doctor --probe --probe-max-mb 12288 --probe-step-mb 256 --estimate-mb 8000
```

Example result:

```text
OOM Risk Estimate
Available             True
Required MB           8000
Available MB          10444
Availability Source   probe_safe_allocatable
Remaining MB          2444
Usage Ratio           76.60%
Risk Level            medium
Reason                Required memory is close to available memory. Based on confirmed safe probe budget.
```

Risk levels:

- `low` — required memory is well below the available budget
- `medium` — required memory is close to the available budget
- `high` — required memory is very close to the available budget
- `critical` — required memory exceeds the available budget
- `unknown` — not enough information to estimate risk

### Availability source

The estimator reports where the available memory value came from:

- `driver_free_at_scan` — driver-level free VRAM from NVML at scan time
- `probe_safe_allocatable` — confirmed safe allocatable budget from the bounded CUDA probe
- `unknown` — no usable memory source was found

These values answer different questions.

`driver_free_at_scan`:

```text
How much VRAM did the driver report as free when the scan ran?
```

`probe_safe_allocatable`:

```text
How much memory did the current process confirm as safely allocatable within the configured probe limits?
```

A small default probe gives a small safe budget, for example `870 MB` from a `1024 MB` probe. That does not mean the GPU only has 870 MB available. It means this run only confirmed 870 MB because the probe limit was small.

## Windows and WSL

`.vramcard` records the runtime, so the same GPU can be compared across environments.

Example Windows runtime:

```text
OS        Windows
Python    3.12.9
WSL       False
NVML      True
```

Example WSL2 runtime:

```text
OS        Linux
Platform  Linux-6.6.x-microsoft-standard-WSL2
Python    3.10.12
WSL       True
NVML      True
```

In tested local runs on an RTX 5080, Windows and WSL2 produced matching bounded probe results with the same configuration:

```text
Probe allocated: 12288 MB
Safe budget:     10444 MB
Risk for 8000MB: medium
```

Compare two cards with `vramsuite compare` (see [`.vramcard` format](vramcard.md#comparing-cards)).

## Python API

```python
import vramsuite

result = vramsuite.run_doctor(
    with_probe=True,
    probe_max_mb=8192,
    probe_step_mb=256,
    probe_floor_mb=2048,
    probe_max_free_ratio=0.90,
    estimate_mb=8000,
)

print(result["probe"])
print(result["risk_estimate"])
```
