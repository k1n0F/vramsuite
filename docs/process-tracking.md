# Process tracking

`vramsuite processes` shows how much VRAM each process uses on one GPU.

It observes processes from outside. Nothing has to run inside ComfyUI, a game, or a browser: VRAM Suite reads the same counters the operating system uses.

It answers questions like:

- Which processes are holding VRAM before inference starts?
- How much VRAM did ComfyUI use at its peak during a generation?
- Is a background app (overlay, browser, wallpaper) eating memory the workflow needs?

## Usage

Snapshot right now:

```bash
uv run vramsuite processes
```

Watch for 60 seconds and report each process's peak:

```bash
uv run vramsuite processes --watch 60
```

Sample faster during a short workload:

```bash
uv run vramsuite processes --watch 30 --interval 0.2
```

Hide small processes:

```bash
uv run vramsuite processes --min-mb 100
```

Pick another GPU (NVML index):

```bash
uv run vramsuite processes --gpu 1
```

Print the result as JSON:

```bash
uv run vramsuite processes --watch 60 --json
```

`Ctrl+C` ends a watch early and still prints the peaks collected so far.

### Typical workflow

1. Start `uv run vramsuite processes --watch 300` in a terminal.
2. Run a generation in ComfyUI.
3. Press `Ctrl+C` when it finishes.
4. Read `Peak MB` for the ComfyUI worker process.

## Where the data comes from

### Windows

NVML cannot report per-process memory on Windows. Under WDDM, Windows manages video memory, and NVML always returns `NOT_AVAILABLE` for a process's usage.

VRAM Suite reads the same data Task Manager shows:

- PDH counters `\GPU Process Memory(*)\Dedicated Usage` and `Shared Usage` (`pdh.dll`)
- Each counter instance carries an adapter LUID. The LUID is resolved to a PCI address through D3DKMT (`gdi32.dll`), so only the selected NVIDIA GPU is counted. Integrated GPUs and Microsoft Basic Render are skipped.
- Process path and command line come from `kernel32` / `ntdll`.

The `source` field is `windows-pdh`.

### Linux

NVML reports per-process memory directly. The `source` field is `nvml`.

## Example output

Real snapshot on Windows with ComfyUI running (shortened):

```text
     GPU 0 NVIDIA GeForce RTX 5080: processes (windows-pdh, 1 samples)
┌───────┬────────────────────┬──────────────┬───────────┬─────────────────────────┐
│ PID   │ Process            │ Dedicated MB │ Shared MB │ Command                 │
├───────┼────────────────────┼──────────────┼───────────┼─────────────────────────┤
│ 2460  │ <protected>        │      10109.9 │      12.4 │                         │
│ 44968 │ python.exe         │       8934.0 │     276.0 │ "...\python.exe"        │
│       │                    │              │           │ main.py --port 8188     │
│ 21908 │ NVIDIA Overlay.exe │       6067.7 │       0.3 │ "...\NVIDIA Overlay.exe"│
│ 16272 │ Discord.exe        │        357.1 │       0.3 │ "...\Discord.exe"       │
│ 9892  │ chrome.exe         │        352.2 │      30.1 │ "...\chrome.exe"        │
│       │                    │              │           │ --type=gpu-process      │
└───────┴────────────────────┴──────────────┴───────────┴─────────────────────────┘
NVML used: 11158.9 MB of 16303.0 MB. Sum over processes: 26979.2 MB (can exceed physical usage on Windows:
shared surfaces are counted in several processes, paged-out allocations are included)
```

### Reading the columns

- `Dedicated MB` — VRAM the process uses on the GPU
- `Peak MB` — highest `Dedicated MB` seen during `--watch` (only in watch mode)
- `Shared MB` — system RAM the GPU uses for this process
- `Command` — the command line, so several `python.exe` processes can be told apart

`<protected>` means Windows does not allow reading the process name. These are system processes such as `dwm.exe` (the desktop compositor) and `csrss.exe`.

### Why the sum is larger than the GPU

On Windows, the sum over processes can exceed both NVML used memory and the physical size of the card. This is expected:

- A surface shared between processes (for example, a window shared with `dwm.exe`) is counted in each of them.
- Allocations that Windows paged out to system RAM are still counted as the process's dedicated usage.

So per-process values are good for comparing processes and for tracking peaks, but `NVML used` is the number for "how full is the card".

### Finding the right ComfyUI process

A ComfyUI started from a venv usually has two `python.exe` processes: a small launcher and the worker that actually uses the GPU. The worker is the one with large `Dedicated MB`. Use the `Command` column and the PID to tell them apart.

## JSON fields

```text
gpu                index, name, total_mb, nvml_used_mb
source             "windows-pdh" or "nvml"
samples            number of samples taken
processes_sum_mb   sum of dedicated memory over all processes (not physical usage)
processes          list, sorted by peak_dedicated_mb:
  pid
  name             executable name, null if protected
  exe              full executable path
  command_line     full command line
  dedicated_mb     usage in the last sample
  peak_dedicated_mb
  shared_mb
```

Processes that exited during a watch keep their row with their peak value and `dedicated_mb` of `0`.

## Python API

```python
from vramsuite.core.processes import snapshot_gpu_processes, watch_gpu_processes

now = snapshot_gpu_processes(device_index=0, min_mb=100)

peaks = watch_gpu_processes(seconds=60, interval=0.5, device_index=0)
for row in peaks["processes"]:
    print(row["pid"], row["name"], row["peak_dedicated_mb"])
```
