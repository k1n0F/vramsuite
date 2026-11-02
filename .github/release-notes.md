
## Install

```bash
pip install vramsuite=={VERSION}
pip install "vramsuite[compute]=={VERSION}"   # + cuBLAS for doctor --compute-check
```

Or download `vramsuite.exe` below (Windows, no Python needed) and run `vramsuite.exe doctor` in a terminal. Double-clicking it runs `doctor` and waits for Enter.

## Requirements

- Windows 10/11 x64 or Linux, an NVIDIA GPU and its driver.
- No Python, PyTorch or CUDA Toolkit needed for the exe.

## Limited without the CUDA Toolkit

Everything works with only the NVIDIA driver, except `doctor --compute-check` (cuBLAS / cuBLASLt FP32 and FP8 checks). It needs the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), or `pip install "vramsuite[compute]"`. The exe cannot install extras; with the toolkit installed it finds cuBLAS on its own.

## Windows SmartScreen

The exe is not code-signed, so Windows may show "Windows protected your PC" / "Unknown publisher". Click *More info* → *Run anyway*, or first check the file against the checksum:

```text
SHA256  {SHA256}  vramsuite.exe
```

```powershell
Get-FileHash .\vramsuite.exe -Algorithm SHA256
```
