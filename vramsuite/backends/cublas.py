"""
cuBLAS backend placeholder.

This module is reserved for future cuBLAS/cuBLASLt-based compute probes.

Planned responsibilities:
- detect cuBLAS library availability
- expose basic cuBLAS capability information
- run controlled GEMM probes through CUDA memory
- measure before/after/peak memory usage
- report cuBLAS status/errors without crashing VRAM Suite

This backend is experimental and should stay optional.
"""