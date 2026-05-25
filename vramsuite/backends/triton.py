"""
Triton backend placeholder.

This module is reserved for future Triton-based compute probes.

Planned responsibilities:
- detect Triton availability and version
- validate whether Triton can compile and launch kernels
- run small memory/compute probe kernels
- report kernel compile/runtime errors without crashing VRAM Suite

This backend is optional and must never be required for basic doctor diagnostics.
"""