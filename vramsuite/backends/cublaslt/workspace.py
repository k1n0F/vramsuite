from __future__ import annotations

import ctypes
from types import TracebackType

from vramsuite.backends.cuda import CudaRuntime



class DeviceWorkspace:
    def __init__(self, cuda: CudaRuntime, size_bytes: int) -> None:
        if size_bytes < 0:
            raise ValueError("workspace size must be >= 0")
        
        self.cuda = cuda
        self.size_bytes = size_bytes
        self.ptr: ctypes.c_void_p | None = None

    def __enter__(self) -> "DeviceWorkspace":
        if self.size_bytes == 0:
            self.ptr = None
            return self
        
        self.ptr = self.cuda.malloc(self.size_bytes)
        return self
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.ptr is not None:
            try:
                self.cuda.free(self.ptr)
            finally:
                self.ptr = None

    @property
    def raw_ptr(self) -> ctypes.c_void_p | None:
        return self.ptr