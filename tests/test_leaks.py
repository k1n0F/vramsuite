import ctypes

import pytest

from vramsuite.backends.cuda import CudaRuntime


def test_free_all_frees_everything_even_if_one_fails():
    cuda = CudaRuntime()
    freed = []

    def fake_free(ptr):
        if ptr is None:
            return
        freed.append(ptr.value)
        if ptr.value == 2:
            raise RuntimeError("cudaFree failed")

    cuda.free = fake_free

    with pytest.raises(RuntimeError, match="cudaFree failed"):
        cuda.free_all(ctypes.c_void_p(1), ctypes.c_void_p(2), None, ctypes.c_void_p(3))

    assert freed == [1, 2, 3]
