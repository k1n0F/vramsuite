import ctypes

from vramsuite.backends.nvml import NVML_CLOCK_MEM, _OptionalNvml


class FakeNvml:
    @staticmethod
    def nvmlDeviceGetFanSpeed(handle, out):
        return 3  # NVML_ERROR_NOT_SUPPORTED

    @staticmethod
    def nvmlDeviceGetClockInfo(handle, clock_type, out):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_uint)).contents.value = 15001
        return 0

    @staticmethod
    def nvmlErrorString(status):
        return b"Not Supported"


def test_failed_getters_record_a_reason_instead_of_vanishing():
    optional = _OptionalNvml(FakeNvml())
    handle = ctypes.c_void_p(1)

    assert optional.uint("nvmlDeviceGetFanSpeed", handle) is None
    assert optional.uint("nvmlDeviceGetNumGpuCores", handle) is None
    assert optional.uint("nvmlDeviceGetClockInfo", handle, NVML_CLOCK_MEM) == 15001

    assert optional.unavailable == {
        "nvmlDeviceGetFanSpeed": "Not Supported (code 3)",
        "nvmlDeviceGetNumGpuCores": "missing_symbol",
    }
