"""
Pure-Python encode/decode for small float formats (FP8 / FP4).

Used to fill host buffers for low-precision cuBLASLt GEMMs without numpy/torch.
Encoding is round-to-nearest-even with saturation to the max finite value,
which matches CUDA's `__nv_cvt_float_to_fp8(..., __NV_SATFINITE, ...)`.
"""

from __future__ import annotations

from bisect import bisect_left
from functools import lru_cache

# name -> (exp_bits, man_bits)
FORMATS = {
    "e4m3": (4, 3),  # FP8, no inf, NaN = S.1111.111, max 448
    "e5m2": (5, 2),  # FP8, IEEE-like: exp=all-ones is inf/NaN, max 57344
    "e2m1": (2, 1),  # FP4, no inf/NaN, max 6
}


def _is_finite_code(fmt: str, exp: int, man: int) -> bool:
    exp_bits, man_bits = FORMATS[fmt]
    exp_max = (1 << exp_bits) - 1

    if fmt == "e4m3":
        return not (exp == exp_max and man == (1 << man_bits) - 1)
    if fmt == "e5m2":
        return exp != exp_max
    return True


def decode(code: int, fmt: str) -> float:
    exp_bits, man_bits = FORMATS[fmt]
    bias = (1 << (exp_bits - 1)) - 1

    sign = -1.0 if (code >> (exp_bits + man_bits)) & 1 else 1.0
    exp = (code >> man_bits) & ((1 << exp_bits) - 1)
    man = code & ((1 << man_bits) - 1)

    if not _is_finite_code(fmt, exp, man):
        return sign * float("inf") if fmt == "e5m2" and man == 0 else float("nan")

    if exp == 0:
        return sign * man / (1 << man_bits) * 2.0 ** (1 - bias)

    return sign * (1 + man / (1 << man_bits)) * 2.0 ** (exp - bias)


@lru_cache(maxsize=None)
def _positive_table(fmt: str) -> tuple[list[float], list[int]]:
    exp_bits, man_bits = FORMATS[fmt]
    values: list[float] = []
    codes: list[int] = []

    for code in range(1 << (exp_bits + man_bits)):
        exp = code >> man_bits
        man = code & ((1 << man_bits) - 1)
        if _is_finite_code(fmt, exp, man):
            values.append(decode(code, fmt))
            codes.append(code)

    # Codes are already in increasing value order for sign=0.
    return values, codes


def encode(value: float, fmt: str) -> int:
    exp_bits, man_bits = FORMATS[fmt]
    sign_bit = 1 << (exp_bits + man_bits)

    if value != value:
        raise ValueError(f"cannot encode NaN as {fmt}")

    sign = sign_bit if value < 0 else 0
    magnitude = abs(value)

    values, codes = _positive_table(fmt)

    if magnitude >= values[-1]:
        return sign | codes[-1]

    index = bisect_left(values, magnitude)
    if values[index] == magnitude:
        return sign | codes[index]

    lo, hi = index - 1, index
    lo_gap = magnitude - values[lo]
    hi_gap = values[hi] - magnitude

    if lo_gap < hi_gap or (lo_gap == hi_gap and codes[lo] % 2 == 0):
        return sign | codes[lo]

    return sign | codes[hi]


def encode_bytes(values: list[float], fmt: str) -> bytes:
    """One code per byte. FP4 nibble packing is not done here."""
    return bytes(encode(value, fmt) for value in values)


if __name__ == "__main__":
    for fmt in FORMATS:
        values, codes = _positive_table(fmt)
        for value, code in zip(values, codes):
            assert encode(value, fmt) == code, (fmt, value, code)
            assert encode(-value, fmt) == code | (1 << sum(FORMATS[fmt])) or value == 0

    assert decode(0x7E, "e4m3") == 448.0
    assert decode(0x7B, "e5m2") == 57344.0
    assert decode(0x07, "e2m1") == 6.0
    assert encode(1.0, "e4m3") == 0x38
    assert encode(1.0625, "e4m3") == 0x38  # tie between 1.0 and 1.125 -> even
    assert encode(1000.0, "e4m3") == 0x7E  # saturates
    assert encode(2.5, "e2m1") == encode(2.0, "e2m1")  # tie between 2 and 3 -> even
    print("minifloat ok")
