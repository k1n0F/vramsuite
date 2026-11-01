from vramsuite.compute.minifloat import FORMATS, _positive_table, decode, encode


def test_every_finite_code_roundtrips():
    for fmt in FORMATS:
        values, codes = _positive_table(fmt)
        for value, code in zip(values, codes):
            assert encode(value, fmt) == code


def test_known_values_rounding_and_saturation():
    assert decode(0x7E, "e4m3") == 448.0
    assert decode(0x7B, "e5m2") == 57344.0
    assert decode(0x07, "e2m1") == 6.0
    assert encode(1.0625, "e4m3") == 0x38  # tie 1.0 / 1.125 -> even
    assert encode(1000.0, "e4m3") == 0x7E  # saturate to 448
    assert encode(-2.5, "e2m1") == 0b1100  # tie 2 / 3 -> even, negative
