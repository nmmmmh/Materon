"""Contract and regression checks for test float equivalence static in the TF2.9 multi-backbone engine."""

import numpy as np


def _ordered_float32_bits(values):
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32).astype(np.uint64)
    sign = (bits & np.uint64(0x80000000)) != 0
    magnitude = bits & np.uint64(0x7FFFFFFF)
    return np.where(
        sign,
        np.uint64(0x80000000) - magnitude,
        np.uint64(0x80000000) + magnitude,
    )


def test_adjacent_positive_float_is_one_ulp():
    x = np.asarray([524288.0], dtype=np.float32)
    y = np.nextafter(x, np.asarray([np.inf], dtype=np.float32))
    distance = np.abs(
        _ordered_float32_bits(y).astype(np.int64)
        - _ordered_float32_bits(x).astype(np.int64)
    )
    assert int(distance[0]) == 1
    assert float(y[0] - x[0]) == 0.0625


def test_signed_zero_maps_to_same_point():
    values = _ordered_float32_bits(
        np.asarray([0.0, -0.0], dtype=np.float32)
    )
    assert int(values[0]) == int(values[1])
