"""probes/common/cuport_oracle.py exact rounding, checked against independent
implementations: CPython's float32 packing (round to nearest even), math.fma,
math.sqrt, numpy's float16 and bf16 truncation."""

import math
import random
import struct
from fractions import Fraction

import numpy as np
import pytest

import cuport_oracle as co

rng = random.Random(1234)


def rand_f64():
    while True:
        x = co.bits_f64(rng.getrandbits(64))
        if math.isfinite(x):
            return x


def test_f32_nearest_matches_struct_packing():
    for _ in range(20000):
        x = rand_f64() if rng.random() < 0.5 else rng.uniform(-1, 1) * 2.0 ** rng.randint(-160, 140)
        try:
            want = struct.unpack("<I", struct.pack("<f", x))[0]
        except OverflowError:  # struct refuses values that round to infinity
            want = co.round_bits(math.copysign(math.inf, x), "f32")
        assert co.round_bits(x, "f32") == want, x


def test_f32_directed_modes_bracket_the_value():
    for _ in range(5000):
        x = Fraction(rng.uniform(-1e6, 1e6)) / 3
        lo, hi = co.bits_f32(co.round_bits(x, "f32", "rd")), co.bits_f32(co.round_bits(x, "f32", "ru"))
        assert lo < x < hi  # x/3 is never an f32
        assert co.f32_bits(hi) - co.f32_bits(lo) in (1, -1)  # adjacent floats
        rz = co.bits_f32(co.round_bits(x, "f32", "rz"))
        assert rz == (lo if x > 0 else hi)


def rand_f64_moderate():
    return rng.uniform(-2, 2) * 2.0 ** rng.randint(-500, 500)


def test_f64_fma_matches_math_fma():
    for _ in range(20000):
        a, b = rand_f64_moderate(), rand_f64_moderate()
        c = -a * b * rng.choice([1.0, 1 + 2.0 ** -40, rng.uniform(-3, 3)])  # include cancellation
        want = math.fma(a, b, c)
        assert co.round_value(Fraction(a) * Fraction(b) + Fraction(c), "f64") == want


def test_f64_sqrt_matches_math_sqrt():
    for _ in range(20000):
        x = abs(rand_f64())
        assert co.round_sqrt_bits(x, "f64") == co.f64_bits(math.sqrt(x))


def test_f32_sqrt_directed():
    for _ in range(5000):
        x = co.bits_f32(rng.getrandbits(31) & 0x7F7FFFFF)
        rn = co.bits_f32(co.round_sqrt_bits(x, "f32", "rn"))
        rd = co.bits_f32(co.round_sqrt_bits(x, "f32", "rd"))
        ru = co.bits_f32(co.round_sqrt_bits(x, "f32", "ru"))
        assert Fraction(rd) ** 2 <= Fraction(x) <= Fraction(ru) ** 2
        assert rn in (rd, ru)
        assert co.round_sqrt_bits(x, "f32", "rz") == co.round_sqrt_bits(x, "f32", "rd")


def test_f16_nearest_matches_numpy():
    xs = [co.bits_f32(rng.getrandbits(32)) for _ in range(20000)]
    xs = [x for x in xs if math.isfinite(x)]
    with np.errstate(over="ignore"):
        want = np.array(xs, dtype=np.float32).astype(np.float16).view(np.uint16)
    assert [co.round_bits(x, "f16") for x in xs] == [int(w) for w in want]


def test_bf16_toward_zero_is_truncation():
    for _ in range(20000):
        b = rng.getrandbits(32)
        x = co.bits_f32(b)
        if math.isfinite(x):
            assert co.round_bits(x, "bf16", "rz") == b >> 16


INF, MAX = 0x7C00, 0x7BFF


@pytest.mark.parametrize("mode,pos,neg", [("rn", INF, INF), ("rz", MAX, MAX), ("ru", INF, MAX), ("rd", MAX, INF)])
def test_f16_overflow(mode, pos, neg):
    assert co.round_bits(70000.0, "f16", mode) == pos
    assert co.round_bits(-70000.0, "f16", mode) == 0x8000 | neg


def test_exact_zero_sign_and_subnormals():
    assert co.round_bits(0, "f32", "rn") == 0
    assert co.round_bits(0, "f32", "rd") == 0x80000000
    for mode in co.ROUNDING_MODES:  # converting a float zero keeps its sign
        assert co.round_bits(0.0, "f16", mode) == 0
        assert co.round_bits(-0.0, "f16", mode) == 0x8000
    assert co.round_bits(Fraction(1, 2 ** 149), "f32") == 1  # smallest f32 subnormal
    assert co.round_bits(Fraction(1, 2 ** 150), "f32") == 0  # tie rounds to even (zero)
    assert co.round_bits(Fraction(3, 2 ** 151), "f32") == 1
    assert co.round_bits(Fraction(1, 2 ** 150), "f32", "ru") == 1
    assert co.round_bits(2.0 ** -126 - 2.0 ** -150, "f32") == 0x00800000  # rounds up into the normals
