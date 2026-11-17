"""Oracle-side helpers. Mirrors probes/common/cuport.h exactly.

An oracle (probes/<family>/<id>/oracle.py) defines:

    def expected() -> dict[str, Buffer]

and may also define TEXT_LINES: list[str], the device printf lines the probe
must print, compared as a multiset because thread order is unspecified.
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction
from typing import NamedTuple, Sequence, Union

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_LCG_MUL = 6364136223846793005
_LCG_INC = 1442695040888963407
_M64 = (1 << 64) - 1

# Hex digits per element on the CUPORT protocol line.
DTYPE_HEX_WIDTH = {"u16": 4, "u32": 8, "i32": 8, "f32": 8, "u64": 16, "f64": 16}


def fnv1a64(s: str) -> int:
    h = _FNV_OFFSET
    for b in s.encode("utf-8"):
        h = ((h ^ b) * _FNV_PRIME) & _M64
    return h


class Lcg:
    """x_{n+1} = 6364136223846793005 * x_n + 1442695040888963407 (mod 2^64),
    x_0 = FNV-1a(probe id). Each draw advances once and returns bits 63..32."""

    def __init__(self, probe_id: str):
        self.x = fnv1a64(probe_id)

    @classmethod
    def from_seed(cls, seed: int) -> "Lcg":
        g = cls.__new__(cls)
        g.x = seed & _M64
        return g

    def next_u32(self) -> int:
        self.x = (_LCG_MUL * self.x + _LCG_INC) & _M64
        return self.x >> 32

    def next_unit_f32(self) -> float:
        """Uniform in [0, 1) with 24 random bits; exactly representable in f32."""
        return (self.next_u32() >> 8) / 16777216.0


def lcg_u32(probe_id: str, n: int) -> list[int]:
    g = Lcg(probe_id)
    return [g.next_u32() for _ in range(n)]


def f32(x: float) -> float:
    """Round a Python float to the nearest float32 value."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def f64_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def bits_f64(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u & _M64))[0]


# ---- exact IEEE 754 rounding ------------------------------------------------
# Formats as (exponent bits, explicit mantissa bits). Rounding modes follow the
# CUDA intrinsic suffixes: rn (nearest, ties to even), rz (toward zero),
# ru (toward +inf), rd (toward -inf).
FORMATS = {"f16": (5, 10), "bf16": (8, 7), "f32": (8, 23), "f64": (11, 52)}
ROUNDING_MODES = ("rn", "rz", "ru", "rd")


def _floor_log2(a: Fraction) -> int:
    """floor(log2(a)) for a > 0, exactly."""
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    elif Fraction(2) ** (e + 1) <= a:
        e += 1
    return e


def _encode(neg: bool, e: int, n: int, above_half: int, inexact: bool, fmt: str, mode: str) -> int:
    """Pack floor significand n (scaled so a normal value has 2^m <= n < 2^(m+1)) at
    exponent e, given how the discarded part compares with one half ulp."""
    ebits, mbits = FORMATS[fmt]
    bias = (1 << (ebits - 1)) - 1
    sign = (1 << (ebits + mbits)) if neg else 0
    if mode == "rn":
        up = above_half > 0 or (above_half == 0 and inexact and n & 1)
    elif mode == "rz":
        up = False
    elif mode == "ru":
        up = inexact and not neg
    elif mode == "rd":
        up = inexact and neg
    else:
        raise ValueError(f"unknown rounding mode {mode!r}")
    n += up
    if n == 1 << (mbits + 1):
        n >>= 1
        e += 1
    if e > bias:  # overflow: infinity, or the largest finite value when rounding toward zero
        to_inf = mode == "rn" or (mode == "ru" and not neg) or (mode == "rd" and neg)
        if to_inf:
            return sign | (((1 << ebits) - 1) << mbits)
        return sign | (((1 << ebits) - 2) << mbits) | ((1 << mbits) - 1)
    if n < 1 << mbits:  # subnormal (or zero)
        return sign | n
    return sign | ((e + bias) << mbits) | (n - (1 << mbits))


def _to_fraction(x) -> Fraction:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        raise ValueError("non-finite values have no exact rational")
    return Fraction(x)


def _inf_bits(fmt: str, neg: bool) -> int:
    ebits, mbits = FORMATS[fmt]
    return ((1 << (ebits + mbits)) if neg else 0) | (((1 << ebits) - 1) << mbits)


def round_bits(x, fmt: str, mode: str = "rn") -> int:
    """Bit pattern of the exact value x (int, float or Fraction) rounded once to fmt.

    A float input is a value being converted, so a float zero keeps its sign and an
    infinity stays infinite. An int or Fraction zero is an exact result: +0, or -0
    under rd, the IEEE 754 sign of an exact-zero sum of opposite-signed operands."""
    ebits, mbits = FORMATS[fmt]
    if isinstance(x, float) and math.isinf(x):
        return _inf_bits(fmt, x < 0)
    if isinstance(x, float) and x == 0:
        return (1 << (ebits + mbits)) if math.copysign(1.0, x) < 0 else 0
    a = _to_fraction(x)
    if a == 0:
        return (1 << (ebits + mbits)) if mode == "rd" else 0
    neg = a < 0
    a = abs(a)
    emin = 2 - (1 << (ebits - 1))
    e = max(_floor_log2(a), emin)
    q = a / Fraction(2) ** (e - mbits)
    n = q.numerator // q.denominator
    rem = q - n
    return _encode(neg, e, n, (rem > Fraction(1, 2)) - (rem < Fraction(1, 2)), rem != 0, fmt, mode)


def round_sqrt_bits(x, fmt: str, mode: str = "rn") -> int:
    """Bit pattern of sqrt(x) rounded once to fmt, exactly, for finite x >= 0."""
    a = _to_fraction(x)
    if a < 0:
        raise ValueError("sqrt of a negative value")
    if a == 0:
        return 0
    ebits, mbits = FORMATS[fmt]
    emin = 2 - (1 << (ebits - 1))
    e = max(_floor_log2(a) // 2, emin)
    q2 = a * Fraction(4) ** (mbits - e)  # (sqrt(a) / 2^(e - mbits))^2
    n = math.isqrt(q2.numerator // q2.denominator)
    exact = Fraction(n * n) == q2
    mid = Fraction(4 * n * n + 4 * n + 1, 4)  # (n + 1/2)^2
    return _encode(False, e, n, (q2 > mid) - (q2 < mid), not exact, fmt, mode)


def round_value(x, fmt: str, mode: str = "rn") -> float:
    """round_bits as a Python float, for f32 and f64 buffers."""
    b = round_bits(x, fmt, mode)
    return bits_f32(b) if fmt == "f32" else bits_f64(b)


class Buffer(NamedTuple):
    """Expected contents of one emitted buffer.

    dtype is a key of DTYPE_HEX_WIDTH. values holds Python ints for integer
    dtypes (i32 as signed ints) and Python floats for f32/f64; float values
    must already be rounded to their dtype (use f32()). u16 holds raw half or
    bf16 bit patterns."""

    dtype: str
    values: Sequence[Union[int, float]]
