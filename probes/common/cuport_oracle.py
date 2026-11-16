"""Oracle-side helpers. Mirrors probes/common/cuport.h exactly.

An oracle (probes/<family>/<id>/oracle.py) defines:

    def expected() -> dict[str, Buffer]

and may also define TEXT_LINES: list[str], the device printf lines the probe
must print, compared as a multiset because thread order is unspecified.
"""

from __future__ import annotations

import struct
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


class Buffer(NamedTuple):
    """Expected contents of one emitted buffer.

    dtype is a key of DTYPE_HEX_WIDTH. values holds Python ints for integer
    dtypes (i32 as signed ints) and Python floats for f32/f64; float values
    must already be rounded to their dtype (use f32()). u16 holds raw half or
    bf16 bit patterns."""

    dtype: str
    values: Sequence[Union[int, float]]
