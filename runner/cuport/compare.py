"""Compare an observed buffer with an oracle Buffer under a probe tolerance."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cuport_oracle import Buffer, bits_f32, bits_f64, f32_bits, f64_bits

_INT_MASK = {"u16": 0xFFFF, "u32": 0xFFFFFFFF, "i32": 0xFFFFFFFF, "u64": (1 << 64) - 1}
_FLOAT = {"f32": (f32_bits, bits_f32, 32), "f64": (f64_bits, bits_f64, 64)}


@dataclass
class Comparison:
    status: str  # bit-exact | within-tolerance | mismatch
    mismatches: int = 0
    detail: str = ""


def parse_tolerance(tol: str) -> tuple[str, float]:
    if tol == "bit-exact":
        return ("bit-exact", 0)
    kind, sep, value = tol.partition(":")
    if not sep or kind not in ("abs", "rel", "ulp"):
        raise ValueError(f"unknown tolerance {tol!r}")
    v = int(value) if kind == "ulp" else float(value)
    if v < 0 or (isinstance(v, float) and math.isnan(v)):
        raise ValueError(f"negative tolerance {tol!r}")
    return (kind, v)


def ulp_distance(dtype: str, a: int, b: int) -> int:
    bits = _FLOAT[dtype][2]
    sign = 1 << (bits - 1)

    def ordered(x: int) -> int:
        return -(x & (sign - 1)) if x & sign else x

    return abs(ordered(a) - ordered(b))


def _expected_bits(dtype: str, v) -> int:
    if dtype in _FLOAT:
        return _FLOAT[dtype][0](v)
    return int(v) & _INT_MASK[dtype]


def _within(dtype: str, kind: str, limit: float, exp_v, got_bits: int) -> bool:
    if kind == "bit-exact":
        return False
    if dtype in _FLOAT:
        got = _FLOAT[dtype][1](got_bits)
        if math.isnan(exp_v) or math.isnan(got):
            return math.isnan(exp_v) and math.isnan(got)
        if math.isinf(exp_v) or math.isinf(got):
            return exp_v == got
        if kind == "ulp":
            return ulp_distance(dtype, _expected_bits(dtype, exp_v), got_bits) <= limit
        if kind == "abs":
            return abs(got - exp_v) <= limit
        return abs(got - exp_v) <= limit * abs(exp_v)
    if kind == "abs" and dtype in ("u32", "i32", "u64"):
        got = got_bits - (1 << 32) if dtype == "i32" and got_bits & 0x80000000 else got_bits
        return abs(got - int(exp_v)) <= limit
    return False


def compare_buffer(expected: Buffer, dtype: str, observed: list[int], tolerance: str) -> Comparison:
    if dtype != expected.dtype:
        return Comparison("mismatch", len(expected.values), f"dtype {dtype}, want {expected.dtype}")
    if len(observed) != len(expected.values):
        return Comparison("mismatch", len(expected.values), f"{len(observed)} elements, want {len(expected.values)}")
    kind, limit = parse_tolerance(tolerance)
    width = 16 if dtype in ("u64", "f64") else 4 if dtype == "u16" else 8
    within = 0
    bad = []
    for i, (exp_v, got) in enumerate(zip(expected.values, observed)):
        want = _expected_bits(dtype, exp_v)
        if got == want:
            continue
        if _within(dtype, kind, limit, exp_v, got):
            within += 1
        else:
            bad.append(f"[{i}] got {got:0{width}x} want {want:0{width}x}")
    if bad:
        return Comparison("mismatch", len(bad), "; ".join(bad[:3]) + (" ..." if len(bad) > 3 else ""))
    if within:
        return Comparison("within-tolerance", 0, f"{within} elements within {tolerance}")
    return Comparison("bit-exact")
