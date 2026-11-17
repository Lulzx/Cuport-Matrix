# CUDA Math API, "Single Precision Intrinsics": __fadd_rd "Add two floating-point
# values in round-down mode", and likewise every __fadd_, __fmul_, __fdiv_,
# __fsqrt_ and __fmaf_ variant with suffix rn (nearest even), rz (toward zero),
# ru (toward +inf), rd (toward -inf); each is documented as IEEE-compliant, i.e.
# the exact result rounded once in that direction.
from fractions import Fraction

from cuport_oracle import ROUNDING_MODES, Buffer, Lcg, bits_f32, round_sqrt_bits, round_value

ID = "fp-semantics/rounding-modes-001"
N = 1024


def _draw(g):
    s = g.next_u32() & 0x80000000
    e = 117 + g.next_u32() % 21
    return bits_f32(s | (e << 23) | (g.next_u32() >> 9))


def inputs():
    g = Lcg(ID)
    a, b, c = [], [], []
    for _ in range(N):
        a.append(_draw(g))
        b.append(_draw(g))
        c.append(_draw(g))
    return a, b, c


def expected():
    a, b, c = inputs()
    F = Fraction
    exact = {
        "add": [F(x) + F(y) for x, y in zip(a, b)],
        "mul": [F(x) * F(y) for x, y in zip(a, b)],
        "div": [F(x) / F(y) for x, y in zip(a, b)],
        "fma": [F(x) * F(y) + F(z) for x, y, z in zip(a, b, c)],
    }
    out = {}
    for op in ("add", "mul", "div", "sqrt", "fma"):
        for m in ROUNDING_MODES:
            if op == "sqrt":
                vals = [bits_f32(round_sqrt_bits(abs(x), "f32", m)) for x in a]
            else:
                vals = [round_value(v, "f32", m) for v in exact[op]]
            out[f"{op}_{m}"] = Buffer("f32", vals)
    return out
