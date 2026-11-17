# CUDA C++ Programming Guide, "Floating-Point Standard": double-precision add and
# multiply are IEEE 754 round-to-nearest-even; "Double-Precision Floating-Point
# Functions": x/y, sqrt(x) and fma(x,y,z) have 0 ulp maximum error (correctly
# rounded). Every result here is the exact value rounded once to binary64.
from fractions import Fraction

from cuport_oracle import Buffer, Lcg, bits_f64, round_sqrt_bits, round_value

ID = "fp-semantics/fp64-emulation-001"
N = 2048


def _draw(g, positive):
    s = 0 if positive else (g.next_u32() & 0x80000000) << 32
    e = 1023 - 8 + (g.next_u32() & 15)
    hi = g.next_u32() & 0xFFFFF
    lo = g.next_u32()
    return bits_f64(s | (e << 52) | (hi << 32) | lo)


def inputs():
    g = Lcg(ID)
    a, b, c = [], [], []
    for i in range(N):
        a.append(_draw(g, False))
        b.append(_draw(g, True))
        r = _draw(g, False)
        c.append(-(a[-1] * b[-1]) if i % 2 == 0 else r)  # Python float * is binary64 RNE
    return a, b, c


def expected():
    a, b, c = inputs()
    F = Fraction
    return {
        "add": Buffer("f64", [round_value(F(x) + F(y), "f64") for x, y in zip(a, b)]),
        "mul": Buffer("f64", [round_value(F(x) * F(y), "f64") for x, y in zip(a, b)]),
        "div": Buffer("f64", [round_value(F(x) / F(y), "f64") for x, y in zip(a, b)]),
        "sqrt": Buffer("f64", [bits_f64(round_sqrt_bits(y, "f64")) for y in b]),
        "fma": Buffer("f64", [round_value(F(x) * F(y) + F(z), "f64") for x, y, z in zip(a, b, c)]),
    }
