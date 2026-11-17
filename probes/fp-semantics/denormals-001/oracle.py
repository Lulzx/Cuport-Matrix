# nvcc documentation, --ftz: "--ftz=false" (the default) keeps single-precision
# denormal inputs and outputs. CUDA C++ Programming Guide, "Floating-Point
# Standard": double precision always supports denormals. Every result is the
# exact value rounded once to nearest even (IEEE 754). A nonzero value keeps its
# sign through rounding; an exact-zero sum of opposite-signed values is +0.
from fractions import Fraction

from cuport_oracle import Buffer, Lcg, bits_f32, bits_f64, round_value

ID = "fp-semantics/denormals-001"
N = 2048
HALF = Fraction(1, 2)


def inputs():
    g = Lcg(ID)
    tiny, sub, sub2, dtiny, dsub = [], [], [], [], []
    for _ in range(N):
        s = g.next_u32() & 0x80000000
        e = 1 + (g.next_u32() & 1)
        tiny.append(bits_f32(s | (e << 23) | (g.next_u32() >> 9)))
        sub.append(bits_f32((g.next_u32() & 0x807FFFFF) | 1))
        sub2.append(bits_f32((g.next_u32() & 0x807FFFFF) | 1))
        ds = (g.next_u32() & 0x80000000) << 32
        de = 1 + (g.next_u32() & 1)
        hi = g.next_u32() & 0xFFFFF
        dtiny.append(bits_f64(ds | (de << 52) | (hi << 32) | g.next_u32()))
        ds2 = (g.next_u32() & 0x80000000) << 32
        hi2 = g.next_u32() & 0xFFFFF
        dsub.append(bits_f64(ds2 | (hi2 << 32) | g.next_u32() | 1))
    return tiny, sub, sub2, dtiny, dsub


def expected():
    tiny, sub, sub2, dtiny, dsub = inputs()
    return {
        "half_tiny": Buffer("f32", [round_value(Fraction(x) * HALF, "f32") for x in tiny]),
        "sub_add": Buffer("f32", [round_value(Fraction(x) + Fraction(y), "f32") for x, y in zip(sub, sub2)]),
        "sub_scale": Buffer("f32", [round_value(Fraction(x) * 2 ** 24, "f32") for x in sub]),
        "d_half": Buffer("f64", [round_value(Fraction(x) * HALF, "f64") for x in dtiny]),
        "d_scale": Buffer("f64", [round_value(Fraction(x) * 2 ** 54, "f64") for x in dsub]),
    }
