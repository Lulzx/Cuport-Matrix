# nvcc documentation, --fmad: "enables (disables) the contraction of floating-point
# multiplies and adds/subtracts into floating-point multiply-add operations (FMAD,
# FFMA, or DFMA). The default is true." CUDA C++ Programming Guide, "Intrinsic
# Functions": __fmul_rn and __fadd_rn are never merged into a multiply-add.
# fmaf is a single correctly rounded operation (IEEE 754 fusedMultiplyAdd).
#
# expr  = a*b + c as written: one rounding, because nvcc contracts it by default
# fused = fmaf(a, b, c): one rounding
# split = __fadd_rn(__fmul_rn(a, b), c): two roundings
from fractions import Fraction

from cuport_oracle import Buffer, Lcg, bits_f32, f32, round_value

ID = "fp-semantics/fma-contraction-001"
N = 4096


def inputs():
    g = Lcg(ID)
    a, b, c = [], [], []
    for i in range(N):
        a.append(bits_f32(0x3F800000 | (g.next_u32() >> 9)))
        b.append(bits_f32(0x3F800000 | (g.next_u32() >> 9)))
        u = g.next_u32()
        c.append(-f32(a[-1] * b[-1]) if i % 2 == 0 else bits_f32(0xC0000000 | (u >> 9)))
    return a, b, c


def expected():
    a, b, c = inputs()
    fused = [round_value(Fraction(x) * Fraction(y) + Fraction(z), "f32") for x, y, z in zip(a, b, c)]
    split = [round_value(Fraction(f32(x * y)) + Fraction(z), "f32") for x, y, z in zip(a, b, c)]
    return {"expr": Buffer("f32", fused), "fused": Buffer("f32", fused), "split": Buffer("f32", split)}
