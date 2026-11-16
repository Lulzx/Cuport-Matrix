# CUDA C++ Programming Guide, "Intrinsic Functions": __sinf(x) has a maximum
# absolute error of 2^-21.41 for x in [-pi, pi]. With 2^-25 added for single-
# precision oracle rounding, the tolerance is abs:3.9e-7.
import math
from cuport_oracle import Buffer, Lcg, bits_f32, f32

ID = "fp-semantics/fast-sinf-001"
N = 4096
PI_F = bits_f32(0x40490FDA)  # 3.1415925f, pi rounded down to stay inside [-pi, pi]


def expected():
    inputs = [0.0, bits_f32(0x80000000), PI_F, -PI_F]
    g = Lcg(ID)
    for _ in range(N):
        u = g.next_unit_f32()
        inputs.append(f32((2.0 * u - 1.0) * PI_F))
    y = [f32(math.sin(x)) for x in inputs]
    return {"y": Buffer("f32", y)}
