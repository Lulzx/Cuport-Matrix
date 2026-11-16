# CUDA C++ Programming Guide, "Intrinsic Functions": __expf(x) has a documented
# error bound of 2 + floor(1.173 * |x|) ulp. For |x| < 0.85, 1.173 * |x| < 1,
# so the bound is 2 ulp.
import math
from cuport_oracle import Buffer, Lcg, bits_f32, f32

ID = "fp-semantics/fast-expf-001"
N = 4096
SCALE = 0.84


def expected():
    inputs = [0.0, bits_f32(0x80000000)]
    g = Lcg(ID)
    for _ in range(N):
        u = g.next_unit_f32()
        inputs.append(f32((2.0 * u - 1.0) * SCALE))
    y = [f32(math.exp(x)) for x in inputs]
    return {"y": Buffer("f32", y)}
