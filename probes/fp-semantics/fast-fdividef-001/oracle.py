# CUDA C++ Programming Guide, "Intrinsic Functions": __fdividef(x, y) has a
# maximum error of 2 ulp for |y| in [2^-126, 2^126] when x/y is a finite normal.
import numpy as np
from cuport_oracle import Buffer, Lcg, bits_f32

ID = "fp-semantics/fast-fdividef-001"
N = 4096


def expected():
    g = Lcg(ID)
    y_out = []
    for _ in range(N):
        ux = g.next_u32()
        uy = g.next_u32()
        exp_x = 70 + ((ux >> 23) % 115)
        exp_y = 70 + ((uy >> 23) % 115)
        bx = (ux & 0x80000000) | (exp_x << 23) | (ux & 0x007FFFFF)
        by = (uy & 0x80000000) | (exp_y << 23) | (uy & 0x007FFFFF)
        fx = bits_f32(bx)
        fy = bits_f32(by)
        res = np.float32(fx) / np.float32(fy)
        y_out.append(float(res))
    return {"y": Buffer("f32", y_out)}
