# CUDA C++ Programming Guide, "Warp Shuffle Functions", __shfl_xor_sync: the
# source lane is laneId ^ laneMask. "If width is less than warpSize then each
# group of width consecutive threads are able to access elements from earlier
# groups of threads, however if they attempt to access elements from later
# groups of threads their own value of var will be returned."
# Equivalently (PTX shfl.bfly with clamp): valid iff src <= (lane | (width-1)).
from cuport_oracle import Buffer, lcg_u32

ID = "warp-primitives/shfl-xor-width-001"
N = 8 * 64
WIDTHS = (32, 16, 8, 4, 2)


def expected():
    v = lcg_u32(ID, N)
    out = []
    for g in range(N):
        base, lane = g - g % 32, g % 32
        for w in WIDTHS:
            for m in range(5):
                src = lane ^ (1 << m)
                out.append(v[base + src] if src <= (lane | (w - 1)) else v[g])
    return {"out": Buffer("u32", out)}
