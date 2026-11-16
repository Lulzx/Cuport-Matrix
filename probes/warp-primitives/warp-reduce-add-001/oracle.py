# Lane 0 of each warp ends with the sum of the warp's 32 inputs mod 2^32
# (CUDA C++ Programming Guide, "Warp Shuffle Functions", reduction example).
from cuport_oracle import Buffer, lcg_u32

ID = "warp-primitives/warp-reduce-add-001"
N = 128 * 256


def expected():
    v = lcg_u32(ID, N)
    return {"sums": Buffer("u32", [sum(v[b:b + 32]) & 0xFFFFFFFF for b in range(0, N, 32)])}
