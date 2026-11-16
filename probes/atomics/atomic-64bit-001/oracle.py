# CUDA C++ Programming Guide, "Atomic Functions": 64-bit unsigned atomicAdd,
# atomicMax and atomicCAS are atomic on the whole 64-bit word. Addition wraps
# mod 2^64; XOR via the CAS loop is order-independent.
from cuport_oracle import Buffer, lcg_u32

ID = "atomics/atomic-64bit-001"
N = 8 * 128
M = (1 << 64) - 1


def expected():
    words = lcg_u32(ID, 2 * N)
    s, mx, x = [0] * 4, [0] * 4, [0] * 4
    for g in range(N):
        v = (words[2 * g] << 32) | words[2 * g + 1]
        b = words[2 * g] % 4
        s[b] = (s[b] + v) & M
        mx[b] = max(mx[b], v)
        x[b] ^= (v * 0x9E3779B97F4A7C15) & M
    return {"sum": Buffer("u64", s), "max": Buffer("u64", mx), "cas_xor": Buffer("u64", x)}
