# CUDA C++ Programming Guide, "atomicAdd()": reads old, stores old + val,
# atomically. int addition wraps in two's complement. Every float addend is an
# integer <= 255 and each bucket's total stays below 2^24, so float addition
# is exact in any order.
from cuport_oracle import Buffer, lcg_u32

ID = "atomics/atomic-add-001"
N = 32 * 128


def to_i32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x & 0x80000000 else x


def expected():
    isum, fsum = [0] * 16, [0] * 16
    for v in lcg_u32(ID, N):
        isum[v % 16] += (v >> 4) - (1 << 27)
        fsum[v % 16] += v & 0xFF
    assert max(fsum) < 2 ** 24
    return {"isum": Buffer("i32", [to_i32(x) for x in isum]), "fsum": Buffer("f32", [float(x) for x in fsum])}
