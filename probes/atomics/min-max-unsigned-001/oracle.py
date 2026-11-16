# CUDA C++ Programming Guide, "atomicMin()" / "atomicMax()": the unsigned
# overloads compare as unsigned, the int overloads as signed two's complement.
from cuport_oracle import Buffer, lcg_u32

ID = "atomics/min-max-unsigned-001"
N = 16 * 256


def s32(x):
    return x - (1 << 32) if x & 0x80000000 else x


def expected():
    buckets = [[] for _ in range(8)]
    for v in lcg_u32(ID, N):
        buckets[(v >> 3) % 8].append(v)
    return {
        "umin": Buffer("u32", [min(b) for b in buckets]),
        "umax": Buffer("u32", [max(b) for b in buckets]),
        "smin": Buffer("i32", [min(map(s32, b)) for b in buckets]),
        "smax": Buffer("i32", [max(map(s32, b)) for b in buckets]),
    }
