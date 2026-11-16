# CUDA C++ Programming Guide, "Time Function": clock64() returns a 64-bit
# per-multiprocessor counter that is incremented every clock cycle. Two reads
# by the same thread separated by 4096 dependent multiply-adds must differ
# and the second must be larger. The raw delta is hardware dependent and is
# not checked; only its sign and non-zeroness are.
from cuport_oracle import Buffer

BLOCKS, THREADS = 2, 64
M = 0xFFFFFFFF


def _lcg_run(i):
    acc = i
    for _ in range(4096):
        acc = (acc * 1664525 + 1013904223) & M
    return acc


def expected():
    n = BLOCKS * THREADS
    return {
        "sink": Buffer("u32", [_lcg_run(i) for i in range(n)]),
        "increasing": Buffer("u32", [1] * n),
        "any_zero_delta": Buffer("u32", [0]),
    }
