# CUDA C++ Programming Guide, "atomicCAS()": stores (old == compare ? val : old)
# and returns old, atomically. A correct retry loop applies every thread's
# update exactly once, so each bucket ends at the sum of its addends, and the
# values returned to the threads of one bucket are pairwise distinct (each is
# a different intermediate total).
from cuport_oracle import Buffer, lcg_u32

ID = "atomics/cas-loop-001"
N = 16 * 128
M = 0xFFFFFFFF


def expected():
    acc, count = [0] * 8, [0] * 8
    for v in lcg_u32(ID, N):
        acc[v % 8] = (acc[v % 8] + v * 2654435761) & M
        count[v % 8] += 1
    return {"acc": Buffer("u32", acc), "distinct_returns": Buffer("u32", count)}
