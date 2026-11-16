# CUDA C++ Programming Guide, "Atomic Functions": atomics on shared memory are
# atomic with respect to the other threads of the block.
from cuport_oracle import Buffer, lcg_u32

ID = "atomics/shared-contention-001"
BLOCKS, THREADS = 64, 512


def expected():
    v = lcg_u32(ID, BLOCKS * THREADS)
    totals, hist = [], []
    for b in range(BLOCKS):
        block = v[b * THREADS:(b + 1) * THREADS]
        totals.append(sum(x & 0xFFFF for x in block) & 0xFFFFFFFF)
        hist.extend(sum(1 for x in block if x % 8 == k) for k in range(8))
    return {"totals": Buffer("u32", totals), "hist": Buffer("u32", hist)}
