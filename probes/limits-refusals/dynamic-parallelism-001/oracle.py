# CUDA C++ Programming Guide, "CUDA Dynamic Parallelism" (CDP2): a child grid
# launched from a parent grid completes before the parent grid is considered
# complete, so host synchronisation on the parent also waits for the child.
# Expected: every word written by the child, in[i] * 5 + 1.
from cuport_oracle import Buffer, lcg_u32

ID = "limits-refusals/dynamic-parallelism-001"
BLOCKS, THREADS = 4, 64
M = 0xFFFFFFFF


def expected():
    v = lcg_u32(ID, BLOCKS * THREADS)
    return {"child_out": Buffer("u32", [(x * 5 + 1) & M for x in v])}
