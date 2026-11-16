# CUDA C++ Programming Guide, "Cooperative Groups" / "Grid Synchronization":
# grid.sync() orders all writes before it against all reads after it across
# the whole grid, for kernels launched with cudaLaunchCooperativeKernel on a
# device reporting cudaDevAttrCooperativeLaunch = 1. A device that does not
# support it fails the launch with an error, which CUPORT_CHECK reports.
from cuport_oracle import Buffer, lcg_u32

ID = "limits-refusals/cooperative-grid-sync-001"
BLOCKS, THREADS = 8, 128
M = 0xFFFFFFFF


def expected():
    n = BLOCKS * THREADS
    v = lcg_u32(ID, n)
    mid = [(x * 2654435761) & M for x in v]
    out = [mid[(i + THREADS) % n] ^ v[i] for i in range(n)]
    return {
        "cooperative_launch_attr": Buffer("u32", [1]),
        "out": Buffer("u32", out),
    }
