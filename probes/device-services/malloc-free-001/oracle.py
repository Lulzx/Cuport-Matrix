# CUDA C++ Programming Guide, "Dynamic Global Memory Allocation and
# Operations": device malloc returns a pointer to at least the requested
# bytes from a heap sized by cudaLimitMallocHeapSize (8 MB here; 256 threads
# x 256 B at most is far below it), or NULL on exhaustion. The memory is
# ordinary global memory, so what a thread writes it reads back.
from cuport_oracle import Buffer, lcg_u32

ID = "device-services/malloc-free-001"
BLOCKS, THREADS = 4, 64
M = 0xFFFFFFFF


def _fold(x):
    n = 16 + (x & 0xF0)
    return sum(((x + 7 * k) & 0xFF) * (k + 1) for k in range(n)) & M


def expected():
    v = lcg_u32(ID, BLOCKS * THREADS)
    return {
        "nulls": Buffer("u32", [0] * len(v)),
        "sums": Buffer("u32", [_fold(x) for x in v]),
    }
