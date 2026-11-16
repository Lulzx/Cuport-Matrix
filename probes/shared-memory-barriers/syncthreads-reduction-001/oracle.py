# CUDA C++ Programming Guide, "Synchronization Functions": __syncthreads()
# waits until every thread of the block reaches it and their shared and global
# writes are visible to the block. Unsigned addition wraps, so order does not matter.
from cuport_oracle import Buffer, lcg_u32

ID = "shared-memory-barriers/syncthreads-reduction-001"
BLOCKS = 128


def expected():
    data = lcg_u32(ID, BLOCKS * 512)
    return {"sums": Buffer("u32", [sum(data[b * 512:(b + 1) * 512]) & 0xFFFFFFFF for b in range(BLOCKS)])}
