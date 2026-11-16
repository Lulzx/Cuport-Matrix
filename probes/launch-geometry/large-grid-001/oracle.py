# CUDA C++ Programming Guide, "Compute Capabilities": maximum x-dimension of a
# grid is 2^31-1 (y and z: 65535); 70001 blocks is a valid 1D launch.
import numpy as np

from cuport_oracle import Buffer

BLOCKS, THREADS = 70001, 256


def expected():
    gid = np.arange(BLOCKS * THREADS, dtype=np.uint64)
    block = (gid // THREADS).astype(np.uint32)
    values = (((gid * 2654435761) & 0xFFFFFFFF).astype(np.uint32) ^ block).reshape(BLOCKS, THREADS)
    fold = np.zeros(BLOCKS, dtype=np.uint64)
    for t in range(THREADS):
        fold = ((fold * 0x01000193) & 0xFFFFFFFF) ^ values[:, t]
    return {"block_fold": Buffer("u32", [int(x) for x in fold])}
