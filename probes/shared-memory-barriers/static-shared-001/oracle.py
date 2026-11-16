# CUDA C++ Programming Guide, "__shared__": one copy per block, visible to all
# threads of the block; __syncthreads() makes the writes visible before reads.
from cuport_oracle import Buffer, lcg_u32

ID = "shared-memory-barriers/static-shared-001"
BLOCKS = 64


def expected():
    data = lcg_u32(ID, BLOCKS * 256)
    out = [data[b * 256 + 255 - t] for b in range(BLOCKS) for t in range(256)]
    return {"out": Buffer("u32", out)}
