# CUDA C++ Programming Guide, "Built-in Variables": blockIdx, threadIdx,
# blockDim and gridDim; a 1D launch numbers threads 0..blockDim.x-1 in each block.
from cuport_oracle import Buffer

BLOCKS, THREADS = 37, 100


def expected():
    packed = [(b << 16) | t for b in range(BLOCKS) for t in range(THREADS)]
    return {"packed": Buffer("u32", packed), "dims": Buffer("u32", [THREADS, BLOCKS])}
