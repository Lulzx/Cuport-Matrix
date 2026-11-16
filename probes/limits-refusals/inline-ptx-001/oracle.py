# PTX ISA, "Special Registers": %laneid is the thread's lane within its warp,
# 0..31. With 64 threads per 1-D block, lane = threadIdx.x % 32.
# This probe has expect: either. A compile-time diagnostic is the expected
# outcome for a CUDA-on-Metal tool; this oracle is what a tool that claims
# to accept inline PTX has to match.
from cuport_oracle import Buffer

BLOCKS, THREADS = 2, 64


def expected():
    return {"lane_id": Buffer("u32", [t % 32 for _ in range(BLOCKS) for t in range(THREADS)])}
