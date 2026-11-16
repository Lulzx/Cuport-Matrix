# CUDA C++ Programming Guide, "Assertion": a failed device assert stops the
# kernel and the next synchronising host call returns cudaErrorAssert.
# CUDA Runtime API, cudaError: cudaErrorAssert = 710. A launch is accepted
# before the assertion runs, so cudaGetLastError right after it is 0.
from cuport_oracle import Buffer, Lcg

ID = "device-services/assert-001"
BLOCKS, THREADS = 2, 32
CUDA_ERROR_ASSERT = 710


def expected():
    n = BLOCKS * THREADS
    g = Lcg(ID)
    inputs = [g.next_u32() for _ in range(n)]
    g.next_u32()  # the bad index; the probe draws it, the oracle does not need it
    return {
        "tripped": Buffer("u32", [1]),
        "codes": Buffer("i32", [0, 0, CUDA_ERROR_ASSERT]),
        "out_a": Buffer("u32", [x ^ 0x5A5A5A5A for x in inputs]),
    }
