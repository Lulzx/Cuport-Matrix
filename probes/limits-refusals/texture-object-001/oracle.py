# CUDA C++ Programming Guide, "Texture Object API": tex1Dfetch on a linear
# resource with cudaReadModeElementType returns element i of the buffer
# unchanged, with no filtering or normalisation. Expected: in[(i*7) % n].
from cuport_oracle import Buffer, lcg_u32

ID = "limits-refusals/texture-object-001"
BLOCKS, THREADS = 4, 64


def expected():
    n = BLOCKS * THREADS
    v = lcg_u32(ID, n)
    return {"gathered": Buffer("u32", [v[(i * 7) % n] for i in range(n)])}
