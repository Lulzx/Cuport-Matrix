# CUDA C++ Programming Guide, "Shared Memory" / cudaFuncSetAttribute: a kernel
# may use at most 48 KB of dynamic shared memory unless
# cudaFuncAttributeMaxDynamicSharedMemorySize is raised first. A launch over
# that limit fails with cudaErrorInvalidValue (1), is not sticky, and runs no
# threads, so the output buffer keeps its memset pattern.
from cuport_oracle import Buffer

THREADS = 256
CUDA_ERROR_INVALID_VALUE = 1


def expected():
    return {
        "refused": Buffer("u32", [1]),
        "codes": Buffer("i32", [CUDA_ERROR_INVALID_VALUE, 0]),
        "untouched": Buffer("u32", [0xABABABAB] * THREADS),
    }
