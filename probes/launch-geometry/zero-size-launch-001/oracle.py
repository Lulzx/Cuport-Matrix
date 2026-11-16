# CUDA Runtime API, cudaError: cudaErrorInvalidConfiguration = 9, returned for a
# launch whose configuration is invalid; a zero-sized grid is invalid. The error
# is not sticky, so the following launch succeeds (cudaSuccess = 0).
from cuport_oracle import Buffer

CUDA_ERROR_INVALID_CONFIGURATION = 9


def expected():
    return {
        "zero_launch_failed": Buffer("u32", [1]),
        "codes": Buffer("i32", [CUDA_ERROR_INVALID_CONFIGURATION, 0, 0]),
        "after_zero": Buffer("u32", [0xABABABAB] * 32),
        "after_one": Buffer("u32", [0x600DF00D] * 32),
    }
