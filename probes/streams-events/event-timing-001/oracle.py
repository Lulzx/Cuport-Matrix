# CUDA Runtime API, "Event Management":
#   cudaEventQuery: cudaSuccess once all work captured by the event has completed.
#   cudaEventElapsedTime: "Computes the elapsed time between two events (in
#     milliseconds with a resolution of around 0.5 microseconds)." "If
#     cudaEventRecord() has not been called on either event, then
#     cudaErrorInvalidResourceHandle is returned." "If either event was created
#     with the cudaEventDisableTiming flag, then this function will return
#     cudaErrorInvalidResourceHandle."
# An interval from an event to itself is zero; intervals between events recorded
# in order in one stream around kernels are positive, grow monotonically, and add.
from cuport_oracle import Buffer

CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_RESOURCE_HANDLE = 400

CHECKS = [
    CUDA_SUCCESS,                        # 0 cudaEventQuery on a completed event
    CUDA_SUCCESS,                        # 1 cudaEventElapsedTime(e0, e0) succeeds
    1,                                   # 2 ... and is exactly 0 ms
    1,                                   # 3 each interval around a kernel is > 0
    1,                                   # 4 t01 <= t02 <= t03
    1,                                   # 5 t01 + t12 + t23 == t03 within 2 us
    CUDA_ERROR_INVALID_RESOURCE_HANDLE,  # 6 end event never recorded
    CUDA_ERROR_INVALID_RESOURCE_HANDLE,  # 7 end event created with cudaEventDisableTiming
]


def expected():
    return {"checks": Buffer("u32", CHECKS)}
