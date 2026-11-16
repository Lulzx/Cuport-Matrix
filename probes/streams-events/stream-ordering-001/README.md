# streams-events/stream-ordering-001

Verifies that CUDA kernel launches issued sequentially into the same non-default
stream execute in strict sequential order.

64 dependent kernel launches are issued into one non-default stream without any
intervening CPU or GPU synchronisation. Each launch `k` (0 <= k < 64) maps every
element `x` of a 65536-element `uint32_t` buffer to
`(x ^ (k * 0x9E3779B9)) * 0x01000193` (mod 2^32). The step does not commute
across `k`, so launches that ran out of order, or overlapped and raced on the
same buffer, produce a different result.

Following the correctness check, the probe runs 20 timed repetitions of the
64-launch chain using `cudaEventRecord` and `cudaEventElapsedTime` to measure
the execution time of the chain. Timing results are informational.
