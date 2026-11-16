# streams-events/stream-ordering-001

Verifies that CUDA kernel launches issued sequentially into the same non-default
stream execute in strict sequential order.

64 dependent kernel launches are issued into one non-default stream without any
intervening CPU or GPU synchronisation. Each launch `k` (0 <= k < 64) adds
`k * 0x9E3779B9` to every element of a 65536-element `uint32_t` buffer.
Because each kernel reads and writes the same buffer in-place, executing launches
out of order or concurrently would introduce write-after-read and write-after-write
data races, producing corrupted buffer values.

Following the correctness check, the probe runs 20 timed repetitions of the
64-launch chain using `cudaEventRecord` and `cudaEventElapsedTime` to measure
the execution time of the chain. Timing results are informational.
