# streams-events/async-memcpy-overlap-001

Verifies pipelined asynchronous data transfers and kernel execution across two non-default streams using page-locked (pinned) host memory.

16 MiB of `uint32_t` host memory (4,194,304 elements) is allocated in pinned memory and divided into 8 chunks of 524,288 elements (2 MiB each). For each chunk `c` (0 <= c < 8), a host-to-device transfer (`cudaMemcpyAsync`), an arithmetic transform kernel (`y = x * 2654435761 + c`), and a device-to-host transfer (`cudaMemcpyAsync`) are issued into stream `c % 2`.

Following a full device synchronization (`cudaDeviceSynchronize`), the host verifies that all 4,194,304 output elements match their expected values and emits `count_wrong` (which must be 0) along with a 4096-element strided sample `sample` (every 1024th element).

The probe then benchmarks 20 repetitions of the same 8-chunk pipeline using a single stream (serial) and using two concurrent streams. Overlap is not guaranteed by CUDA (concurrency depends on device copy engines, hardware capabilities, and driver scheduling), and all timing lines are informational.
