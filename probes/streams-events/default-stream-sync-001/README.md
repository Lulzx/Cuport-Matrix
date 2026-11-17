# streams-events/default-stream-sync-001

Checks the implicit barriers of the legacy default stream. Three kernels are
launched with no explicit synchronisation between them:

| Kernel | Stream | Reads | Writes |
|---|---|---|---|
| `stage_a` | `s` (blocking, from `cudaStreamCreate`) | `in` | `x = mix(in, 4096)` |
| `stage_b` | legacy default stream `0` | `x` | `y = mix(x ^ 0x9E3779B9, 4096)` |
| `stage_c` | `s` | `y` | `z = y ^ 0xA5A5A5A5` |

`stage_b` is correct only if stream 0 waited for `stage_a`, and `stage_c` only
if stream `s` waited for `stage_b`. The results are read back with plain
`cudaMemcpy`, which runs in stream 0 and so waits for `stage_c`. `x`, `y` and
`z` start zeroed. `mix` repeats `x -> (x ^ (x >> 15)) * 0x2C1B3C6D + r`. The
first two kernels do 4096 dependent rounds per element on a deliberately small
grid (8 blocks of 256 threads, grid-stride), so the GPU is mostly idle while
they run and a consumer that did not wait would be scheduled at once and read
zeros or partial results.

65536 elements per buffer, tolerance bit-exact; the oracle is a vectorised
numpy model of the three kernels.

Compiling with `--default-stream per-thread` removes these barriers, so a tool
whose default stream is per-thread, or which treats stream 0 as an ordinary
stream, can fail this probe. A tool that serialises every stream passes it.

The probe can only catch a missing barrier on a device that really runs
streams concurrently. On the RTX 4500 Ada (Windows, WDDM driver 596.71) the
same source built with `--default-stream per-thread`, or with `s` made
`cudaStreamNonBlocking`, still produced correct results in every run
[reproduced 2026-11-17], because the driver serialised the kernels. The same
WDDM box shows no copy/compute overlap in `async-memcpy-overlap-001` either.
A negative control on one of the Linux RTX 4090 boxes is still to be run.
