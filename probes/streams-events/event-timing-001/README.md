# streams-events/event-timing-001

Checks the parts of `cudaEventElapsedTime` and `cudaEventQuery` that do not
depend on how fast the GPU is. Four events `e0..e3` are recorded in one
stream around three kernel launches (each 256 dependent integer rounds over
1M elements, after an untimed warm-up launch). The probe emits one `u32`
buffer, `checks`:

| # | Check | Expected |
|---|---|---|
| 0 | `cudaEventQuery(e3)` after it completed | `cudaSuccess` (0) |
| 1 | `cudaEventElapsedTime(e0, e0)` | `cudaSuccess` (0) |
| 2 | ... and its value | exactly 0 ms (1 = true) |
| 3 | `t01`, `t12`, `t23` | all > 0 (1 = true) |
| 4 | `t01 <= t02 <= t03` | 1 |
| 5 | `abs(t01 + t12 + t23 - t03)` | within 2 us plus 0.01 % (1 = true) |
| 6 | end event never recorded | `cudaErrorInvalidResourceHandle` (400) |
| 7 | end event created with `cudaEventDisableTiming` | `cudaErrorInvalidResourceHandle` (400) |

The error codes and both flag behaviours are quoted from the Runtime API
reference in `oracle.py`. The 2 us additivity bound is four times the
documented 0.5 us resolution.

The measured intervals are printed on a `timing events_ms ...` line, which is
informational: never compared by `refcompare` and ignored by `classify`.
