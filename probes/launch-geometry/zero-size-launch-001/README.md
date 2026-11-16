# launch-geometry/zero-size-launch-001

`touch<<<0, 32>>>` must fail at launch with `cudaErrorInvalidConfiguration`
(9), must not write anything, and must not poison the context: a following
`<<<1, 32>>>` launch returns `cudaSuccess` and writes its 32 words.
`zero_launch_failed` separates "failed with a different code" (only `codes`
mismatches) from "did not fail at all" (both mismatch).
