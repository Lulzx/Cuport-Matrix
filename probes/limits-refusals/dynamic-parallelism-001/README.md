# limits-refusals/dynamic-parallelism-001

A parent kernel of 4 blocks launches one child block per parent block from
device code (`child<<<1, 64>>>` inside `parent`). The host synchronises on
the parent only; under CDP2 that also waits for the children, so
`child_out` holds every child's writes.

Metal has no device-side launch, so the expected tool outcome is a compile
diagnostic at the device-side `<<<>>>`, recorded as `refused-with-diagnostic`.
A tool that compiles the launch as a plain device function call is
observable here: the child would then run as 64 ordinary calls from thread 0
of each block, and the output is still correct, which is an acceptable
emulation; a tool that drops the launch leaves zeros and is `silently-wrong`.

Note for NVIDIA reference drops: stock `nvcc` requires `-rdc=true` and the
device runtime (`-lcudadevrt`) for this probe. The adapters in `runner/`
pass neither, because no CUDA-on-Metal tool accepts them; record the flags
with the reference drop.
