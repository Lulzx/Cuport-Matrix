# limits-refusals/inline-ptx-001

A kernel that reads the PTX special register `%laneid` through an
`asm volatile` statement. This is valid nvcc input and something no
Metal-targeting tool can lower, so the probe is `expect: either`: a compile
diagnostic that names the asm statement is the correct outcome and is
recorded as `refused-with-diagnostic`.

The interesting failure modes are the other two. A tool that silently drops
unrecognised asm leaves `l` uninitialised and lands in `silently-wrong`. A
tool that pattern-matches common PTX snippets has to produce
`threadIdx.x % 32` for every thread to pass, which the oracle checks.
