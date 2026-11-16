# shared-memory-barriers/syncthreads-reduction-001

The textbook shared-memory tree reduction over 512-thread blocks. Every step
after the first five reads values written by another SIMD group, so a barrier
that only orders a single SIMD group, or one dropped inside the loop, gives
wrong sums. Inputs come from the LCG stream; sums wrap mod 2^32.
`baselines/metal/kernels/threadgroup_reduce.metal` is the same algorithm in
native Metal.
