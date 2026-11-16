# shared-memory-barriers/syncthreads-count-001

`__syncthreads_count`, `__syncthreads_and` and `__syncthreads_or` across
256 blocks of 128 threads (four SIMD groups each). The predicates are true
with probability 1/8, 255/256 and 1/4096 per thread, so `count` varies, and
both `all` and `any` come out true for some blocks and false for others. A
translation that reduces over one SIMD group instead of the whole block
gives counts that are too small and `all`/`any` results that are wrong.
