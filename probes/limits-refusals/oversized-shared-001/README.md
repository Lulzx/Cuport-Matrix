# limits-refusals/oversized-shared-001

One launch of a 256-thread block that asks for 64 KB of dynamic shared
memory without the `cudaFuncSetAttribute` opt-in. CUDA refuses it at launch
with `cudaErrorInvalidValue` (1), the error is not sticky, and nothing runs,
so the output keeps its `0xAB` memset pattern.

`refused` separates "refused with a different code" (only `codes`
mismatches, usually a `documented-divergence` candidate for triage) from
"ran anyway" (`refused` and `untouched` both mismatch, which is
`silently-wrong`: the tool gave the kernel less shared memory than it asked
for). The companion `shared-memory-barriers/shared-32kb-001` probes the
Apple 32 KB boundary with a valid CUDA request; this one checks that an
invalid request is reported as such.
