# Divergence catalogue: m6-12c

Every cell where mvcc or CuMetal does not match CUDA on the Apple M6
(`m6-12c`, 12-core GPU, macOS 27.2). Source rows:
`results/mvcc/m6-12c/2026-11-16.json` and
`results/cumetal/m6-12c/2026-11-16.json`, both run from repo commit
`0474ead` with mvcc v1.3.0 (`8a814c2`) and CuMetal v0.8.0 (`6651796`). The
corpus has 35 probes.

| | bit-exact | within tolerance | documented divergence | undocumented divergence | refused with diagnostic | silently wrong |
|---|---|---|---|---|---|---|
| mvcc | 24 | 2 | 1 | 1 | 6 | 1 |
| CuMetal | 25 | 2 | 1 | 0 | 4 | 3 |

Each entry was checked against the tool's source at its pinned commit, so
none of them is a probe bug. "Silently wrong" stays the class until the
upstream issue exists; triage then moves it to undocumented divergence.

## Wrong results

### `shared-memory-barriers/syncthreads-count-001`: mvcc

`__syncthreads_count`, `__syncthreads_and` and `__syncthreads_or` compile to
a plain barrier, and the result variable is never assigned, so the kernel
stores whatever the output word already held. The README says unsupported
intrinsics are compile errors. Undocumented divergence, filed as
[doximity/mvcc#20](https://github.com/doximity/mvcc/issues/20). CuMetal
refuses this probe at compile time instead.

### `device-services/printf-001`: mvcc

Device `printf` compiles, returns 0 and prints nothing.
[Documented](https://github.com/doximity/mvcc/blob/v1.3.0/README.md?plain=1#L111):
Metal has no counterpart to CUDA's printf buffer.

### `device-services/assert-001`: CuMetal

A failing device `assert` neither stops the kernel nor makes the next
synchronize return `cudaErrorAssert` (710); both calls return 0. The CUDA
headers give `__assert_rtn` an empty `__device__` overload.
[Documented](https://github.com/Lulzx/cuda-metal/blob/v0.8.0/CHANGELOG.md?plain=1#L600-L602)
in the changelog, and `docs/known-gaps/runtime.md` lists exact
`cudaErrorAssert` as not implemented. mvcc refuses this probe at compile time.

### `limits-refusals/oversized-shared-001`: mvcc and CuMetal

A launch asking for 64 KB of dynamic shared memory without raising
`cudaFuncAttributeMaxDynamicSharedMemorySize` succeeds and the kernel runs.
CUDA rejects anything over 48 KB without that opt-in with
`cudaErrorInvalidValue`, as both NVIDIA reference drops show. The two tools
get there differently:

- mvcc moves shared memory above the M6's 32 KB into a device-memory pool,
  which is documented. Its runtime defines the 48 KB limit
  (`SMEM_DEFAULT_LIMIT` in `crates/mvcc-cudart/src/api.rs`) but only reports
  it through `cudaFuncGetAttributes`; the launch never checks it.
- CuMetal passes the request straight to `setThreadgroupMemoryLength`. With
  `MTL_DEBUG_LAYER=1` the launch aborts with
  `length(65536) must be <= 32768`.

Upstream issue drafted for each tool, not filed yet.

### `warp-primitives/shfl-xor-width-001`: CuMetal

With `width < 32`, `__shfl_xor_sync` returns the caller's own value whenever
the partner lane is in another group. CUDA only does that for later groups;
a lane may read from an earlier one. CuMetal sends `lane ^ laneMask` through
its indexed shuffle, which wraps the source into the caller's group. 2560 of
the probe's outputs differ. mvcc is bit-exact. Upstream issue drafted, not
filed.

### `launch-geometry/zero-size-launch-001`: CuMetal

A zero grid or block dimension returns `cudaErrorInvalidValue` (1) instead
of `cudaErrorInvalidConfiguration` (9). No kernel runs, so only the error
code is wrong. mvcc returns 9. Upstream issue drafted, not filed.

## Passes with a caveat

- `shared-memory-barriers/shared-32kb-001` is bit-exact on CuMetal, but its
  32772-byte launch exceeds the M6's `maxThreadgroupMemoryLength` (32768).
  Under `MTL_DEBUG_LAYER=1` it hits the same assertion as the 64 KB case. The
  release driver happens to tolerate it. mvcc runs it from the spill pool and
  passes validation.
- The fast-math probes are within CUDA's documented error bounds on both
  tools: `fast-expf-001` has 1199 (mvcc) and 1127 (CuMetal) elements off by
  at most 2 ulp, `fast-sinf-001` has 1708 elements within the absolute bound
  on both, and `fast-fdividef-001` is bit-exact on both. Real CUDA is not
  bit-exact against the correctly rounded value either.

## Refusals

A refusal means the tool stopped with a diagnostic, so nothing ran wrong.
Cells not listed are bit-exact or appear above.

| Probe | mvcc | CuMetal |
|---|---|---|
| `device-services/assert-001` | Compile error: no device overload of `__assert_rtn`, so the macOS SDK's host `assert` is used | (wrong result, above) |
| `device-services/clock64-001` | Compile error: unsupported intrinsic `clock64` | bit-exact |
| `device-services/malloc-free-001` | Compile error: no device `malloc` | bit-exact |
| `device-services/printf-001` | (wrong result, above) | Compile error: typed `vprintf` needs tightly packed 32/64-bit arguments, and the probe mixes `int`, `double` and `long long` |
| `limits-refusals/cooperative-grid-sync-001` | Compile error: no `cooperative_groups.h`; [README](https://github.com/doximity/mvcc/blob/v1.3.0/README.md?plain=1#L113) says grid sync is not available | Runtime error 720: CuMetal reports one multiprocessor and caps cooperative grids at 4 blocks; the probe launches 8. CUDA may refuse a grid that cannot be co-resident, so this is allowed |
| `limits-refusals/dynamic-parallelism-001` | Compile error: no `cudaLaunchDevice`; [README](https://github.com/doximity/mvcc/blob/v1.3.0/README.md?plain=1#L113) says dynamic parallelism is not available | Compile error: device launch needs relocatable device code, which `cumetalc` does not support (`docs/known-gaps/compiler.md`) |
| `limits-refusals/texture-object-001` | Compile error: no `tex1Dfetch`; [README](https://github.com/doximity/mvcc/blob/v1.3.0/README.md?plain=1#L60) says textures are not supported | bit-exact |
| `shared-memory-barriers/syncthreads-count-001` | (wrong result, above) | Compile error: unsupported intrinsic `llvm.nvvm.barrier.cta.red.popc` |

## Upstream issues

| Tool | Probe | Issue |
|---|---|---|
| mvcc | `syncthreads-count-001` | [#20](https://github.com/doximity/mvcc/issues/20), open |
| mvcc | `oversized-shared-001` | drafted |
| CuMetal | `oversized-shared-001` | drafted |
| CuMetal | `shfl-xor-width-001` | drafted |
| CuMetal | `zero-size-launch-001` | drafted |

Once an issue is filed, its cell gets a `matrix/triage.yaml` entry with the
issue URL and the rows are regenerated.
