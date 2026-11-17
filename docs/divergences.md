# Divergence catalogue: m6-12c

Every cell where mvcc or CuMetal does not match CUDA on the Apple M6
(`m6-12c`, 12-core GPU, macOS 27.2). Source rows:
`results/mvcc/m6-12c/2026-11-17.json` and
`results/cumetal/m6-12c/2026-11-17.json`, both run from repo commit
`49718db` with mvcc v1.3.0 (`8a814c2`) and CuMetal v0.8.0 (`6651796`). The
corpus has 42 probes.

| | bit-exact | within tolerance | documented divergence | undocumented divergence | refused with diagnostic | silently wrong |
|---|---|---|---|---|---|---|
| mvcc | 29 | 2 | 1 | 2 | 8 | 0 |
| CuMetal | 27 | 2 | 3 | 3 | 5 | 2 |

Each entry was checked against the tool's source at its pinned commit, so
none of them is a probe bug. Every undocumented divergence has an upstream
issue, recorded in `matrix/triage.yaml`. The two silently wrong CuMetal cells
are undocumented too; they stay in that class until their issues are filed.

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

Undocumented divergence in both, filed as
[doximity/mvcc#21](https://github.com/doximity/mvcc/issues/21) and
[Lulzx/cuda-metal#205](https://github.com/Lulzx/cuda-metal/issues/205).

### `warp-primitives/shfl-xor-width-001`: CuMetal

With `width < 32`, `__shfl_xor_sync` returns the caller's own value whenever
the partner lane is in another group. CUDA only does that for later groups;
a lane may read from an earlier one. CuMetal sends `lane ^ laneMask` through
its indexed shuffle, which wraps the source into the caller's group. 2560 of
the probe's outputs differ. mvcc is bit-exact. Undocumented divergence, filed as
[Lulzx/cuda-metal#206](https://github.com/Lulzx/cuda-metal/issues/206).

### `launch-geometry/zero-size-launch-001`: CuMetal

A zero grid or block dimension returns `cudaErrorInvalidValue` (1) instead
of `cudaErrorInvalidConfiguration` (9). No kernel runs, so only the error
code is wrong. mvcc returns 9. Undocumented divergence, filed as
[Lulzx/cuda-metal#207](https://github.com/Lulzx/cuda-metal/issues/207).

### `fp-semantics/fp64-emulation-001`: CuMetal

`double` add, multiply, divide, square root and `fma` miss the correctly
rounded result in the last few bits. CuMetal's default FP64 mode, `fast48`,
is FP32-pair arithmetic with about 48 significand bits.
[Documented](https://github.com/Lulzx/cuda-metal/blob/v0.8.0/docs/fp64-policy.md?plain=1#L17-L34);
rebuilt with `--fp64=ieee64`, every buffer is bit-exact. mvcc refuses this
probe (below).

### `fp-semantics/denormals-001`: CuMetal

Double subnormals come back as zero: `d_half` and `d_scale` are wrong in
every element, while all three float buffers are bit-exact. `fast48` keeps
the binary32 exponent range, which has no room for them. Documented in the
same FP64 policy, and bit-exact with `--fp64=ieee64`. mvcc is bit-exact.
The triage entry is limited to `m6-12c`, because on the paravirtual GPU
CuMetal also flushes float subnormals, which the FP64 policy does not cover.

### `fp-semantics/fma-contraction-001`: CuMetal

`__fadd_rn(__fmul_rn(a, b), c)` is fused into one FMA, so all 4096 `split`
values equal the `fmaf` result. CUDA guarantees these intrinsics are never
contracted. CuMetal's NVVM importer maps `__nv_fmul_rn` and `__nv_fadd_rn` to
its ordinary multiply and add opcodes (`compiler/ir/src/nvvm_importer.cpp`,
lines 1966-1969), after which the backend contracts them like `a * b + c`.
The default contraction of `a * b + c` itself matches CUDA. mvcc is
bit-exact. Undocumented; issue drafted, not filed yet.

### `streams-events/event-timing-001`: CuMetal

`cudaEventElapsedTime` returns `cudaErrorInvalidValue` (1) for an event that
was never recorded and for one created with `cudaEventDisableTiming`. CUDA
returns `cudaErrorInvalidResourceHandle` (400) for both, as do mvcc and the
RTX 4500 Ada drop. Every timing check passes. Undocumented; issue drafted,
not filed yet.

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
| `fp-semantics/fp64-emulation-001` | Compile error: `__nv_fma` is not emulated; mvcc's software binary64 covers `+ - * /`, `sqrt`, rounding, comparisons and conversions | (wrong result, above) |
| `fp-semantics/half-bf16-001` | bit-exact | Compile error: `cuda_bf16.h` not found |
| `fp-semantics/rounding-modes-001` | Compile error: unsupported libdevice function `__nv_fadd_rz` | bit-exact |
| `limits-refusals/texture-object-001` | Compile error: no `tex1Dfetch`; [README](https://github.com/doximity/mvcc/blob/v1.3.0/README.md?plain=1#L60) says textures are not supported | bit-exact |
| `shared-memory-barriers/syncthreads-count-001` | (wrong result, above) | Compile error: unsupported intrinsic `llvm.nvvm.barrier.cta.red.popc` |

## Upstream issues

| Tool | Probe | Issue |
|---|---|---|
| mvcc | `syncthreads-count-001` | [#20](https://github.com/doximity/mvcc/issues/20), open |
| mvcc | `oversized-shared-001` | [#21](https://github.com/doximity/mvcc/issues/21), open |
| CuMetal | `oversized-shared-001` | [#205](https://github.com/Lulzx/cuda-metal/issues/205), open |
| CuMetal | `shfl-xor-width-001` | [#206](https://github.com/Lulzx/cuda-metal/issues/206), open |
| CuMetal | `zero-size-launch-001` | [#207](https://github.com/Lulzx/cuda-metal/issues/207), open |
| CuMetal | `fma-contraction-001` | drafted |
| CuMetal | `event-timing-001` | drafted |

When a tool fixes one of these, its raw class changes, the triage entry stops
matching, and the cell falls back to whatever the runner measures.
