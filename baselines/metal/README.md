# baselines/metal/

Native Metal kernels that run the same behaviour as a CUDA probe directly on
the Apple GPU, with no CUDA translation layer in between.

When a tool's output differs from the oracle, the matrix has to decide
whether the tool mistranslated the probe or the hardware behaves differently.
If the hand-written Metal kernel matches the CUDA-semantics reference, the
hardware can do it and the divergence belongs to the tool.

Oracles stay as defined in `probes/README.md` (spec, CPU reference, or
committed vector); a baseline is never used as one. Baselines also produce no
result rows, since `results/` holds rows for tools only.

## Kernels

| Baseline id | Mirrors feature | Kernel |
|---|---|---|
| `metal/simd-shuffle-xor-001` | `warp-primitives.shfl_xor_width` | `kernels/simd_shuffle_xor.metal` |
| `metal/simd-ballot-001` | `warp-primitives.ballot_any_all` | `kernels/simd_ballot.metal` |
| `metal/threadgroup-reduce-001` | `shared-memory-barriers.syncthreads_reduction` | `kernels/threadgroup_reduce.metal` |

Each CPU reference in `run.m` encodes CUDA semantics, including a 32-wide
warp, so a GPU with a different simdgroup width fails the shuffle and ballot
baselines.

## Running

```
make check      # build the runner, self-test, run every baseline
make selftest   # FNV-1a and LCG checks only
```

The runner compiles the `.metal` sources at run time through the Metal
framework. It needs a Metal device and the Xcode Command Line Tools, and
works without full Xcode or the offline `metal` compiler. It writes no
`.metallib` or `.air` files.

For each baseline it prints the seed, grid size, threadgroup size and
simdgroup width, after a header with the device and macOS build, then a
classification from the `matrix/schema.md` enum. The exit status is non-zero
unless every baseline is `bit-exact`.

## Inputs

Inputs come from the 64-bit LCG in `matrix/schema.md`, seeded with the
FNV-1a hash of the baseline id. The schema does not say which bits to use;
this runner takes the high 32 bits of x₁, x₂, … and never emits the seed x₀.

## Adding a baseline

Put the kernel under `kernels/` and add a row with its CPU reference to
`BASELINES` in `run.m`. As with probes, each kernel checks one behaviour.
