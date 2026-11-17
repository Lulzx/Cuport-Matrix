# probes/

Probe corpus for the Cuport-Matrix conformance and divergence matrix.

## Layout

```
probes/
  common/
    cuport.h                # input stream, output protocol, CUDA error check
    cuport_oracle.py        # the same stream and encodings for oracles
  <family>/                 # one of the families in matrix/features.yaml
    <name>-NNN/             # probe id is "<family>/<name>-NNN"
      probe.cu              # the CUDA source under test (plain CUDA C++)
      oracle.py             # expected() -> dict[str, Buffer]; spec citations in comments
      probe.yaml            # metadata, below
      README.md             # what the probe checks and why it is interesting
```

`probe.cu` includes the shared header as `#include "../../common/cuport.h"`.

## Output protocol

A probe prints its results with the `cuport_emit_*` helpers, one line per
buffer, and finishes with `return cuport_done();`:

```
CUPORT <name> <dtype> <count> <hex> <hex> ...
CUPORT-DONE
```

`dtype` is `u16`, `u32`, `i32`, `u64`, `f32` or `f64`. Floats are emitted
as raw bits, so the comparison never depends on printf formatting. A run
without `CUPORT-DONE` is a crash. Wrap every runtime call in `CUPORT_CHECK`,
which prints `CUPORT-ERROR` and exits 3. Any other stdout line is device
`printf` text, which an oracle can match with `TEXT_LINES`. Lines on stdout
starting with `timing ` are informational: never compared by `refcompare`,
ignored by `classify` (oracles for these probes define no `TEXT_LINES`).

Inputs come from `CuportLcg g("<probe id>")` on the host. The oracle draws the
same values with `cuport_oracle.Lcg("<probe id>")`.

## probe.yaml

```yaml
id: warp-primitives/shfl-xor-001
family: warp-primitives
features: [shfl_xor_width]       # ids from matrix/features.yaml
oracle: cpu-reference            # spec | cpu-reference | reference-vector
tolerance: bit-exact             # default for every buffer: bit-exact | abs:<x> | rel:<x> | ulp:<n>
outputs:                         # optional per-buffer overrides
  y: {tolerance: "ulp:2"}
expect: output                   # output | refusal | either
timeout_s: 60
observe_only: false              # true for undefined behaviour: record, never pass or fail
```

`expect: refusal` is for `limits-refusals` probes whose correct outcome is a
diagnostic at compile or run time. `either` accepts a correct result or a
diagnostic.

## Probe rules

- Deterministic inputs. Use the fixed-seed LCG described in
  `matrix/schema.md`; never `rand()` or time-based seeds.
- One behaviour per probe. A probe that checks shuffles and atomics is two
  probes.
- Every probe states its oracle class: `spec`, `cpu-reference`, or
  `reference-vector`.
- Every probe states a tolerance: `bit-exact`, an absolute or relative
  epsilon, or `ulp:<n>`.
- Probes must compile with stock `nvcc` syntax; no tool-specific extensions.

## Reference drops

Reference drops capture raw NVIDIA `nvcc` outputs to validate determinism
across hardware and ground `reference-vector` oracles. Drops live at:

```
probes/<family>/<id>/reference/<gpu>-cuda<ver>/
  output.txt
  stderr.txt
  build.log
  meta.txt
```

Two runner commands manage drops:

```bash
# Capture reference drops using stock nvcc
python -m cuport refdrop --tag <gpu>-cuda<ver> [--probe PREFIX]... [--force]

# Compare two sets of reference drops across probes that have both
python -m cuport refcompare --a TAG --b TAG [--probe PREFIX]...
```

`refdrop` runs selected probes through the `nvcc` adapter, captures build and
run outputs, writes `meta.txt` with environment metadata, and validates each
probe against its oracle (exiting 1 on unexpected divergence). `refcompare`
checks agreement of build exit, run exit, `CUPORT-DONE`, `CUPORT-ERROR`, buffer
contents, and device text (ignoring `timing ` lines and treating stderr
differences as platform notes).

### Rules for NVIDIA reference drops

- A drop holds the probe's raw `CUPORT` output, the driver and CUDA version,
  the `nvcc --version` line, the host compiler and the exact command
  (`meta.txt`). Label the oracle `reference-vector` when the drop is the oracle.
- Set `CUPORT_NVCC` if `nvcc` is not on `PATH`. On Windows the adapter finds
  MSVC through `vswhere` (override with `CUPORT_NVCC_CCBIN`) and passes
  `-Xcompiler /Zc:preprocessor`, which CCCL 13 needs.
- Buffers must match bit-for-bit across cards and printf lines as a multiset;
  `timing ` lines are ignored and stderr differences (paths, host compiler)
  are reported as notes.
- Drops from two cards of the same architecture must match. If they do not,
  the probe is `observe_only` until the difference is explained.
- Fast-math probes keep a `cpu-reference` oracle with CUDA's documented error
  bound as the tolerance: NVIDIA output is not correctly rounded either, so a
  bit-exact NVIDIA oracle would fail tools that meet the documented bound. The
  drop records that real CUDA passes.
- `limits-refusals/dynamic-parallelism-001` needs `-rdc=true -lcudadevrt` on
  stock `nvcc`, and the drop's `meta.txt` records it.
- Never commit fatbins, PTX, SASS or `cuobjdump` output.

Committed drops: `rtx4500ada-cuda13.5` (RTX 4500 Ada, nvcc 13.5, MSVC 14.44,
driver 596.71) and `rtx4090-cuda12.6` (two RTX 4090s, nvcc 12.6, gcc 12.3,
drivers 595.99 and 580.178). All are sm_89; every pair compared agrees.

## Families (milestone v0)

1. `launch-geometry`
2. `shared-memory-barriers`
3. `warp-primitives`
4. `atomics`
5. `fp-semantics`
6. `streams-events`
7. `device-services`
8. `limits-refusals`
