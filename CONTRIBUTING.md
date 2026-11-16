# Contributing to Cuport-Matrix

Cuport-Matrix is a conformance and divergence matrix for CUDA-on-Apple-GPU
tools. Contributions fall into three kinds: probes, result rows, and tool or
hardware registrations.

## Ground rules

- Every claim in `docs/` and every cell in `results/` carries an evidence
  label: `[source]`, `[official-doc]`, `[paper]`, `[self-reported]`,
  `[secondary]`, `[inferred]` or `[unverified]`. Do not upgrade a label
  without a link to the primary evidence.
- A probe must have an oracle that does not depend on an NVIDIA GPU: the CUDA
  specification, a CPU reference implementation, or a committed reference
  vector whose provenance is recorded.
- Divergences are reported upstream before they are catalogued as
  `undocumented-divergence`. Link the upstream issue in the result row.
- Never commit vendor binaries, fatbins or SASS dumps. See the legal note in
  `docs/survey-2026-11-15.md`, section 5.

## Adding a probe

1. Pick a family from `matrix/features.yaml` or propose a new one in a PR.
2. Create `probes/<family>/<probe-id>/` with `probe.cu`, `oracle.*`, and
   `probe.yaml` (see `probes/README.md`).
3. Run it through at least one registered tool and submit the result row.

## Adding a result row

Result rows are JSON files at `results/<tool>/<hardware>/<date>.json`
conforming to `matrix/schema.md`. Include the exact tool version, macOS build,
`system_profiler SPDisplaysDataType` output and the command used.

## Adding a tool or hardware target

Edit `matrix/tools.yaml` or the hardware list in `matrix/schema.md`. Pin a
tag or commit; unpinned "latest" entries are not accepted.

## Commit hygiene

Small, single-purpose commits. No generated files. Keep the matrix
reproducible: every row must be re-creatable from the committed probe,
tool pin and hardware description.

## Work split (milestone v0)

Status as of 2026-11-16. ✅ = done and merged on `main`, ❌ = not started or
not merged. Owners are chosen by hardware: Vivian runs the M6 (the only Apple
lane) and an RTX 4090; Guru owns the NVIDIA reference work on the RTX 4500
Ada. Both NVIDIA cards are Ada (sm_89), so reference vectors from either are
comparable.

### Vivian (M6 + RTX 4090)

| # | Task | Status |
|---|---|---|
| 1 | Runner skeleton, schema, adapters for CuMetal and mvcc | ✅ |
| 2 | Family `launch-geometry` (5 probes) | ✅ |
| 3 | Family `shared-memory-barriers` (5 probes) | ✅ |
| 4 | Family `warp-primitives` (6 probes) | ✅ |
| 5 | Family `atomics` (5 probes) | ✅ |
| 6 | First `m6-12c` rows for CuMetal and mvcc | ✅ |
| 7 | Re-run all 40 probes on `m6-12c` for both tools once families 5-8 land | ❌ |
| 8 | Self-hosted M6 runner registered and the commented lane in `.github/workflows/matrix.yml` enabled | ❌ |
| 9 | Divergence catalogue (`docs/`) and upstream issues for every `undocumented-divergence` | ❌ |
| 10 | Family `fp-semantics` (6 probes: fma contraction, fast-math, denormals, FP64 emulation, half/bf16, rounding modes) | ❌ |
| 11 | Family `streams-events` (4 probes: stream ordering, event timing, async copy overlap, default-stream sync) | ❌ |
| 12 | Hosted `paravirtual` lane: replace the TODOs in `.github/workflows/matrix.yml` with a real run that writes `results/<tool>/paravirtual/<date>.json` | ❌ |
| 13 | Scoreboard generator (Markdown from `results/`) | ❌ |
| 14 | Hosted-vs-M6 disagreement rate reported as a number in the scoreboard | ❌ |
| 15 | Review and merge Guru's family 7-8 PRs; Guru reviews Vivian's family 5-6 PRs | ❌ |

### Guru (RTX 4500 Ada)

| # | Task | Status |
|---|---|---|
| 1 | Family `device-services` (4 probes: `printf`, `clock64`, `assert`, device `malloc/free`) | ✅ |
| 2 | Family `limits-refusals` (5 probes: inline PTX, oversized shared, cooperative grid sync, texture object, dynamic parallelism; `expect: either`) | ✅ |
| 3 | Capture stock `nvcc` diagnostics on the 4500 Ada for every `limits-refusals` probe and commit them as the refusal reference | ❌ |
| 4 | Reference vectors on the 4500 Ada for `device-services` (`printf` formatting, `assert` text) with `nvidia-smi` and driver version | ❌ |
| 5 | Fast-math and libdevice ulp sweeps: full-range `__sinf`, `__expf`, `__fdividef` outputs committed as `reference-vector` oracles | ❌ |
| 6 | Run every reference vector on two NVIDIA cards and confirm they agree bit-for-bit before committing (run-to-run determinism check) | ❌ |
| 7 | `async_memcpy_overlap` and `stream_ordering` reference timings on NVIDIA, so the oracle has a documented NVIDIA baseline | ❌ |

Items 3-7 need the CUDA toolkit (`nvcc`) and MSVC on the 4500 Ada machine;
the driver (596.71) is present but no compiler is.

### Shared rules for NVIDIA reference drops

- Commit the probe's raw `CUPORT` output under `probes/<family>/<id>/reference/`
  with `nvidia-smi -q` driver and CUDA version, the `nvcc --version` line, and
  the exact command. Label the oracle `reference-vector`.
- `limits-refusals/dynamic-parallelism-001` needs `-rdc=true -lcudadevrt` on
  stock `nvcc`; record both flags with its reference drop.
- Never commit fatbins, PTX, SASS or `cuobjdump` output (legal note in
  `docs/survey-2026-11-15.md`, section 5).
- A reference vector from a 4090 and one from the 4500 Ada must match. If they
  do not, the probe is `observe_only` until the difference is explained.
