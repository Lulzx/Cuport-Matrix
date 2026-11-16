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

Status as of 2026-11-16. ✅ = done and on `main`, ⏳ = in progress or
waiting on the other person, ❌ = not started. Owners are chosen by
hardware: Vivian runs the M6 (the only Apple lane) and two RTX 4090s; Guru has
the RTX 4500 Ada. All three NVIDIA cards are Ada (sm_89), so reference vectors
from any of them are comparable.

Guru's former items 5-7 (two-card check, fast-math, stream timings) were done
on Vivian's RTX 4090s and moved to Vivian's table.

### Vivian (M6 + RTX 4090)

| # | Task | Status |
|---|---|---|
| 1 | Runner skeleton, schema, adapters for CuMetal and mvcc | ✅ |
| 2 | Family `launch-geometry` (5 probes) | ✅ |
| 3 | Family `shared-memory-barriers` (5 probes) | ✅ |
| 4 | Family `warp-primitives` (6 probes) | ✅ |
| 5 | Family `atomics` (5 probes) | ✅ |
| 6 | First `m6-12c` rows for CuMetal and mvcc | ✅ |
| 7 | Re-run every probe on `m6-12c` for both tools. Done for all 35 probes on `main` (`096bf0b`); the 5 still to come from Guru's items 5-6 go through the M6 lane when they land | ✅ |
| 8 | Self-hosted M6 lane in `.github/workflows/matrix.yml` (`m6: true` on dispatch), running `.github/scripts/m6-lane.sh`. Lane and script done and tested from a clean clone; registering the runner needs a token from Guru (repo admin, Guru's item 11), steps in `docs/m6-runner.md` | ⏳ |
| 9 | Divergence catalogue (`docs/divergences.md`) and upstream issues for every `undocumented-divergence`. Catalogue done; issues filed as mvcc#20 and #21 and cuda-metal#205, #206 and #207, all triaged | ✅ |
| 10 | Stock `nvcc` adapter and the `refdrop` / `refcompare` runner commands, so NVIDIA reference drops are reproducible from the repo | ✅ |
| 11 | Two-card check: RTX 4090 drops (`reference/rtx4090-cuda12.6/`) for every probe with a reference. All 9 match the 4500 Ada drops, and all 14 match a second RTX 4090 | ✅ |
| 12 | `fp-semantics` fast-math probes `fast-sinf-001`, `fast-expf-001`, `fast-fdividef-001` (CPU reference, CUDA's documented bound as tolerance) with RTX 4090 drops | ✅ |
| 13 | Full-range sweep of `__sinf`, `__expf`, `__fdividef`, `sinf`, `expf` on the RTX 4090 (`tools/fastmath-sweep/`), summaries committed under `tools/fastmath-sweep/results/` | ⏳ |
| 14 | `streams-events` probes `stream-ordering-001` and `async-memcpy-overlap-001` with RTX 4090 reference timings (`timing ` lines in the drops) | ✅ |
| 15 | Review and merge Guru's family 7-8 PRs; Guru reviews Vivian's family 5-6 PRs. Families 7-8 merged in #4; families 5-6 went in with #2 and #3 without a separate review; work now goes straight to `main` | ✅ |

### Guru (RTX 4500 Ada)

| # | Task | Status |
|---|---|---|
| 1 | Family `device-services` (4 probes: `printf`, `clock64`, `assert`, device `malloc/free`) | ✅ |
| 2 | Family `limits-refusals` (5 probes: inline PTX, oversized shared, cooperative grid sync, texture object, dynamic parallelism; `expect: either`) | ✅ |
| 3 | Capture stock `nvcc` output on the 4500 Ada for every `limits-refusals` probe and commit it as the reference (`reference/rtx4500ada-cuda13.5/`; all five build and match their oracle bit-exact, so NVIDIA is the `expect: either` success arm) | ✅ |
| 4 | Reference vectors on the 4500 Ada for `device-services` (`printf` formatting, `assert` text) with `nvidia-smi` and driver version | ✅ |
| 5 | Family `fp-semantics`, remaining 5 probes: fma contraction, denormals, FP64 emulation, half/bf16, rounding modes (fast-math is done) | ❌ |
| 6 | Family `streams-events`, remaining 2 probes: event timing, default-stream sync | ❌ |
| 7 | 4500 Ada drops for the new `fp-semantics` and `streams-events` probes, checked with `refcompare` against `rtx4090-cuda12.6` | ❌ |
| 8 | Hosted `paravirtual` lane: replace the TODOs in `.github/workflows/matrix.yml` with a real run that writes `results/<tool>/paravirtual/<date>.json` | ❌ |
| 9 | Scoreboard generator (Markdown from `results/`) | ❌ |
| 10 | Hosted-vs-M6 disagreement rate reported as a number in the scoreboard | ❌ |
| 11 | Create the self-hosted runner registration token for the M6 (Settings → Actions → Runners; needs repo admin) and send it to Vivian, per `docs/m6-runner.md` | ❌ |

The 4500 Ada box runs CUDA 13.5 (nvcc V13.5.37) with MSVC 14.44 on driver
596.71. The RTX 4090 boxes run nvcc 12.6 (V12.6.20) with gcc 12.3 on drivers
595.99 and 580.178.

### Shared rules for NVIDIA reference drops

- Commit the probe's raw `CUPORT` output under `probes/<family>/<id>/reference/`
  with `nvidia-smi -q` driver and CUDA version, the `nvcc --version` line, and
  the exact command. Label the oracle `reference-vector` when the drop is the
  oracle.
- On Linux, write drops with `PYTHONPATH=runner python -m cuport refdrop --tag
  <gpu>-cuda<ver>` (set `CUPORT_NVCC` if `nvcc` is not on `PATH`) and check
  two cards with `python -m cuport refcompare --a <tag> --b <tag>`. Buffers
  must match bit-for-bit and printf lines as a multiset; `timing ` lines are
  ignored and stderr differences (paths, host compiler) are reported as notes.
- Fast-math probes keep a `cpu-reference` oracle with CUDA's documented error
  bound as the tolerance. NVIDIA output is not bit-exact against the correctly
  rounded value either, so a bit-exact NVIDIA oracle would fail tools that meet
  the documented bound. The NVIDIA drop records that real CUDA passes.
- `limits-refusals/dynamic-parallelism-001` needs `-rdc=true -lcudadevrt` on
  stock `nvcc`; `cooperative-grid-sync-001` needs `-std=c++17` (and
  `-Xcompiler /Zc:preprocessor` with MSVC) because CCCL 13 requires C++17.
  Record the flags with the reference drop (`meta.txt`).
- Layout: `reference/<gpu>-cuda<ver>/{output.txt,stderr.txt,build.log,meta.txt}`;
  `meta.txt` carries GPU, driver, nvcc, host compiler, exact command and exit codes.
- Never commit fatbins, PTX, SASS or `cuobjdump` output (legal note in
  `docs/survey-2026-11-15.md`, section 5).
- A reference vector from a 4090 and one from the 4500 Ada must match. If they
  do not, the probe is `observe_only` until the difference is explained.
