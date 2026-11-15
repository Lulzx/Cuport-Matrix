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
