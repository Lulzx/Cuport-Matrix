# Cuport-Matrix

An independent, cross-tool, cross-hardware **conformance and divergence
matrix for CUDA on Apple GPUs**.

Several projects now compile or run CUDA code on Apple silicon (CuMetal,
mvcc, MUDA, Bare CUDA Metal, MetaXuda, and others). Each one reports its own
pass counts on its own test set on its own hardware. Nobody runs the *same*
probes through *all* of them on the *same* machines and publishes where
they agree, where they diverge, and whether the divergence is documented.
Cuport-Matrix is that scoreboard.

Status: **research and scaffolding only**. The opportunity survey that
justifies this direction is in
[docs/survey-2026-11-15.md](docs/survey-2026-11-15.md). No probes or runner
code exist yet.

## What the matrix records

For every (feature, tool, hardware, date) cell, one of:

`bit-exact` · `within-tolerance` · `documented-divergence` ·
`undocumented-divergence` · `refused-with-diagnostic` · `silently-wrong` ·
`crash-hang` · `not-run`

Oracles are the CUDA specification, a CPU reference, or a committed reference
vector. No NVIDIA GPU is required to run the matrix.

## Hardware lanes

| Lane | Hardware id | Notes |
|---|---|---|
| Self-hosted | `m6-12c` | Apple M6 Mac mini, 12-core GPU. Primary target. |
| GitHub-hosted | `paravirtual` | `macos-26` runner, "Apple Paravirtual device". A feature subset of real silicon; rows are kept separate. |
| Community | `m1` … `m5-pro` | Rows submitted by PR with hardware evidence attached. |

## Repository layout

```
docs/        survey and design notes
matrix/      feature catalogue, tool registry, result schema
probes/      probe corpus (CUDA source + oracle + metadata)
baselines/   native Metal kernels that separate hardware from tool divergence
results/     committed result rows, results/<tool>/<hardware>/<date>.json
.github/     CI skeleton: hosted paravirtual lane, commented self-hosted M6 lane
```

## First milestone (v0)

Forty probes in eight families, run through CuMetal v0.8.0 and mvcc v1.3.0
on `m6-12c` and `paravirtual`, with a scoreboard and a divergence catalogue.
Scope, success criteria, schedule and fallback are in section 6 of the
survey.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Every claim carries an evidence
label; every divergence is reported upstream before it is catalogued.

## License

Apache-2.0. See [LICENSE](LICENSE).
