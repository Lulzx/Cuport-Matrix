# Result schema

A result row answers one question: **for feature F, through tool T, on
hardware H, at date D, what happened?**

## Coordinates

| Axis | Source of truth | Example |
|---|---|---|
| feature | `matrix/features.yaml` (`family.feature_id`) | `warp-primitives.shfl_xor_width` |
| tool | `matrix/tools.yaml` (`id` + pinned `version`) | `cumetal@v0.8.0` |
| hardware | hardware list below | `m6-12c` |
| date | ISO 8601, UTC | `2026-12-01` |

## Hardware identifiers

| id | Description | Notes |
|---|---|---|
| `paravirtual` | GitHub-hosted macOS arm64 runner, "Apple Paravirtual device" | feature subset; no simdgroup matrix, no bf16, 32 KB threadgroup memory |
| `m1`, `m1-pro`, `m1-max` | Apple M1 family | community rows |
| `m2`, `m2-max`, `m3`, `m3-max`, `m4`, `m4-pro`, `m5`, `m5-pro` | Apple M2–M5 families | community rows |
| `m6-12c` | Apple M6, 12-core GPU (Mac mini) | primary validation target |

New identifiers are added by PR with `system_profiler SPDisplaysDataType`
output attached.

## Classification enum

| Class | Meaning |
|---|---|
| `bit-exact` | Output matches the oracle bit-for-bit |
| `within-tolerance` | Matches within the probe's declared tolerance |
| `documented-divergence` | Differs, and the tool's own docs say it will |
| `undocumented-divergence` | Differs, and the tool's docs do not say so (upstream issue required) |
| `refused-with-diagnostic` | Tool rejected the probe with an actionable message |
| `silently-wrong` | Compiled and ran, produced wrong output, no diagnostic |
| `crash-hang` | Compiler or runtime crashed, or the probe exceeded its timeout |
| `not-run` | Cell not yet exercised |

## Row format

One JSON file per `results/<tool>/<hardware>/<date>.json`:

```json
{
  "schema": 1,
  "tool": {"id": "cumetal", "version": "v0.8.0", "commit": "<sha>"},
  "hardware": {"id": "m6-12c", "macos": "27.2 (26B123)", "profiler": "<system_profiler excerpt>"},
  "date": "2026-12-01",
  "runner": {"command": "<exact invocation>", "duration_s": 1234},
  "cells": [
    {
      "feature": "warp-primitives.shfl_xor_width",
      "probe": "warp-primitives/shfl-xor-001",
      "class": "within-tolerance",
      "oracle": "cpu-reference",
      "tolerance": "bit-exact",
      "evidence": "[reproduced]",
      "upstream": null,
      "notes": ""
    }
  ]
}
```

## Deterministic inputs

Probes generate inputs with a 64-bit LCG, `x_{n+1} = 6364136223846793005 * x_n + 1442695040888963407`,
seeded with the probe id hashed by FNV-1a. The oracle uses the same stream.

## Evidence labels

`[reproduced]` is the only label a result row may carry; the matrix exists to
turn `[self-reported]` claims into `[reproduced]` rows. Claims in `docs/`
use the full label set listed in `CONTRIBUTING.md`.
