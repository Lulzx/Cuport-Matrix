# results/

Committed result rows, one JSON file per `<tool>/<hardware>/<date>.json`.
Format: `matrix/schema.md`.

Two lanes write rows for CuMetal v0.8.0 and mvcc v1.3.0:

- `m6-12c`: the self-hosted Apple M6 (`.github/scripts/m6-lane.sh`), committed
  by hand after review.
- `paravirtual`: the GitHub-hosted runner (`.github/scripts/hosted-lane.sh`),
  committed by the nightly `matrix` workflow together with the regenerated
  `docs/scoreboard.md`.

A row records the tool commit, macOS build, GPU and the repo commit it ran
from. `docs/scoreboard.md` is generated from the newest row per tool and
lane (`PYTHONPATH=runner python -m cuport scoreboard`).
