# Self-hosted M6 runner

The `m6-self-hosted` job in `.github/workflows/matrix.yml` runs on the Apple
M6 Mac mini (`m6-12c`, 12-core GPU) and writes the same rows as a local run.
It runs `.github/scripts/m6-lane.sh`, which also works from a local checkout.

## Register the runner

Registering a repository runner needs admin rights on
`guru-bharadwaj20/Cuport-Matrix`, so it takes both of us:

1. Guru opens Settings → Actions → Runners → New self-hosted runner, picks
   macOS and ARM64, and sends Vivian the `config.sh` token from that page.
   The token expires after an hour.
2. Vivian, on the M6, runs the download commands from the same page into
   `~/actions-runner`, then:

   ```sh
   ./config.sh --url https://github.com/guru-bharadwaj20/Cuport-Matrix \
     --token <TOKEN> --name m6-12c --labels m6 --unattended
   ./svc.sh install
   ./svc.sh start
   ```

   GitHub adds the `self-hosted`, `macOS` and `ARM64` labels itself; `m6`
   is the one the job selects on.

The service runs as Vivian's user, so it finds the tools in
`~/.local/cuport-tools` (built by `build-tools.sh` at the pins in
`matrix/tools.yaml`) and `uv` in `~/.local/bin`. Set `CUPORT_TOOLS` in the
runner's `.env` file if the tools live elsewhere.

## Run it

Actions → matrix → Run workflow, with `m6` ticked and `tool` set to `all`,
`mvcc` or `cumetal`. The job:

- refuses to run unless `system_profiler` reports an Apple M6 with 12 GPU
  cores, so rows labelled `m6-12c` only come from that machine;
- runs the runner's unit tests, then every probe through each tool;
- validates the rows and uploads `results/*/m6-12c/` as the
  `results-m6-12c` artifact.

The job does not commit. To publish a row, download the artifact, check it
against the previous row, and commit it under `results/` as usual. A row
records the repo commit it ran from, and the lane only runs from a clean
checkout, so its `repo_dirty` is always false.

The job is off by default (`m6: false`) so a dispatch does not wait in the
queue while the M6 is offline.
