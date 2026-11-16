# limits-refusals/cooperative-grid-sync-001

Eight blocks of 128 threads, launched with `cudaLaunchCooperativeKernel`.
Each thread writes a hash of its input, the grid synchronises with
`cooperative_groups::this_grid().sync()`, and each thread then reads the
value written by the corresponding thread of the next block.

Three honest outcomes exist. A tool without grid sync refuses at compile
time (unknown header or intrinsic) or at launch time (`CUPORT-ERROR` from
the cooperative launch), both `refused-with-diagnostic`. A tool that
implements it produces the oracle's `out` bit-exactly. A tool that lowers
`grid.sync()` to a block barrier or a no-op compiles and launches, but the
cross-block reads race against the writes and `out` is wrong in some
positions: `silently-wrong`. Metal has no grid-wide barrier, so the third
outcome is the one this probe exists to catch.
