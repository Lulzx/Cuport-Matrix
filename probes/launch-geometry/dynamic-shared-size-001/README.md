# launch-geometry/dynamic-shared-size-001

Three launches of the same kernel with `<<<8, 128, Ns>>>` for Ns = 256 B,
4 KB and 16 KB. Each block fills exactly `Ns / 4` words of `extern __shared__`
memory from the probe's LCG stream (XORed with `blockIdx.x`), then thread 0
folds them in reverse. A tool that ignores Ns, caps it, or sizes shared
memory from the first launch only produces wrong folds or crashes for the
larger launches.
