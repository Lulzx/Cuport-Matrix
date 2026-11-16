# launch-geometry/large-grid-001

70001 blocks of 256 threads under `__launch_bounds__(256)`: 17,920,256
threads, past the 65535 limit that applies to the y and z axes but not x.
Each thread writes `(gid * 2654435761) ^ blockIdx.x`, and the host folds
each block's 256 words into one value, so a dropped, duplicated or
misplaced block shows up at its own index in `block_fold`.
