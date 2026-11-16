# warp-primitives/warp-reduce-add-001

The idiom most CUDA code uses for warp sums: `v += __shfl_down_sync(full,
v, offset)` for offsets 16, 8, 4, 2, 1. Only lane 0's result is defined
and checked, over 1024 warps. It combines shuffle-down boundary handling
with the assumption that a warp is 32 lanes.
