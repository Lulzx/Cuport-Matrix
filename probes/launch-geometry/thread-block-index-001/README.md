# launch-geometry/thread-block-index-001

Every thread of a 37 x 100 launch records its `blockIdx.x` and `threadIdx.x`,
and thread 0 records `blockDim.x` and `gridDim.x`. A block of 100 threads ends
in a partial warp of 4, which a translator that rounds blocks up to whole
SIMD groups must mask off. Words a thread failed to write keep the
`0xFFFFFFFF` fill and show up as mismatches.
