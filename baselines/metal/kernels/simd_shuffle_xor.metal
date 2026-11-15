// Native baseline for warp-primitives.shfl_xor_width.
//
// Each thread reads one input word, then records what simd_shuffle_xor
// returns for every butterfly mask 1, 2, 4, 8 and 16. Lane 0 of each
// simdgroup also records the simdgroup width, because CUDA code assumes 32.
//
// Output layout: out[gid * 5 + k] = simd_shuffle_xor(in[gid], 1u << k)
//                width[gid / width] = threads_per_simdgroup

#include <metal_stdlib>
using namespace metal;

kernel void simd_shuffle_xor_baseline(
    device const uint *in    [[buffer(0)]],
    device uint       *out   [[buffer(1)]],
    device uint       *width [[buffer(2)]],
    uint gid    [[thread_position_in_grid]],
    uint lane   [[thread_index_in_simdgroup]],
    uint simd_w [[threads_per_simdgroup]])
{
    uint v = in[gid];
    for (uint k = 0; k < 5; ++k) {
        out[gid * 5 + k] = simd_shuffle_xor(v, ushort(1u << k));
    }
    if (lane == 0) {
        width[gid / simd_w] = simd_w;
    }
}
