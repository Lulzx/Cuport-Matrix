// Native baseline for warp-primitives.ballot_any_all.
//
// The predicate is (in[gid] & 0xF) != 0, which is true with probability
// 15/16 per lane, so roughly one simdgroup in eight is all-true and both
// simd_all outcomes are exercised.
//
// Output layout: out[gid * 3 + 0] = low 32 bits of simd_ballot(pred)
//                out[gid * 3 + 1] = simd_any(pred)
//                out[gid * 3 + 2] = simd_all(pred)

#include <metal_stdlib>
using namespace metal;

kernel void simd_ballot_baseline(
    device const uint *in  [[buffer(0)]],
    device uint       *out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    bool pred = (in[gid] & 0xFu) != 0u;
    simd_vote::vote_t mask = static_cast<simd_vote::vote_t>(simd_ballot(pred));
    out[gid * 3 + 0] = uint(mask & 0xFFFFFFFFul);
    out[gid * 3 + 1] = simd_any(pred) ? 1u : 0u;
    out[gid * 3 + 2] = simd_all(pred) ? 1u : 0u;
}
