// Native baseline for shared-memory-barriers.syncthreads_reduction.
//
// Classic shared-memory tree reduction: each threadgroup of 256 threads sums
// its 256 input words with threadgroup_barrier between steps, the Metal
// counterpart of __syncthreads. Unsigned addition wraps, so the result is
// order-independent and the check is bit-exact.
//
// Output layout: out[tg] = sum of in[tg * 256 .. tg * 256 + 255] mod 2^32

#include <metal_stdlib>
using namespace metal;

constant uint TG_SIZE = 256;

kernel void threadgroup_reduce_baseline(
    device const uint *in  [[buffer(0)]],
    device uint       *out [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg  [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[TG_SIZE];
    scratch[tid] = in[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = TG_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out[tg] = scratch[0];
    }
}
