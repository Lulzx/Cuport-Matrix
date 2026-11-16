// Blocks of 48 threads: the second warp of each block has only 16 lanes.
// Each warp builds its member mask explicitly and uses it for
// __ballot_sync and __shfl_sync, so nothing touches a non-existent lane.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "warp-primitives/partial-warp-001";

__global__ void partial(const uint32_t *in, uint32_t *mask_out, uint32_t *ballot_out, uint32_t *shfl_out) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x, v = in[gid];
    uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    uint32_t members = blockDim.x - warp * 32 < 32 ? blockDim.x - warp * 32 : 32;
    uint32_t mask = members == 32 ? 0xffffffffu : (1u << members) - 1u;
    mask_out[gid] = mask;
    ballot_out[gid] = __ballot_sync(mask, (v & 1u) != 0u);
    shfl_out[gid] = __shfl_sync(mask, v, (int)((lane + 1) % members));
}

int main() {
    const uint32_t blocks = 8, threads = 48, n = blocks * threads;
    std::vector<uint32_t> in(n), out(3 * n);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in, *d_out;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_out, 3 * n * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    partial<<<blocks, threads>>>(d_in, d_out, d_out + n, d_out + 2 * n);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(out.data(), d_out, 3 * n * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("mask", out.data(), n);
    cuport_emit_u32("ballot", out.data() + n, n);
    cuport_emit_u32("shfl", out.data() + 2 * n, n);
    return cuport_done();
}
