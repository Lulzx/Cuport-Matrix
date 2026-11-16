// The standard shuffle-down warp reduction: five __shfl_down_sync steps,
// lane 0 holds the warp's sum.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "warp-primitives/warp-reduce-add-001";

__global__ void warp_sum(const uint32_t *in, uint32_t *sums) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x, v = in[gid];
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    if (threadIdx.x % 32 == 0) sums[gid / 32] = v;
}

int main() {
    const uint32_t blocks = 128, threads = 256, n = blocks * threads, warps = n / 32;
    std::vector<uint32_t> in(n), sums(warps);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in, *d_sums;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_sums, warps * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    warp_sum<<<blocks, threads>>>(d_in, d_sums);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(sums.data(), d_sums, warps * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("sums", sums.data(), warps);
    return cuport_done();
}
