// Tree reduction in shared memory: each 512-thread block sums its inputs,
// with __syncthreads() between the nine halving steps.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "shared-memory-barriers/syncthreads-reduction-001";

__global__ void block_sum(const uint32_t *in, uint32_t *sums) {
    __shared__ uint32_t s[512];
    uint32_t t = threadIdx.x;
    s[t] = in[blockIdx.x * 512 + t];
    __syncthreads();
    for (uint32_t stride = 256; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0) sums[blockIdx.x] = s[0];
}

int main() {
    const uint32_t blocks = 128, n = blocks * 512;
    std::vector<uint32_t> in(n), sums(blocks);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in, *d_sums;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_sums, blocks * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    block_sum<<<blocks, 512>>>(d_in, d_sums);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(sums.data(), d_sums, blocks * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("sums", sums.data(), blocks);
    return cuport_done();
}
