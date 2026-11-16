// A 1D grid of 70001 blocks (more than the 65535 limit of the y and z axes)
// with __launch_bounds__(256). Each thread writes a hash of its global id;
// the host folds each block's 256 words so the output stays small.
#include "../../common/cuport.h"
#include <vector>

__global__ void __launch_bounds__(256) large_grid(uint32_t *out) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    out[gid] = (gid * 2654435761u) ^ blockIdx.x;
}

int main() {
    const uint32_t blocks = 70001, threads = 256;
    const size_t n = (size_t)blocks * threads;
    uint32_t *d_out;
    CUPORT_CHECK(cudaMalloc(&d_out, n * 4));
    CUPORT_CHECK(cudaMemset(d_out, 0, n * 4));
    large_grid<<<blocks, threads>>>(d_out);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> out(n), fold(blocks);
    CUPORT_CHECK(cudaMemcpy(out.data(), d_out, n * 4, cudaMemcpyDeviceToHost));
    for (uint32_t b = 0; b < blocks; ++b) {
        uint32_t f = 0;
        for (uint32_t t = 0; t < threads; ++t) f = (f * 0x01000193u) ^ out[(size_t)b * threads + t];
        fold[b] = f;
    }
    cuport_emit_u32("block_fold", fold.data(), blocks);
    return cuport_done();
}
