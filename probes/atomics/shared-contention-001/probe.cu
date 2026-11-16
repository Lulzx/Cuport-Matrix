// Heavily contended shared-memory atomics: every thread of a block hits one
// shared counter, and an 8-bin shared histogram, then thread 0 publishes.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "atomics/shared-contention-001";

__global__ void contend(const uint32_t *in, uint32_t *totals, uint32_t *hist) {
    __shared__ uint32_t total;
    __shared__ uint32_t bins[8];
    if (threadIdx.x == 0) total = 0;
    if (threadIdx.x < 8) bins[threadIdx.x] = 0;
    __syncthreads();
    uint32_t v = in[blockIdx.x * blockDim.x + threadIdx.x];
    atomicAdd(&total, v & 0xFFFFu);
    atomicAdd(&bins[v % 8u], 1u);
    __syncthreads();
    if (threadIdx.x == 0) totals[blockIdx.x] = total;
    if (threadIdx.x < 8) hist[blockIdx.x * 8 + threadIdx.x] = bins[threadIdx.x];
}

int main() {
    const uint32_t blocks = 64, threads = 512, n = blocks * threads;
    std::vector<uint32_t> in(n), totals(blocks), hist(blocks * 8);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in, *d_totals, *d_hist;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_totals, blocks * 4));
    CUPORT_CHECK(cudaMalloc(&d_hist, blocks * 8 * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    contend<<<blocks, threads>>>(d_in, d_totals, d_hist);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(totals.data(), d_totals, blocks * 4, cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(hist.data(), d_hist, blocks * 8 * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("totals", totals.data(), blocks);
    cuport_emit_u32("hist", hist.data(), blocks * 8);
    return cuport_done();
}
