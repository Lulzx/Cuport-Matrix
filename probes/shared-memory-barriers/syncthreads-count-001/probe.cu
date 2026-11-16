// __syncthreads_count, __syncthreads_and and __syncthreads_or over
// 128-thread blocks, with predicates chosen so each result varies by block.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "shared-memory-barriers/syncthreads-count-001";

__global__ void vote(const uint32_t *in, uint32_t *count, uint32_t *all, uint32_t *any) {
    uint32_t v = in[blockIdx.x * blockDim.x + threadIdx.x];
    int c = __syncthreads_count((v & 7u) == 0u);
    int a = __syncthreads_and((v & 0xFFu) != 0u);
    int o = __syncthreads_or((v & 0xFFFu) == 0u);
    if (threadIdx.x == 0) {
        count[blockIdx.x] = (uint32_t)c;
        all[blockIdx.x] = a != 0;
        any[blockIdx.x] = o != 0;
    }
}

int main() {
    const uint32_t blocks = 256, threads = 128, n = blocks * threads;
    std::vector<uint32_t> in(n), count(blocks), all(blocks), any(blocks);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in, *d_out;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_out, 3 * blocks * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    vote<<<blocks, threads>>>(d_in, d_out, d_out + blocks, d_out + 2 * blocks);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(count.data(), d_out, blocks * 4, cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(all.data(), d_out + blocks, blocks * 4, cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(any.data(), d_out + 2 * blocks, blocks * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("count", count.data(), blocks);
    cuport_emit_u32("all", all.data(), blocks);
    cuport_emit_u32("any", any.data(), blocks);
    return cuport_done();
}
