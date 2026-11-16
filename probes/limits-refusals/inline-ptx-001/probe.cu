// Inline PTX: read %laneid through asm(). No CUDA-on-Metal tool can execute
// PTX, so the correct outcome is a compile-time diagnostic. A tool that
// accepts the asm and emits something else must still produce lane ids, or
// it is silently wrong.
#include "../../common/cuport.h"
#include <vector>

__global__ void lane(uint32_t *out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t l;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(l));
    out[i] = l;
}

int main() {
    const uint32_t blocks = 2, threads = 64, n = blocks * threads;
    uint32_t *d_out;
    CUPORT_CHECK(cudaMalloc(&d_out, n * 4));
    lane<<<blocks, threads>>>(d_out);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> out(n);
    CUPORT_CHECK(cudaMemcpy(out.data(), d_out, n * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("lane_id", out.data(), n);
    return cuport_done();
}
