// A launch that asks for 64 KB of dynamic shared memory with no
// cudaFuncSetAttribute opt-in. CUDA refuses it at launch
// (cudaErrorInvalidValue) and runs nothing. On Apple hardware the limit is
// 32 KB, so the refusal is doubly required; the question is whether the
// tool refuses or silently runs with less memory than asked for.
#include "../../common/cuport.h"
#include <vector>

__global__ void fill(uint32_t *out) {
    extern __shared__ uint32_t s[];
    s[threadIdx.x] = threadIdx.x * 3u + 1u;
    __syncthreads();
    out[threadIdx.x] = s[blockDim.x - 1 - threadIdx.x];
}

int main() {
    const uint32_t threads = 256;
    uint32_t *d_out;
    CUPORT_CHECK(cudaMalloc(&d_out, threads * 4));
    CUPORT_CHECK(cudaMemset(d_out, 0xAB, threads * 4));

    fill<<<1, threads, 64 * 1024>>>(d_out);
    int32_t launch = (int32_t)cudaGetLastError();
    int32_t sync = (int32_t)cudaDeviceSynchronize();

    std::vector<uint32_t> out(threads);
    CUPORT_CHECK(cudaMemcpy(out.data(), d_out, threads * 4, cudaMemcpyDeviceToHost));
    int32_t codes[2] = {launch, sync};
    uint32_t refused = launch != 0;
    cuport_emit_u32("refused", &refused, 1);
    cuport_emit_i32("codes", codes, 2);
    cuport_emit_u32("untouched", out.data(), threads);
    return cuport_done();
}
