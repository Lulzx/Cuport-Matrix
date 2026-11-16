// Global atomicAdd on int and float into 16 buckets. Float addends are small
// integers, so every summation order gives the same exact result.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "atomics/atomic-add-001";

__global__ void add(const uint32_t *in, int *isum, float *fsum) {
    uint32_t v = in[blockIdx.x * blockDim.x + threadIdx.x];
    uint32_t bucket = v % 16u;
    atomicAdd(&isum[bucket], (int)(v >> 4) - (1 << 27));
    atomicAdd(&fsum[bucket], (float)(v & 0xFFu));
}

int main() {
    const uint32_t blocks = 32, threads = 128, n = blocks * threads;
    std::vector<uint32_t> in(n);
    std::vector<int32_t> isum(16);
    std::vector<float> fsum(16);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in;
    int *d_isum;
    float *d_fsum;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_isum, 16 * 4));
    CUPORT_CHECK(cudaMalloc(&d_fsum, 16 * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemset(d_isum, 0, 16 * 4));
    CUPORT_CHECK(cudaMemset(d_fsum, 0, 16 * 4));
    add<<<blocks, threads>>>(d_in, d_isum, d_fsum);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(isum.data(), d_isum, 16 * 4, cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(fsum.data(), d_fsum, 16 * 4, cudaMemcpyDeviceToHost));
    cuport_emit_i32("isum", isum.data(), 16);
    cuport_emit_f32("fsum", fsum.data(), 16);
    return cuport_done();
}
