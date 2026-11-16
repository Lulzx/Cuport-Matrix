// 64-bit atomics: atomicAdd with carries across the 32-bit boundary,
// atomicMax, and a 64-bit atomicCAS retry loop.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "atomics/atomic-64bit-001";

__global__ void wide(const uint32_t *in, unsigned long long *sum, unsigned long long *mx,
                     unsigned long long *casx) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long v = ((unsigned long long)in[2 * gid] << 32) | in[2 * gid + 1];
    uint32_t b = in[2 * gid] % 4u;
    atomicAdd(&sum[b], v);
    atomicMax(&mx[b], v);
    unsigned long long old = casx[b], assumed;
    do {
        assumed = old;
        old = atomicCAS(&casx[b], assumed, assumed ^ (v * 0x9E3779B97F4A7C15ull));
    } while (old != assumed);
}

int main() {
    const uint32_t blocks = 8, threads = 128, n = blocks * threads;
    std::vector<uint32_t> in(2 * n);
    std::vector<uint64_t> out(12);
    cuport_fill_u32(ID, in.data(), in.size());
    uint32_t *d_in;
    unsigned long long *d_out;
    CUPORT_CHECK(cudaMalloc(&d_in, in.size() * 4));
    CUPORT_CHECK(cudaMalloc(&d_out, 12 * 8));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemset(d_out, 0, 12 * 8));
    wide<<<blocks, threads>>>(d_in, d_out, d_out + 4, d_out + 8);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(out.data(), d_out, 12 * 8, cudaMemcpyDeviceToHost));
    cuport_emit_u64("sum", out.data(), 4);
    cuport_emit_u64("max", out.data() + 4, 4);
    cuport_emit_u64("cas_xor", out.data() + 8, 4);
    return cuport_done();
}
