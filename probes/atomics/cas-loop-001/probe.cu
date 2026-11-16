// The classic atomicCAS retry loop implementing an atomic add of a hashed
// addend. Addition commutes, so each bucket's final value is order-independent.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "atomics/cas-loop-001";

__device__ uint32_t cas_add(uint32_t *addr, uint32_t v) {
    uint32_t old = *addr, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, assumed + v);
    } while (old != assumed);
    return old;
}

__global__ void cas(const uint32_t *in, uint32_t *acc, uint32_t *returned) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x, v = in[gid];
    returned[gid] = cas_add(&acc[v % 8u], v * 2654435761u);
}

int main() {
    const uint32_t blocks = 16, threads = 128, n = blocks * threads;
    std::vector<uint32_t> in(n), acc(8), returned(n);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in, *d_acc, *d_ret;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_acc, 8 * 4));
    CUPORT_CHECK(cudaMalloc(&d_ret, n * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemset(d_acc, 0, 8 * 4));
    cas<<<blocks, threads>>>(d_in, d_acc, d_ret);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(acc.data(), d_acc, 8 * 4, cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(returned.data(), d_ret, n * 4, cudaMemcpyDeviceToHost));
    // Which intermediate total each thread saw depends on the order, but within
    // a bucket they must all differ. Emit the number of distinct values per bucket.
    std::vector<uint32_t> distinct(8, 0);
    for (uint32_t b = 0; b < 8; ++b) {
        std::vector<uint32_t> seen;
        for (uint32_t i = 0; i < n; ++i)
            if (in[i] % 8u == b) seen.push_back(returned[i]);
        std::vector<uint32_t> sorted(seen);
        for (size_t i = 1; i < sorted.size(); ++i)
            for (size_t j = i; j > 0 && sorted[j - 1] > sorted[j]; --j) { uint32_t t = sorted[j]; sorted[j] = sorted[j - 1]; sorted[j - 1] = t; }
        uint32_t d = sorted.empty() ? 0 : 1;
        for (size_t i = 1; i < sorted.size(); ++i) d += sorted[i] != sorted[i - 1];
        distinct[b] = d;
    }
    cuport_emit_u32("acc", acc.data(), 8);
    cuport_emit_u32("distinct_returns", distinct.data(), 8);
    return cuport_done();
}
