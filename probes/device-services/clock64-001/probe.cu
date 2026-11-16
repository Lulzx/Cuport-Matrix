// clock64() read twice around a dependent 4096-step LCG loop. The counter
// must strictly increase across that much work. The loop result is emitted
// so the compiler cannot delete the work between the two reads.
#include "../../common/cuport.h"
#include <vector>

__global__ void tick(uint32_t *sink, uint32_t *increasing, uint64_t *delta) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    long long t0 = clock64();
    uint32_t acc = i;
    for (int k = 0; k < 4096; ++k) acc = acc * 1664525u + 1013904223u;
    long long t1 = clock64();
    sink[i] = acc;
    increasing[i] = (t1 > t0) ? 1u : 0u;
    delta[i] = (uint64_t)(t1 - t0);
}

int main() {
    const uint32_t blocks = 2, threads = 64, n = blocks * threads;
    uint32_t *d_sink, *d_inc;
    uint64_t *d_delta;
    CUPORT_CHECK(cudaMalloc(&d_sink, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_inc, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_delta, n * 8));
    tick<<<blocks, threads>>>(d_sink, d_inc, d_delta);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> sink(n), inc(n);
    std::vector<uint64_t> delta(n);
    CUPORT_CHECK(cudaMemcpy(sink.data(), d_sink, n * 4, cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(inc.data(), d_inc, n * 4, cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(delta.data(), d_delta, n * 8, cudaMemcpyDeviceToHost));
    uint32_t any_zero_delta = 0;
    for (uint32_t i = 0; i < n; ++i) any_zero_delta |= (delta[i] == 0);
    cuport_emit_u32("sink", sink.data(), n);
    cuport_emit_u32("increasing", inc.data(), n);
    cuport_emit_u32("any_zero_delta", &any_zero_delta, 1);
    return cuport_done();
}
