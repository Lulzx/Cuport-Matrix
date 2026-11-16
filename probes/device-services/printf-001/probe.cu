// Device printf: 64 threads each print one line with integer, hex, float,
// long long and string conversions plus a literal percent. The host matches
// the lines as a multiset, so thread order does not matter.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "device-services/printf-001";

__global__ void say(const uint32_t *v) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t x = v[i];
    int s = (int)(x >> 8) - 0x800000;              // signed, [-2^23, 2^23)
    float f = (float)(x & 0x3FFFu) / 8.0f;         // k/8: exact in 3 decimals, no ties
    const char *w = (x & 1u) ? "odd" : "even";
    printf("b%u t%u d=%d u=%u x=%08x f=%.3f ll=%lld s=%s%%\n",
           blockIdx.x, threadIdx.x, s, x, x, f, (long long)x * -3, w);
}

int main() {
    const uint32_t blocks = 4, threads = 16, n = blocks * threads;
    std::vector<uint32_t> in(n);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    say<<<blocks, threads>>>(d_in);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    fflush(stdout);
    uint32_t lines = n;
    cuport_emit_u32("lines", &lines, 1);
    return cuport_done();
}
