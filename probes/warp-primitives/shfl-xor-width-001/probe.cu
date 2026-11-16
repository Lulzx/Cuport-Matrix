// __shfl_xor_sync with every butterfly mask (1..16) and every width
// (32, 16, 8, 4, 2). Masks >= width cross into neighbouring groups: earlier
// groups are readable, later ones return the caller's own value.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "warp-primitives/shfl-xor-width-001";

__global__ void xor_width(const uint32_t *in, uint32_t *out) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x, v = in[gid];
    const int widths[5] = {32, 16, 8, 4, 2};
    for (int w = 0; w < 5; ++w)
        for (int m = 0; m < 5; ++m)
            out[gid * 25 + w * 5 + m] = __shfl_xor_sync(0xffffffffu, v, 1 << m, widths[w]);
}

int main() {
    const uint32_t blocks = 8, threads = 64, n = blocks * threads;
    std::vector<uint32_t> in(n), out(n * 25);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in, *d_out;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_out, n * 25 * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    xor_width<<<blocks, threads>>>(d_in, d_out);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(out.data(), d_out, n * 25 * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("out", out.data(), n * 25);
    return cuport_done();
}
