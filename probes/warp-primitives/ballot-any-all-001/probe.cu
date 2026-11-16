// __ballot_sync, __any_sync and __all_sync on full warps, with predicates
// chosen so both outcomes of any/all occur.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "warp-primitives/ballot-any-all-001";

__global__ void votes(const uint32_t *in, uint32_t *ballot, uint32_t *any, uint32_t *all) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x, v = in[gid];
    uint32_t b = __ballot_sync(0xffffffffu, (v & 0xFu) != 0u);
    int a = __any_sync(0xffffffffu, (v & 0x3FFu) == 0u);
    int l = __all_sync(0xffffffffu, (v & 0xFu) != 0u);
    if (threadIdx.x % 32 == 0) {
        uint32_t warp = gid / 32;
        ballot[warp] = b;
        any[warp] = a != 0;
        all[warp] = l != 0;
    }
}

int main() {
    const uint32_t blocks = 64, threads = 128, n = blocks * threads, warps = n / 32;
    std::vector<uint32_t> in(n), out(3 * warps);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in, *d_out;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_out, 3 * warps * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    votes<<<blocks, threads>>>(d_in, d_out, d_out + warps, d_out + 2 * warps);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(out.data(), d_out, 3 * warps * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("ballot", out.data(), warps);
    cuport_emit_u32("any", out.data() + warps, warps);
    cuport_emit_u32("all", out.data() + 2 * warps, warps);
    return cuport_done();
}
