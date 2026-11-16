// __match_any_sync: each lane gets the mask of lanes holding the same key.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "warp-primitives/match-any-001";

__global__ void match(const uint32_t *in, uint32_t *out) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    out[gid] = __match_any_sync(0xffffffffu, in[gid] % 5u);
}

int main() {
    const uint32_t blocks = 16, threads = 128, n = blocks * threads;
    std::vector<uint32_t> in(n), out(n);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in, *d_out;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_out, n * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    match<<<blocks, threads>>>(d_in, d_out);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(out.data(), d_out, n * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("masks", out.data(), n);
    return cuport_done();
}
