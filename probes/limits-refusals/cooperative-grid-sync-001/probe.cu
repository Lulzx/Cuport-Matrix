// Cooperative groups grid-wide sync. Each block writes a value, the whole
// grid synchronises, then each block reads the next block's value. Without
// a real grid barrier the reads race and the fold is wrong. The launch goes
// through cudaLaunchCooperativeKernel, which a tool without the feature must
// refuse.
#include "../../common/cuport.h"
#include <cooperative_groups.h>
#include <vector>

namespace cg = cooperative_groups;

static const char *ID = "limits-refusals/cooperative-grid-sync-001";

__global__ void ring(const uint32_t *in, uint32_t *mid, uint32_t *out) {
    cg::grid_group grid = cg::this_grid();
    uint32_t n = gridDim.x * blockDim.x;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    mid[i] = in[i] * 2654435761u;
    grid.sync();
    out[i] = mid[(i + blockDim.x) % n] ^ in[i];
}

int main() {
    const uint32_t blocks = 8, threads = 128, n = blocks * threads;
    std::vector<uint32_t> in(n), out(n);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in, *d_mid, *d_out;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_mid, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_out, n * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemset(d_mid, 0, n * 4));

    int supported = 0;
    CUPORT_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, 0));
    void *args[] = {&d_in, &d_mid, &d_out};
    CUPORT_CHECK(cudaLaunchCooperativeKernel((const void *)ring, dim3(blocks), dim3(threads), args, 0, 0));
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(out.data(), d_out, n * 4, cudaMemcpyDeviceToHost));

    uint32_t attr = (uint32_t)supported;
    cuport_emit_u32("cooperative_launch_attr", &attr, 1);
    cuport_emit_u32("out", out.data(), n);
    return cuport_done();
}
