// __shfl_sync (indexed), __shfl_up_sync and __shfl_down_sync on full warps,
// including the lanes whose source falls off the end and keep their own value.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "warp-primitives/shfl-variants-001";

__global__ void shuffles(const uint32_t *in, uint32_t *idx, uint32_t *up, uint32_t *down) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane = threadIdx.x % 32, v = in[gid];
    idx[gid] = __shfl_sync(0xffffffffu, v, (int)((lane * 7 + 3) % 32));
    up[gid] = __shfl_up_sync(0xffffffffu, v, 5);
    down[gid] = __shfl_down_sync(0xffffffffu, v, 3);
}

int main() {
    const uint32_t blocks = 16, threads = 128, n = blocks * threads;
    std::vector<uint32_t> in(n), out(3 * n);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in, *d_out;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_out, 3 * n * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    shuffles<<<blocks, threads>>>(d_in, d_out, d_out + n, d_out + 2 * n);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(out.data(), d_out, 3 * n * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("idx", out.data(), n);
    cuport_emit_u32("up", out.data() + n, n);
    cuport_emit_u32("down", out.data() + 2 * n, n);
    return cuport_done();
}
