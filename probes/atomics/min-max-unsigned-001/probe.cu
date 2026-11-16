// atomicMin/atomicMax on unsigned and signed ints over the same bit patterns,
// half of which have the top bit set, so unsigned and signed orders differ.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "atomics/min-max-unsigned-001";

__global__ void minmax(const uint32_t *in, uint32_t *umin, uint32_t *umax, int *smin, int *smax) {
    uint32_t v = in[blockIdx.x * blockDim.x + threadIdx.x], b = (v >> 3) % 8u;
    atomicMin(&umin[b], v);
    atomicMax(&umax[b], v);
    atomicMin(&smin[b], (int)v);
    atomicMax(&smax[b], (int)v);
}

int main() {
    const uint32_t blocks = 16, threads = 256, n = blocks * threads;
    std::vector<uint32_t> in(n), out(32);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t init[32];
    for (int i = 0; i < 8; ++i) {
        init[i] = 0xFFFFFFFFu;      // umin
        init[8 + i] = 0u;           // umax
        init[16 + i] = 0x7FFFFFFFu; // smin (INT_MAX)
        init[24 + i] = 0x80000000u; // smax (INT_MIN)
    }
    uint32_t *d_in, *d_out;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_out, 32 * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemcpy(d_out, init, 32 * 4, cudaMemcpyHostToDevice));
    minmax<<<blocks, threads>>>(d_in, d_out, d_out + 8, (int *)(d_out + 16), (int *)(d_out + 24));
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(out.data(), d_out, 32 * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("umin", out.data(), 8);
    cuport_emit_u32("umax", out.data() + 8, 8);
    cuport_emit_i32("smin", (const int32_t *)(out.data() + 16), 8);
    cuport_emit_i32("smax", (const int32_t *)(out.data() + 24), 8);
    return cuport_done();
}
