// Dynamic parallelism (CDP2, CUDA 12): a parent kernel launches one child
// block per parent block from thread 0. The parent grid is not complete
// until its children are, so a host cudaDeviceSynchronize after the parent
// sees the children's writes. No device-side synchronisation is used
// because CDP2 removed device-side cudaDeviceSynchronize.
//
// Stock nvcc needs -rdc=true and links cudadevrt for this probe; the
// reference-drop command must record both flags (CONTRIBUTING.md).
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "limits-refusals/dynamic-parallelism-001";

__global__ void child(const uint32_t *in, uint32_t *out, uint32_t base) {
    uint32_t i = base + threadIdx.x;
    out[i] = in[i] * 5u + 1u;
}

__global__ void parent(const uint32_t *in, uint32_t *out) {
    if (threadIdx.x == 0) child<<<1, blockDim.x>>>(in, out, blockIdx.x * blockDim.x);
}

int main() {
    const uint32_t blocks = 4, threads = 64, n = blocks * threads;
    std::vector<uint32_t> in(n), out(n);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in, *d_out;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_out, n * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemset(d_out, 0, n * 4));

    parent<<<blocks, threads>>>(d_in, d_out);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(out.data(), d_out, n * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("child_out", out.data(), n);
    return cuport_done();
}
