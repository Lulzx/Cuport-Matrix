// threadIdx/blockIdx/blockDim/gridDim on a 1D launch whose block size (100)
// is not a multiple of the warp size, so the last warp of every block is partial.
#include "../../common/cuport.h"
#include <vector>

__global__ void index_probe(uint32_t *packed, uint32_t *dims) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    packed[gid] = (blockIdx.x << 16) | threadIdx.x;
    if (gid == 0) {
        dims[0] = blockDim.x;
        dims[1] = gridDim.x;
    }
}

int main() {
    const uint32_t blocks = 37, threads = 100, n = blocks * threads;
    uint32_t *d_packed, *d_dims;
    CUPORT_CHECK(cudaMalloc(&d_packed, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_dims, 2 * 4));
    CUPORT_CHECK(cudaMemset(d_packed, 0xFF, n * 4));
    index_probe<<<blocks, threads>>>(d_packed, d_dims);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> packed(n), dims(2);
    CUPORT_CHECK(cudaMemcpy(packed.data(), d_packed, n * 4, cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(dims.data(), d_dims, 2 * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("packed", packed.data(), n);
    cuport_emit_u32("dims", dims.data(), 2);
    return cuport_done();
}
