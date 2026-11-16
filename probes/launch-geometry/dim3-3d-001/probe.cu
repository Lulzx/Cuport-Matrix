// 3D grid and 3D block: every thread writes all six coordinates, packed,
// at its linear position (x fastest, then y, then z).
#include "../../common/cuport.h"
#include <vector>

__global__ void dims3(uint32_t *packed, uint32_t *dims) {
    uint32_t block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    uint32_t thread = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    uint32_t idx = block * (blockDim.x * blockDim.y * blockDim.z) + thread;
    packed[idx] = blockIdx.x | (blockIdx.y << 4) | (blockIdx.z << 8) |
                  (threadIdx.x << 12) | (threadIdx.y << 16) | (threadIdx.z << 20);
    if (idx == 0) {
        dims[0] = blockDim.x; dims[1] = blockDim.y; dims[2] = blockDim.z;
        dims[3] = gridDim.x;  dims[4] = gridDim.y;  dims[5] = gridDim.z;
    }
}

int main() {
    const dim3 grid(3, 2, 4), block(8, 4, 2);
    const uint32_t n = 3 * 2 * 4 * 8 * 4 * 2;
    uint32_t *d_packed, *d_dims;
    CUPORT_CHECK(cudaMalloc(&d_packed, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_dims, 6 * 4));
    CUPORT_CHECK(cudaMemset(d_packed, 0xFF, n * 4));
    dims3<<<grid, block>>>(d_packed, d_dims);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> packed(n), dims(6);
    CUPORT_CHECK(cudaMemcpy(packed.data(), d_packed, n * 4, cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(dims.data(), d_dims, 6 * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("packed", packed.data(), n);
    cuport_emit_u32("dims", dims.data(), 6);
    return cuport_done();
}
