// The third launch parameter sizes extern __shared__ memory. Three launches
// request 256 B, 4 KB and 16 KB; each block fills all of it, then thread 0
// folds it in reverse order.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "launch-geometry/dynamic-shared-size-001";

__global__ void fill_fold(const uint32_t *in, uint32_t n_words, uint32_t *folds) {
    extern __shared__ uint32_t s[];
    for (uint32_t i = threadIdx.x; i < n_words; i += blockDim.x) s[i] = in[i] ^ blockIdx.x;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t acc = 0;
        for (uint32_t i = 0; i < n_words; ++i) acc = acc * 0x01000193u + s[n_words - 1 - i];
        folds[blockIdx.x] = acc;
    }
}

int main() {
    const uint32_t sizes[3] = {64, 1024, 4096}, blocks = 8, threads = 128;
    std::vector<uint32_t> in(4096), folds(3 * blocks);
    cuport_fill_u32(ID, in.data(), in.size());
    uint32_t *d_in, *d_folds;
    CUPORT_CHECK(cudaMalloc(&d_in, in.size() * 4));
    CUPORT_CHECK(cudaMalloc(&d_folds, folds.size() * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    for (int l = 0; l < 3; ++l) {
        fill_fold<<<blocks, threads, sizes[l] * 4>>>(d_in, sizes[l], d_folds + l * blocks);
        CUPORT_CHECK(cudaGetLastError());
    }
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(folds.data(), d_folds, folds.size() * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("folds", folds.data(), folds.size());
    return cuport_done();
}
