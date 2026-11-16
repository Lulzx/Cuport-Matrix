// Dynamic shared memory at the Apple threadgroup limit (exactly 32 KB) and
// one word past it (32 KB + 4 B). CUDA allows up to 48 KB without opt-in,
// so both launches are valid CUDA.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "shared-memory-barriers/shared-32kb-001";

__global__ void fill_fold(const uint32_t *in, uint32_t n_words, uint32_t *folds) {
    extern __shared__ uint32_t s[];
    for (uint32_t i = threadIdx.x; i < n_words; i += blockDim.x) s[i] = in[i] ^ (blockIdx.x * 0x9E3779B9u);
    __syncthreads();
    uint32_t acc = 0;
    for (uint32_t i = threadIdx.x; i < n_words; i += blockDim.x) acc += s[n_words - 1 - i] * (i | 1u);
    __syncthreads();  // all reads of s are done; reuse it for the partial sums
    s[threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t total = 0;
        for (uint32_t t = 0; t < blockDim.x; ++t) total += s[t];
        folds[blockIdx.x] = total;
    }
}

int main() {
    const uint32_t sizes[2] = {8192, 8193}, blocks = 4, threads = 256;
    std::vector<uint32_t> in(8193), folds(2 * blocks);
    cuport_fill_u32(ID, in.data(), in.size());
    uint32_t *d_in, *d_folds;
    CUPORT_CHECK(cudaMalloc(&d_in, in.size() * 4));
    CUPORT_CHECK(cudaMalloc(&d_folds, folds.size() * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    for (int l = 0; l < 2; ++l) {
        fill_fold<<<blocks, threads, sizes[l] * 4>>>(d_in, sizes[l], d_folds + l * blocks);
        CUPORT_CHECK(cudaGetLastError());
        CUPORT_CHECK(cudaDeviceSynchronize());
    }
    CUPORT_CHECK(cudaMemcpy(folds.data(), d_folds, folds.size() * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("folds", folds.data(), folds.size());
    return cuport_done();
}
