// Device-side malloc/free: each thread allocates 16..256 bytes from the
// device heap, writes a pattern, folds it back and frees it. A null pointer
// is reported rather than dereferenced.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "device-services/malloc-free-001";

__global__ void heap(const uint32_t *v, uint32_t *sums, uint32_t *nulls) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t x = v[i];
    uint32_t n = 16 + (x & 0xF0u);
    unsigned char *p = (unsigned char *)malloc(n);
    if (p == nullptr) {
        nulls[i] = 1;
        sums[i] = 0;
        return;
    }
    for (uint32_t k = 0; k < n; ++k) p[k] = (unsigned char)(x + 7u * k);
    uint32_t s = 0;
    for (uint32_t k = 0; k < n; ++k) s += (uint32_t)p[k] * (k + 1);
    free(p);
    nulls[i] = 0;
    sums[i] = s;
}

int main() {
    const uint32_t blocks = 4, threads = 64, n = blocks * threads;
    std::vector<uint32_t> in(n), sums(n), nulls(n);
    cuport_fill_u32(ID, in.data(), n);
    CUPORT_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8u << 20));
    uint32_t *d_in, *d_sums, *d_nulls;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_sums, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_nulls, n * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    heap<<<blocks, threads>>>(d_in, d_sums, d_nulls);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(sums.data(), d_sums, n * 4, cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(nulls.data(), d_nulls, n * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("nulls", nulls.data(), n);
    cuport_emit_u32("sums", sums.data(), n);
    return cuport_done();
}
