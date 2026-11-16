// A static __shared__ array: each block stages 256 words and writes them back
// reversed, so every output word was written by a different thread.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "shared-memory-barriers/static-shared-001";

__global__ void reverse_block(const uint32_t *in, uint32_t *out) {
    __shared__ uint32_t tile[256];
    uint32_t gid = blockIdx.x * 256 + threadIdx.x;
    tile[threadIdx.x] = in[gid];
    __syncthreads();
    out[gid] = tile[255 - threadIdx.x];
}

int main() {
    const uint32_t blocks = 64, n = blocks * 256;
    std::vector<uint32_t> in(n), out(n);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in, *d_out;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_out, n * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    reverse_block<<<blocks, 256>>>(d_in, d_out);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(out.data(), d_out, n * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("out", out.data(), n);
    return cuport_done();
}
