// Texture objects over a linear buffer, read with tex1Dfetch. This is the
// CUDA 12 texture API (texture references were removed in CUDA 12.0). A
// gather through a texture object must return the same words as a plain
// global load.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "limits-refusals/texture-object-001";

__global__ void gather(cudaTextureObject_t tex, uint32_t *out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = tex1Dfetch<uint32_t>(tex, (int)((i * 7u) % n));
}

int main() {
    const uint32_t blocks = 4, threads = 64, n = blocks * threads;
    std::vector<uint32_t> in(n), out(n);
    cuport_fill_u32(ID, in.data(), n);
    uint32_t *d_in, *d_out;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_out, n * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));

    cudaResourceDesc rd;
    memset(&rd, 0, sizeof(rd));
    rd.resType = cudaResourceTypeLinear;
    rd.res.linear.devPtr = d_in;
    rd.res.linear.desc = cudaCreateChannelDesc<uint32_t>();
    rd.res.linear.sizeInBytes = n * 4;
    cudaTextureDesc td;
    memset(&td, 0, sizeof(td));
    td.readMode = cudaReadModeElementType;
    cudaTextureObject_t tex = 0;
    CUPORT_CHECK(cudaCreateTextureObject(&tex, &rd, &td, nullptr));

    gather<<<blocks, threads>>>(tex, d_out, n);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaDestroyTextureObject(tex));
    CUPORT_CHECK(cudaMemcpy(out.data(), d_out, n * 4, cudaMemcpyDeviceToHost));
    cuport_emit_u32("gathered", out.data(), n);
    return cuport_done();
}
