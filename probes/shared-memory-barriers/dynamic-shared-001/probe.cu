// extern __shared__ carved into a float array and a uint32 array, the usual
// way to put two arrays in one dynamic allocation. Block size 192.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "shared-memory-barriers/dynamic-shared-001";

__global__ void carve(const float *fin, const uint32_t *uin, float *fout, uint32_t *uout) {
    extern __shared__ uint32_t buf[];
    float *f = reinterpret_cast<float *>(buf);
    uint32_t *u = buf + blockDim.x;
    uint32_t t = threadIdx.x, gid = blockIdx.x * blockDim.x + t;
    f[t] = fin[gid];
    u[t] = uin[gid];
    __syncthreads();
    fout[gid] = f[(t + 1) % blockDim.x] * 2.0f;
    uout[gid] = u[(t + 7) % blockDim.x];
}

int main() {
    const uint32_t blocks = 32, threads = 192, n = blocks * threads;
    std::vector<float> fin(n), fout(n);
    std::vector<uint32_t> uin(n), uout(n);
    CuportLcg g(ID);
    for (uint32_t i = 0; i < n; ++i) fin[i] = g.next_unit_f32();
    for (uint32_t i = 0; i < n; ++i) uin[i] = g.next_u32();
    float *d_fin, *d_fout;
    uint32_t *d_uin, *d_uout;
    CUPORT_CHECK(cudaMalloc(&d_fin, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_fout, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_uin, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_uout, n * 4));
    CUPORT_CHECK(cudaMemcpy(d_fin, fin.data(), n * 4, cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemcpy(d_uin, uin.data(), n * 4, cudaMemcpyHostToDevice));
    carve<<<blocks, threads, 2 * threads * 4>>>(d_fin, d_uin, d_fout, d_uout);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(fout.data(), d_fout, n * 4, cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(uout.data(), d_uout, n * 4, cudaMemcpyDeviceToHost));
    cuport_emit_f32("fout", fout.data(), n);
    cuport_emit_u32("uout", uout.data(), n);
    return cuport_done();
}
