// __expf(x) intrinsic over |x| < 0.85 where error bound is 2 ulp.
// CUDA C++ Programming Guide, "Intrinsic Functions".
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "fp-semantics/fast-expf-001";
static const float SCALE = 0.84f;

__global__ void fast_exp(const float *in, float *out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __expf(in[idx]);
    }
}

int main() {
    std::vector<float> in = {0.0f, -0.0f};
    const uint32_t n_lcg = 4096;
    CuportLcg g(ID);
    for (uint32_t i = 0; i < n_lcg; ++i) {
        float u = g.next_unit_f32();
        in.push_back((2.0f * u - 1.0f) * SCALE);
    }
    const uint32_t n = in.size();
    std::vector<float> y(n);

    float *d_in, *d_y;
    CUPORT_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUPORT_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const uint32_t threads = 128;
    const uint32_t blocks = (n + threads - 1) / threads;
    fast_exp<<<blocks, threads>>>(d_in, d_y, n);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());

    CUPORT_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaFree(d_in));
    CUPORT_CHECK(cudaFree(d_y));

    cuport_emit_f32("y", y.data(), n);
    return cuport_done();
}
