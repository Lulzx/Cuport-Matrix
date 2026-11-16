// __fdividef(x, y) intrinsic over |y| in [2^-126, 2^126] with finite normal quotient.
// CUDA C++ Programming Guide, "Intrinsic Functions".
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "fp-semantics/fast-fdividef-001";

__global__ void fast_fdivide(const float *x, const float *y, float *out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __fdividef(x[idx], y[idx]);
    }
}

int main() {
    const uint32_t n = 4096;
    std::vector<float> h_x(n), h_y(n), h_out(n);
    CuportLcg g(ID);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t ux = g.next_u32();
        uint32_t uy = g.next_u32();
        uint32_t exp_x = 70 + ((ux >> 23) % 115);
        uint32_t exp_y = 70 + ((uy >> 23) % 115);
        uint32_t bx = (ux & 0x80000000u) | (exp_x << 23) | (ux & 0x007FFFFFu);
        uint32_t by = (uy & 0x80000000u) | (exp_y << 23) | (uy & 0x007FFFFFu);
        memcpy(&h_x[i], &bx, sizeof(float));
        memcpy(&h_y[i], &by, sizeof(float));
    }

    float *d_x, *d_y, *d_out;
    CUPORT_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUPORT_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUPORT_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUPORT_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const uint32_t threads = 128;
    const uint32_t blocks = (n + threads - 1) / threads;
    fast_fdivide<<<blocks, threads>>>(d_x, d_y, d_out, n);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());

    CUPORT_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaFree(d_x));
    CUPORT_CHECK(cudaFree(d_y));
    CUPORT_CHECK(cudaFree(d_out));

    cuport_emit_f32("y", h_out.data(), n);
    return cuport_done();
}
