// Default FMA contraction. nvcc's --fmad defaults to true, which fuses a*b+c
// into one fused multiply-add (one rounding). __fmul_rn and __fadd_rn are never
// fused (two roundings), and fmaf is always one rounding.
// CUDA C++ Programming Guide, "Intrinsic Functions"; nvcc documentation, --fmad.
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "fp-semantics/fma-contraction-001";
static const uint32_t N = 4096;

__global__ void contract(const float *a, const float *b, const float *c, float *expr, float *fused,
                         float *split, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        expr[i] = a[i] * b[i] + c[i];                       // contracted by default
        fused[i] = fmaf(a[i], b[i], c[i]);                  // always fused
        split[i] = __fadd_rn(__fmul_rn(a[i], b[i]), c[i]);  // never fused
    }
}

static float from_bits(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

int main() {
    // a, b in [1, 2). Even i: c = -(a*b rounded), so a fused result is the exact
    // rounding error of the product and an unfused one is 0. Odd i: c in [-4, -2).
    std::vector<float> a(N), b(N), c(N);
    CuportLcg g(ID);
    for (uint32_t i = 0; i < N; ++i) {
        a[i] = from_bits(0x3f800000u | (g.next_u32() >> 9));
        b[i] = from_bits(0x3f800000u | (g.next_u32() >> 9));
        uint32_t u = g.next_u32();
        if (i % 2 == 0) {
            float p = a[i] * b[i];  // one host multiply, nothing to contract it with
            c[i] = -p;
        } else {
            c[i] = from_bits(0xc0000000u | (u >> 9));
        }
    }

    float *d[6];
    for (auto &p : d) CUPORT_CHECK(cudaMalloc(&p, N * sizeof(float)));
    CUPORT_CHECK(cudaMemcpy(d[0], a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemcpy(d[1], b.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemcpy(d[2], c.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    contract<<<(N + 127) / 128, 128>>>(d[0], d[1], d[2], d[3], d[4], d[5], N);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(N);
    const char *names[3] = {"expr", "fused", "split"};
    for (int k = 0; k < 3; ++k) {
        CUPORT_CHECK(cudaMemcpy(out.data(), d[3 + k], N * sizeof(float), cudaMemcpyDeviceToHost));
        cuport_emit_f32(names[k], out.data(), N);
    }
    for (auto p : d) CUPORT_CHECK(cudaFree(p));
    return cuport_done();
}
