// Double precision is real IEEE binary64: +, *, /, sqrt and fma are each
// correctly rounded to nearest even. Apple GPUs have no FP64 hardware, so a
// tool must emulate it, demote it to float, or refuse it.
// CUDA C++ Programming Guide, "Floating-Point Standard" and "Double-Precision
// Floating-Point Functions" (x/y, sqrt and fma have 0 ulp error).
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "fp-semantics/fp64-emulation-001";
static const uint32_t N = 2048;

__global__ void fp64(const double *a, const double *b, const double *c, double *add, double *mul,
                     double *div, double *root, double *fused, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        add[i] = a[i] + b[i];
        mul[i] = a[i] * b[i];
        div[i] = a[i] / b[i];
        root[i] = sqrt(b[i]);
        fused[i] = fma(a[i], b[i], c[i]);
    }
}

static double from_bits(uint64_t u) {
    double f;
    memcpy(&f, &u, sizeof f);
    return f;
}

// sign | biased exponent 1023 + [-8, 7] | 52 random mantissa bits
static double draw(CuportLcg &g, bool positive) {
    uint64_t s = positive ? 0 : (uint64_t)(g.next_u32() & 0x80000000u) << 32;
    uint64_t e = 1023 - 8 + (g.next_u32() & 15);
    uint64_t hi = g.next_u32() & 0xfffffu;
    uint64_t lo = g.next_u32();
    return from_bits(s | (e << 52) | (hi << 32) | lo);
}

int main() {
    std::vector<double> a(N), b(N), c(N);
    CuportLcg g(ID);
    for (uint32_t i = 0; i < N; ++i) {
        a[i] = draw(g, false);
        b[i] = draw(g, true);
        double r = draw(g, false);
        if (i % 2 == 0) {
            double p = a[i] * b[i];  // one host multiply, nothing to contract it with
            c[i] = -p;               // fma then returns the product's rounding error
        } else {
            c[i] = r;
        }
    }

    double *d[8];
    for (auto &p : d) CUPORT_CHECK(cudaMalloc(&p, N * sizeof(double)));
    CUPORT_CHECK(cudaMemcpy(d[0], a.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemcpy(d[1], b.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemcpy(d[2], c.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    fp64<<<(N + 127) / 128, 128>>>(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], N);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());

    std::vector<double> out(N);
    const char *names[5] = {"add", "mul", "div", "sqrt", "fma"};
    for (int k = 0; k < 5; ++k) {
        CUPORT_CHECK(cudaMemcpy(out.data(), d[3 + k], N * sizeof(double), cudaMemcpyDeviceToHost));
        cuport_emit_f64(names[k], out.data(), N);
    }
    for (auto p : d) CUPORT_CHECK(cudaFree(p));
    return cuport_done();
}
