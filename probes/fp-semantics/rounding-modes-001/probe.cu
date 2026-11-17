// Directed-rounding single-precision intrinsics: __fadd_, __fmul_, __fdiv_,
// __fsqrt_ and __fmaf_ with each suffix rn, rz, ru, rd. Each is one IEEE 754
// operation rounded in the stated direction.
// CUDA Math API, "Single Precision Intrinsics".
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "fp-semantics/rounding-modes-001";
static const uint32_t N = 1024;
static const int OPS = 5, MODES = 4;

__global__ void directed(const float *a, const float *b, const float *c, float *out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i], y = b[i], z = c[i], r = fabsf(a[i]);
    float *o = out + i;
    // out[(op * MODES + mode) * n + i]; modes in order rn, rz, ru, rd
    o[0 * n] = __fadd_rn(x, y);   o[1 * n] = __fadd_rz(x, y);   o[2 * n] = __fadd_ru(x, y);   o[3 * n] = __fadd_rd(x, y);
    o[4 * n] = __fmul_rn(x, y);   o[5 * n] = __fmul_rz(x, y);   o[6 * n] = __fmul_ru(x, y);   o[7 * n] = __fmul_rd(x, y);
    o[8 * n] = __fdiv_rn(x, y);   o[9 * n] = __fdiv_rz(x, y);   o[10 * n] = __fdiv_ru(x, y);  o[11 * n] = __fdiv_rd(x, y);
    o[12 * n] = __fsqrt_rn(r);    o[13 * n] = __fsqrt_rz(r);    o[14 * n] = __fsqrt_ru(r);    o[15 * n] = __fsqrt_rd(r);
    o[16 * n] = __fmaf_rn(x, y, z); o[17 * n] = __fmaf_rz(x, y, z); o[18 * n] = __fmaf_ru(x, y, z); o[19 * n] = __fmaf_rd(x, y, z);
}

static float from_bits(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

// random sign | biased exponent 127 + [-10, 10] | 23 random mantissa bits
static float draw(CuportLcg &g) {
    uint32_t s = g.next_u32() & 0x80000000u;
    uint32_t e = 117 + g.next_u32() % 21;
    return from_bits(s | (e << 23) | (g.next_u32() >> 9));
}

int main() {
    std::vector<float> a(N), b(N), c(N);
    CuportLcg g(ID);
    for (uint32_t i = 0; i < N; ++i) {
        a[i] = draw(g);
        b[i] = draw(g);
        c[i] = draw(g);
    }

    float *da, *db, *dc, *dout;
    CUPORT_CHECK(cudaMalloc(&da, N * sizeof(float)));
    CUPORT_CHECK(cudaMalloc(&db, N * sizeof(float)));
    CUPORT_CHECK(cudaMalloc(&dc, N * sizeof(float)));
    CUPORT_CHECK(cudaMalloc(&dout, OPS * MODES * N * sizeof(float)));
    CUPORT_CHECK(cudaMemcpy(da, a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemcpy(db, b.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemcpy(dc, c.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    directed<<<(N + 127) / 128, 128>>>(da, db, dc, dout, N);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(OPS * MODES * N);
    CUPORT_CHECK(cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const char *ops[OPS] = {"add", "mul", "div", "sqrt", "fma"};
    const char *modes[MODES] = {"rn", "rz", "ru", "rd"};
    char name[16];
    for (int op = 0; op < OPS; ++op) {
        for (int m = 0; m < MODES; ++m) {
            snprintf(name, sizeof name, "%s_%s", ops[op], modes[m]);
            cuport_emit_f32(name, &out[(op * MODES + m) * N], N);
        }
    }
    CUPORT_CHECK(cudaFree(da));
    CUPORT_CHECK(cudaFree(db));
    CUPORT_CHECK(cudaFree(dc));
    CUPORT_CHECK(cudaFree(dout));
    return cuport_done();
}
