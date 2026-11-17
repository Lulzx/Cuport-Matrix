// Subnormal (denormal) operands and results in float and double arithmetic.
// Stock nvcc defaults to --ftz=false: single-precision subnormals are kept, not
// flushed to zero. Double precision never flushes.
// nvcc documentation, --ftz; CUDA C++ Programming Guide, "Floating-Point Standard".
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "fp-semantics/denormals-001";
static const uint32_t N = 2048;

__global__ void denorm(const float *tiny, const float *sub, const float *sub2, const double *dtiny,
                       const double *dsub, float *half_tiny, float *sub_add, float *sub_scale,
                       double *d_half, double *d_scale, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        half_tiny[i] = tiny[i] * 0.5f;                 // smallest normals -> subnormal results
        sub_add[i] = sub[i] + sub2[i];                 // subnormal operands, exact sum
        sub_scale[i] = sub[i] * 16777216.0f;           // subnormal * 2^24, exact normal result
        d_half[i] = dtiny[i] * 0.5;
        d_scale[i] = dsub[i] * 18014398509481984.0;    // subnormal * 2^54
    }
}

template <class T, class U> static T from_bits(U u) {
    T f;
    memcpy(&f, &u, sizeof f);
    return f;
}

int main() {
    std::vector<float> tiny(N), sub(N), sub2(N);
    std::vector<double> dtiny(N), dsub(N);
    CuportLcg g(ID);
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t s = g.next_u32() & 0x80000000u;
        uint32_t e = 1 + (g.next_u32() & 1);  // biased exponent 1 or 2
        tiny[i] = from_bits<float>(s | (e << 23) | (g.next_u32() >> 9));
        sub[i] = from_bits<float>((g.next_u32() & 0x807fffffu) | 1u);  // nonzero subnormal
        sub2[i] = from_bits<float>((g.next_u32() & 0x807fffffu) | 1u);
        uint64_t ds = (uint64_t)(g.next_u32() & 0x80000000u) << 32;
        uint64_t de = 1 + (g.next_u32() & 1);
        uint64_t hi = g.next_u32() & 0xfffffu;
        uint64_t m = (hi << 32) | g.next_u32();
        dtiny[i] = from_bits<double>(ds | (de << 52) | m);
        uint64_t ds2 = (uint64_t)(g.next_u32() & 0x80000000u) << 32;
        uint64_t hi2 = g.next_u32() & 0xfffffu;
        uint64_t m2 = (hi2 << 32) | g.next_u32() | 1u;
        dsub[i] = from_bits<double>(ds2 | m2);
    }

    float *f[6];
    double *d[4];
    for (auto &p : f) CUPORT_CHECK(cudaMalloc(&p, N * sizeof(float)));
    for (auto &p : d) CUPORT_CHECK(cudaMalloc(&p, N * sizeof(double)));
    CUPORT_CHECK(cudaMemcpy(f[0], tiny.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemcpy(f[1], sub.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemcpy(f[2], sub2.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemcpy(d[0], dtiny.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemcpy(d[1], dsub.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    denorm<<<(N + 127) / 128, 128>>>(f[0], f[1], f[2], d[0], d[1], f[3], f[4], f[5], d[2], d[3], N);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());

    std::vector<float> fo(N);
    std::vector<double> dout(N);
    const char *fnames[3] = {"half_tiny", "sub_add", "sub_scale"};
    for (int k = 0; k < 3; ++k) {
        CUPORT_CHECK(cudaMemcpy(fo.data(), f[3 + k], N * sizeof(float), cudaMemcpyDeviceToHost));
        cuport_emit_f32(fnames[k], fo.data(), N);
    }
    const char *dnames[2] = {"d_half", "d_scale"};
    for (int k = 0; k < 2; ++k) {
        CUPORT_CHECK(cudaMemcpy(dout.data(), d[2 + k], N * sizeof(double), cudaMemcpyDeviceToHost));
        cuport_emit_f64(dnames[k], dout.data(), N);
    }
    for (auto p : f) CUPORT_CHECK(cudaFree(p));
    for (auto p : d) CUPORT_CHECK(cudaFree(p));
    return cuport_done();
}
