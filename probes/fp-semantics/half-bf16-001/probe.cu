// float <-> half and float <-> bfloat16 conversions in each rounding mode.
// CUDA Math API, "Half Precision Conversion and Data Movement" and "Bfloat16
// Precision Conversion and Data Movement": __float2half_{rn,rz,ru,rd},
// __float2bfloat16_{rn,rz}, __half2float, __bfloat162float.
#include "../../common/cuport.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>

static const char *ID = "fp-semantics/half-bf16-001";
static const uint32_t N = 2048;

// Edge cases first in both inputs: signed zeros, infinities, the half overflow
// boundary (65504, the 65520 tie), half subnormal edges, half and bf16 ties,
// float max and the smallest float subnormals.
static const uint32_t EDGES[] = {
    0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x477fe000u, 0x477ff000u, 0x477fefffu,
    0xc77ff000u, 0x33800000u, 0x33000000u, 0x33400000u, 0x387fc000u, 0x38800000u, 0x3f801000u,
    0x3f803000u, 0x7f7fffffu, 0x7f7f8000u, 0x00000001u, 0x00008000u, 0x00018000u, 0x3f808000u,
    0x3f818000u, 0xff7f8000u, 0x80018000u,
};
static const uint32_t N_EDGES = sizeof(EDGES) / sizeof(EDGES[0]);

__global__ void convert(const float *x, const float *z, uint16_t *h, float *h_back, uint16_t *bf,
                        float *bf_back, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        h[0 * n + i] = __half_as_ushort(__float2half_rn(x[i]));
        h[1 * n + i] = __half_as_ushort(__float2half_rz(x[i]));
        h[2 * n + i] = __half_as_ushort(__float2half_ru(x[i]));
        h[3 * n + i] = __half_as_ushort(__float2half_rd(x[i]));
        h_back[i] = __half2float(__float2half_rn(x[i]));
        bf[0 * n + i] = __bfloat16_as_ushort(__float2bfloat16_rn(z[i]));
        bf[1 * n + i] = __bfloat16_as_ushort(__float2bfloat16_rz(z[i]));
        bf_back[i] = __bfloat162float(__float2bfloat16_rn(z[i]));
    }
}

static float from_bits(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

int main() {
    // x: half range, biased float exponent 101..144 (below the smallest half
    // subnormal to above the largest half). z: any finite float.
    std::vector<float> x(N), z(N);
    CuportLcg g(ID);
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t s = g.next_u32() & 0x80000000u;
        uint32_t e = 101 + g.next_u32() % 44;
        uint32_t m = g.next_u32() >> 9;
        uint32_t zb = g.next_u32();
        if ((zb & 0x7f800000u) == 0x7f800000u) zb &= 0xbfffffffu;  // no inf/NaN
        x[i] = from_bits(i < N_EDGES ? EDGES[i] : (s | (e << 23) | m));
        z[i] = from_bits(i < N_EDGES ? EDGES[i] : zb);
    }

    float *dx, *dz, *dhb, *dbb;
    uint16_t *dh, *dbf;
    CUPORT_CHECK(cudaMalloc(&dx, N * sizeof(float)));
    CUPORT_CHECK(cudaMalloc(&dz, N * sizeof(float)));
    CUPORT_CHECK(cudaMalloc(&dhb, N * sizeof(float)));
    CUPORT_CHECK(cudaMalloc(&dbb, N * sizeof(float)));
    CUPORT_CHECK(cudaMalloc(&dh, 4 * N * sizeof(uint16_t)));
    CUPORT_CHECK(cudaMalloc(&dbf, 2 * N * sizeof(uint16_t)));
    CUPORT_CHECK(cudaMemcpy(dx, x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUPORT_CHECK(cudaMemcpy(dz, z.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    convert<<<(N + 127) / 128, 128>>>(dx, dz, dh, dhb, dbf, dbb, N);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h(4 * N), bf(2 * N);
    std::vector<float> hb(N), bb(N);
    CUPORT_CHECK(cudaMemcpy(h.data(), dh, 4 * N * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(bf.data(), dbf, 2 * N * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(hb.data(), dhb, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(bb.data(), dbb, N * sizeof(float), cudaMemcpyDeviceToHost));

    cuport_emit_u16("h_rn", &h[0 * N], N);
    cuport_emit_u16("h_rz", &h[1 * N], N);
    cuport_emit_u16("h_ru", &h[2 * N], N);
    cuport_emit_u16("h_rd", &h[3 * N], N);
    cuport_emit_f32("h_back", hb.data(), N);
    cuport_emit_u16("bf_rn", &bf[0 * N], N);
    cuport_emit_u16("bf_rz", &bf[1 * N], N);
    cuport_emit_f32("bf_back", bb.data(), N);

    CUPORT_CHECK(cudaFree(dx));
    CUPORT_CHECK(cudaFree(dz));
    CUPORT_CHECK(cudaFree(dhb));
    CUPORT_CHECK(cudaFree(dbb));
    CUPORT_CHECK(cudaFree(dh));
    CUPORT_CHECK(cudaFree(dbf));
    return cuport_done();
}
