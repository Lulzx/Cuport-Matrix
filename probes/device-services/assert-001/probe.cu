// Device-side assert. Phase A: the assertion holds in every thread, the
// kernel completes and its output is correct. Phase B: exactly one thread
// trips the assertion; cudaDeviceSynchronize must then report
// cudaErrorAssert (710). The context is unusable after a device assert, so
// phase B is last and nothing after it is wrapped in CUPORT_CHECK.
#include "../../common/cuport.h"
#include <cassert>
#include <vector>

static const char *ID = "device-services/assert-001";

__global__ void guarded(const uint32_t *v, uint32_t *out, uint32_t bad_index) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    assert(i != bad_index);
    out[i] = v[i] ^ 0x5A5A5A5Au;
}

int main() {
    const uint32_t blocks = 2, threads = 32, n = blocks * threads;
    std::vector<uint32_t> in(n), out(n);
    CuportLcg g(ID);
    for (uint32_t i = 0; i < n; ++i) in[i] = g.next_u32();
    uint32_t bad = g.next_u32() % n;

    uint32_t *d_in, *d_out;
    CUPORT_CHECK(cudaMalloc(&d_in, n * 4));
    CUPORT_CHECK(cudaMalloc(&d_out, n * 4));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));

    guarded<<<blocks, threads>>>(d_in, d_out, 0xFFFFFFFFu);  // no thread matches
    CUPORT_CHECK(cudaGetLastError());
    int32_t code_a = (int32_t)cudaDeviceSynchronize();
    CUPORT_CHECK(cudaMemcpy(out.data(), d_out, n * 4, cudaMemcpyDeviceToHost));

    guarded<<<blocks, threads>>>(d_in, d_out, bad);  // exactly one thread trips
    int32_t launch_b = (int32_t)cudaGetLastError();
    int32_t code_b = (int32_t)cudaDeviceSynchronize();

    int32_t codes[3] = {code_a, launch_b, code_b};
    uint32_t tripped = code_b != 0;
    cuport_emit_u32("tripped", &tripped, 1);
    cuport_emit_i32("codes", codes, 3);
    cuport_emit_u32("out_a", out.data(), n);
    return cuport_done();
}
