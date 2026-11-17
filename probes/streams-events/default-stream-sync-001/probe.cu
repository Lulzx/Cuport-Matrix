// Legacy default stream synchronisation. Work in the legacy default stream
// (stream 0) waits for all earlier work in blocking streams, and later work in
// blocking streams waits for it. No explicit synchronisation is used between
// the three kernels below; only these implicit barriers order them.
// CUDA Runtime API, "API synchronization behavior" and "Stream synchronization
// behavior" (legacy default stream); CUDA C++ Programming Guide, "Default Stream".
#include "../../common/cuport.h"
#include <vector>

static const char *ID = "streams-events/default-stream-sync-001";
static const uint32_t N = 65536;
static const uint32_t ROUNDS = 4096;

__device__ uint32_t mix(uint32_t x, uint32_t rounds) {
    for (uint32_t r = 0; r < rounds; ++r) x = (x ^ (x >> 15)) * 0x2c1b3c6du + r;
    return x;
}

// stage_a and stage_b are slow and run on a small grid-stride grid, so most of the
// GPU stays idle while they run: a consumer that did not wait would be scheduled
// at once and read zeros or partial results.
__global__ void stage_a(const uint32_t *in, uint32_t *x, uint32_t n) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        x[i] = mix(in[i], ROUNDS);
}

__global__ void stage_b(const uint32_t *x, uint32_t *y, uint32_t n) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[i] = mix(x[i] ^ 0x9e3779b9u, ROUNDS);
}

__global__ void stage_c(const uint32_t *y, uint32_t *z, uint32_t n) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        z[i] = y[i] ^ 0xa5a5a5a5u;
}

int main() {
    std::vector<uint32_t> in(N);
    cuport_fill_u32(ID, in.data(), N);
    uint32_t *d_in, *d_x, *d_y, *d_z;
    for (uint32_t **p : {&d_in, &d_x, &d_y, &d_z}) CUPORT_CHECK(cudaMalloc(p, N * sizeof(uint32_t)));
    CUPORT_CHECK(cudaMemcpy(d_in, in.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    for (uint32_t *p : {d_x, d_y, d_z}) CUPORT_CHECK(cudaMemset(p, 0, N * sizeof(uint32_t)));
    CUPORT_CHECK(cudaDeviceSynchronize());

    cudaStream_t s;
    CUPORT_CHECK(cudaStreamCreate(&s));  // a blocking stream: it synchronises with stream 0

    const uint32_t threads = 256, blocks = 8;  // 2048 threads, 32 elements each
    stage_a<<<blocks, threads, 0, s>>>(d_in, d_x, N);  // blocking stream
    stage_b<<<blocks, threads, 0, 0>>>(d_x, d_y, N);   // legacy default stream: waits for stage_a
    stage_c<<<blocks, threads, 0, s>>>(d_y, d_z, N);   // blocking stream: waits for stage_b
    CUPORT_CHECK(cudaGetLastError());

    // cudaMemcpy runs in the legacy default stream, so it waits for stage_c.
    std::vector<uint32_t> x(N), y(N), z(N);
    CUPORT_CHECK(cudaMemcpy(z.data(), d_z, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(y.data(), d_y, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUPORT_CHECK(cudaMemcpy(x.data(), d_x, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cuport_emit_u32("x", x.data(), N);
    cuport_emit_u32("y", y.data(), N);
    cuport_emit_u32("z", z.data(), N);

    CUPORT_CHECK(cudaStreamDestroy(s));
    for (uint32_t *p : {d_in, d_x, d_y, d_z}) CUPORT_CHECK(cudaFree(p));
    return cuport_done();
}
