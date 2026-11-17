// cudaEventElapsedTime semantics that do not depend on how fast the GPU is:
// error codes, a zero self-interval, positive and monotonic intervals across a
// chain of events in one stream, and additivity of adjacent intervals.
// CUDA Runtime API, "Event Management": cudaEventElapsedTime, cudaEventQuery,
// cudaEventCreateWithFlags.
#include "../../common/cuport.h"
#include <cmath>
#include <vector>

static const char *ID = "streams-events/event-timing-001";
static const uint32_t N = 1u << 20;
static const uint32_t ROUNDS = 256;
static const int CHECKS = 8;

// Enough dependent integer work per element that each launch takes measurable time.
__global__ void work(uint32_t *data, uint32_t n, uint32_t key) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t x = data[i] ^ key;
        for (uint32_t r = 0; r < ROUNDS; ++r) x = (x ^ (x >> 15)) * 0x2c1b3c6du + r;
        data[i] = x;
    }
}

int main() {
    std::vector<uint32_t> h(N);
    cuport_fill_u32(ID, h.data(), N);
    uint32_t *d;
    CUPORT_CHECK(cudaMalloc(&d, N * sizeof(uint32_t)));
    CUPORT_CHECK(cudaMemcpy(d, h.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    cudaStream_t s;
    CUPORT_CHECK(cudaStreamCreate(&s));
    cudaEvent_t e[4], never, untimed;
    for (auto &ev : e) CUPORT_CHECK(cudaEventCreate(&ev));
    CUPORT_CHECK(cudaEventCreate(&never));  // never recorded
    CUPORT_CHECK(cudaEventCreateWithFlags(&untimed, cudaEventDisableTiming));

    // Warm-up launch, so the first timed interval does not include module loading.
    work<<<N / 256, 256, 0, s>>>(d, N, 0);
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaStreamSynchronize(s));

    CUPORT_CHECK(cudaEventRecord(e[0], s));
    for (uint32_t k = 1; k < 4; ++k) {
        work<<<N / 256, 256, 0, s>>>(d, N, k);
        CUPORT_CHECK(cudaGetLastError());
        CUPORT_CHECK(cudaEventRecord(e[k], s));
    }
    CUPORT_CHECK(cudaEventRecord(untimed, s));
    CUPORT_CHECK(cudaEventSynchronize(e[3]));
    CUPORT_CHECK(cudaEventSynchronize(untimed));

    float t00 = -1, t01, t12, t23, t02, t03, dummy = 0;
    uint32_t checks[CHECKS];
    checks[0] = (uint32_t)cudaEventQuery(e[3]);                   // cudaSuccess once complete
    checks[1] = (uint32_t)cudaEventElapsedTime(&t00, e[0], e[0]);  // cudaSuccess
    CUPORT_CHECK(cudaEventElapsedTime(&t01, e[0], e[1]));
    CUPORT_CHECK(cudaEventElapsedTime(&t12, e[1], e[2]));
    CUPORT_CHECK(cudaEventElapsedTime(&t23, e[2], e[3]));
    CUPORT_CHECK(cudaEventElapsedTime(&t02, e[0], e[2]));
    CUPORT_CHECK(cudaEventElapsedTime(&t03, e[0], e[3]));
    checks[2] = t00 == 0.0f;
    checks[3] = t01 > 0.0f && t12 > 0.0f && t23 > 0.0f;
    checks[4] = t01 <= t02 && t02 <= t03;
    // Resolution is about 0.5 us per event, so three intervals add to within 2 us.
    checks[5] = std::fabs((t01 + t12 + t23) - t03) <= 0.002f + 1e-4f * t03;
    checks[6] = (uint32_t)cudaEventElapsedTime(&dummy, e[0], never);    // never recorded
    checks[7] = (uint32_t)cudaEventElapsedTime(&dummy, e[0], untimed);  // timing disabled
    (void)cudaGetLastError();  // clear the two expected errors

    cuport_emit_u32("checks", checks, CHECKS);
    printf("timing events_ms t01=%.4f t12=%.4f t23=%.4f t03=%.4f\n", t01, t12, t23, t03);

    for (auto ev : e) CUPORT_CHECK(cudaEventDestroy(ev));
    CUPORT_CHECK(cudaEventDestroy(never));
    CUPORT_CHECK(cudaEventDestroy(untimed));
    CUPORT_CHECK(cudaStreamDestroy(s));
    CUPORT_CHECK(cudaFree(d));
    return cuport_done();
}
