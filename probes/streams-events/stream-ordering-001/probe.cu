// Verifies in-stream sequential execution order of 64 dependent kernel launches.
#include "../../common/cuport.h"
#include <algorithm>
#include <vector>

static const char *ID = "streams-events/stream-ordering-001";
static const size_t N = 65536;
static const uint32_t KERNEL_COUNT = 64;
static const uint32_t MULTIPLIER = 0x9E3779B9u;
static const int TIMING_REPS = 20;

__global__ void add_k(uint32_t *data, uint32_t n, uint32_t delta) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] += delta;
    }
}

int main() {
    std::vector<uint32_t> h_data(N);
    cuport_fill_u32(ID, h_data.data(), N);

    cudaStream_t stream;
    CUPORT_CHECK(cudaStreamCreate(&stream));

    uint32_t *d_data = nullptr;
    CUPORT_CHECK(cudaMalloc(&d_data, N * sizeof(uint32_t)));
    CUPORT_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    const uint32_t threads = 256;
    const uint32_t blocks = (N + threads - 1) / threads;

    // Correctness run: 64 dependent launches in one non-default stream
    for (uint32_t k = 0; k < KERNEL_COUNT; ++k) {
        add_k<<<blocks, threads, 0, stream>>>(d_data, N, k * MULTIPLIER);
    }
    CUPORT_CHECK(cudaGetLastError());

    std::vector<uint32_t> out(N);
    CUPORT_CHECK(cudaMemcpyAsync(out.data(), d_data, N * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUPORT_CHECK(cudaStreamSynchronize(stream));

    cuport_emit_u32("out", out.data(), N);

    // Timed repetitions of the 64-launch chain with cudaEvents
    cudaEvent_t start, stop;
    CUPORT_CHECK(cudaEventCreate(&start));
    CUPORT_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(TIMING_REPS);

    for (int rep = 0; rep < TIMING_REPS; ++rep) {
        CUPORT_CHECK(cudaEventRecord(start, stream));
        for (uint32_t k = 0; k < KERNEL_COUNT; ++k) {
            add_k<<<blocks, threads, 0, stream>>>(d_data, N, k * MULTIPLIER);
        }
        CUPORT_CHECK(cudaEventRecord(stop, stream));
        CUPORT_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUPORT_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    std::sort(times_ms.begin(), times_ms.end());
    float min_ms = times_ms.front();
    float max_ms = times_ms.back();
    float median_ms = 0.5f * (times_ms[TIMING_REPS / 2 - 1] + times_ms[TIMING_REPS / 2]);

    printf("timing chain64_ms median=%.3f min=%.3f max=%.3f reps=%d\n", median_ms, min_ms, max_ms, TIMING_REPS);

    CUPORT_CHECK(cudaEventDestroy(start));
    CUPORT_CHECK(cudaEventDestroy(stop));
    CUPORT_CHECK(cudaFree(d_data));
    CUPORT_CHECK(cudaStreamDestroy(stream));

    return cuport_done();
}
