// Pipelined async memory transfers and kernel execution across two non-default streams.
#include "../../common/cuport.h"
#include <algorithm>
#include <vector>

static const char *ID = "streams-events/async-memcpy-overlap-001";
static const size_t N = 4194304; // 16 MiB of uint32_t
static const size_t NUM_CHUNKS = 8;
static const size_t CHUNK_ELEMS = N / NUM_CHUNKS; // 524288
static const size_t CHUNK_BYTES = CHUNK_ELEMS * sizeof(uint32_t);
static const size_t SAMPLE_COUNT = 4096;
static const size_t SAMPLE_STRIDE = 1024;
static const uint32_t MULTIPLIER = 2654435761u;
static const int TIMING_REPS = 20;

__global__ void transform_chunk(const uint32_t *in, uint32_t *out, uint32_t n, uint32_t chunk_idx) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * MULTIPLIER + chunk_idx;
    }
}

int main() {
    uint32_t *h_in = nullptr;
    uint32_t *h_out = nullptr;
    CUPORT_CHECK(cudaMallocHost(&h_in, N * sizeof(uint32_t)));
    CUPORT_CHECK(cudaMallocHost(&h_out, N * sizeof(uint32_t)));
    cuport_fill_u32(ID, h_in, N);

    uint32_t *d_in = nullptr;
    uint32_t *d_out = nullptr;
    CUPORT_CHECK(cudaMalloc(&d_in, N * sizeof(uint32_t)));
    CUPORT_CHECK(cudaMalloc(&d_out, N * sizeof(uint32_t)));

    cudaStream_t streams[2];
    CUPORT_CHECK(cudaStreamCreate(&streams[0]));
    CUPORT_CHECK(cudaStreamCreate(&streams[1]));

    const uint32_t threads = 256;
    const uint32_t blocks = (CHUNK_ELEMS + threads - 1) / threads;

    // Correctness run: 8 chunks across 2 streams
    for (size_t c = 0; c < NUM_CHUNKS; ++c) {
        size_t offset = c * CHUNK_ELEMS;
        cudaStream_t s = streams[c % 2];
        CUPORT_CHECK(cudaMemcpyAsync(d_in + offset, h_in + offset, CHUNK_BYTES, cudaMemcpyHostToDevice, s));
        transform_chunk<<<blocks, threads, 0, s>>>(d_in + offset, d_out + offset, (uint32_t)CHUNK_ELEMS, (uint32_t)c);
        CUPORT_CHECK(cudaMemcpyAsync(h_out + offset, d_out + offset, CHUNK_BYTES, cudaMemcpyDeviceToHost, s));
    }
    CUPORT_CHECK(cudaGetLastError());
    CUPORT_CHECK(cudaDeviceSynchronize());

    uint32_t count_wrong = 0;
    for (size_t i = 0; i < N; ++i) {
        uint32_t c = (uint32_t)(i / CHUNK_ELEMS);
        uint32_t expected = h_in[i] * MULTIPLIER + c;
        if (h_out[i] != expected) {
            count_wrong++;
        }
    }

    std::vector<uint32_t> sample(SAMPLE_COUNT);
    for (size_t j = 0; j < SAMPLE_COUNT; ++j) {
        sample[j] = h_out[j * SAMPLE_STRIDE];
    }

    cuport_emit_u32("count_wrong", &count_wrong, 1);
    cuport_emit_u32("sample", sample.data(), SAMPLE_COUNT);

    // Timing 20 reps serial (single stream)
    cudaEvent_t start, stop, stop1;
    CUPORT_CHECK(cudaEventCreate(&start));
    CUPORT_CHECK(cudaEventCreate(&stop));
    CUPORT_CHECK(cudaEventCreate(&stop1));

    std::vector<float> serial_times_ms;
    serial_times_ms.reserve(TIMING_REPS);

    for (int rep = 0; rep < TIMING_REPS; ++rep) {
        CUPORT_CHECK(cudaEventRecord(start, streams[0]));
        for (size_t c = 0; c < NUM_CHUNKS; ++c) {
            size_t offset = c * CHUNK_ELEMS;
            CUPORT_CHECK(cudaMemcpyAsync(d_in + offset, h_in + offset, CHUNK_BYTES, cudaMemcpyHostToDevice, streams[0]));
            transform_chunk<<<blocks, threads, 0, streams[0]>>>(d_in + offset, d_out + offset, (uint32_t)CHUNK_ELEMS, (uint32_t)c);
            CUPORT_CHECK(cudaMemcpyAsync(h_out + offset, d_out + offset, CHUNK_BYTES, cudaMemcpyDeviceToHost, streams[0]));
        }
        CUPORT_CHECK(cudaEventRecord(stop, streams[0]));
        CUPORT_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUPORT_CHECK(cudaEventElapsedTime(&ms, start, stop));
        serial_times_ms.push_back(ms);
    }

    // Timing 20 reps two concurrent streams
    std::vector<float> twostream_times_ms;
    twostream_times_ms.reserve(TIMING_REPS);

    for (int rep = 0; rep < TIMING_REPS; ++rep) {
        CUPORT_CHECK(cudaEventRecord(start, streams[0]));
        CUPORT_CHECK(cudaStreamWaitEvent(streams[1], start, 0));

        for (size_t c = 0; c < NUM_CHUNKS; ++c) {
            size_t offset = c * CHUNK_ELEMS;
            cudaStream_t s = streams[c % 2];
            CUPORT_CHECK(cudaMemcpyAsync(d_in + offset, h_in + offset, CHUNK_BYTES, cudaMemcpyHostToDevice, s));
            transform_chunk<<<blocks, threads, 0, s>>>(d_in + offset, d_out + offset, (uint32_t)CHUNK_ELEMS, (uint32_t)c);
            CUPORT_CHECK(cudaMemcpyAsync(h_out + offset, d_out + offset, CHUNK_BYTES, cudaMemcpyDeviceToHost, s));
        }

        CUPORT_CHECK(cudaEventRecord(stop1, streams[1]));
        CUPORT_CHECK(cudaStreamWaitEvent(streams[0], stop1, 0));
        CUPORT_CHECK(cudaEventRecord(stop, streams[0]));
        CUPORT_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUPORT_CHECK(cudaEventElapsedTime(&ms, start, stop));
        twostream_times_ms.push_back(ms);
    }

    std::sort(serial_times_ms.begin(), serial_times_ms.end());
    float serial_min = serial_times_ms.front();
    float serial_max = serial_times_ms.back();
    float serial_median = 0.5f * (serial_times_ms[TIMING_REPS / 2 - 1] + serial_times_ms[TIMING_REPS / 2]);

    std::sort(twostream_times_ms.begin(), twostream_times_ms.end());
    float twostream_min = twostream_times_ms.front();
    float twostream_max = twostream_times_ms.back();
    float twostream_median = 0.5f * (twostream_times_ms[TIMING_REPS / 2 - 1] + twostream_times_ms[TIMING_REPS / 2]);

    float overlap_ratio = (twostream_median > 0.0f) ? (serial_median / twostream_median) : 1.0f;

    printf("timing serial_ms median=%.3f min=%.3f max=%.3f reps=%d\n", serial_median, serial_min, serial_max, TIMING_REPS);
    printf("timing twostream_ms median=%.3f min=%.3f max=%.3f reps=%d\n", twostream_median, twostream_min, twostream_max, TIMING_REPS);
    printf("timing overlap_ratio=%.3f\n", overlap_ratio);

    CUPORT_CHECK(cudaEventDestroy(start));
    CUPORT_CHECK(cudaEventDestroy(stop));
    CUPORT_CHECK(cudaEventDestroy(stop1));
    CUPORT_CHECK(cudaFree(d_in));
    CUPORT_CHECK(cudaFree(d_out));
    CUPORT_CHECK(cudaStreamDestroy(streams[0]));
    CUPORT_CHECK(cudaStreamDestroy(streams[1]));
    CUPORT_CHECK(cudaFreeHost(h_in));
    CUPORT_CHECK(cudaFreeHost(h_out));

    return cuport_done();
}
