// Fast-math full-range sweep driver.
// Usage: sweep <fn> <start> <count>
// fn in: __sinf __expf __fdividef sinf expf
// Unary functions take input bit pattern (start + i).
// __fdividef reads count (x, y) f32 pairs from stdin.
// Writes raw little-endian f32 outputs to stdout.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__global__ void kernel_sinf_fast(uint32_t start, uint32_t count, float *out) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint32_t bits = start + idx;
        float x;
        memcpy(&x, &bits, sizeof(float));
        out[idx] = __sinf(x);
    }
}

__global__ void kernel_sinf(uint32_t start, uint32_t count, float *out) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint32_t bits = start + idx;
        float x;
        memcpy(&x, &bits, sizeof(float));
        out[idx] = sinf(x);
    }
}

__global__ void kernel_expf_fast(uint32_t start, uint32_t count, float *out) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint32_t bits = start + idx;
        float x;
        memcpy(&x, &bits, sizeof(float));
        out[idx] = __expf(x);
    }
}

__global__ void kernel_expf(uint32_t start, uint32_t count, float *out) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint32_t bits = start + idx;
        float x;
        memcpy(&x, &bits, sizeof(float));
        out[idx] = expf(x);
    }
}

__global__ void kernel_fdividef(const float *in_pairs, uint32_t count, float *out) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float x = in_pairs[2 * idx];
        float y = in_pairs[2 * idx + 1];
        out[idx] = __fdividef(x, y);
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <fn> <start> <count>\n", argv[0]);
        return 1;
    }

    std::string fn = argv[1];
    uint64_t start_val = strtoull(argv[2], nullptr, 0);
    uint64_t count_val = strtoull(argv[3], nullptr, 0);

    if (count_val == 0) {
        return 0;
    }

    uint32_t count = static_cast<uint32_t>(count_val);
    uint32_t start = static_cast<uint32_t>(start_val);

    float *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out, count * sizeof(float)));

    const uint32_t threads = 256;
    const uint32_t blocks = (count + threads - 1) / threads;

    if (fn == "__fdividef") {
        // Read count (x, y) f32 pairs from stdin
        std::vector<float> h_in(count * 2);
        size_t needed = count * 2;
        size_t read_count = 0;
        while (read_count < needed) {
            size_t n = fread(h_in.data() + read_count, sizeof(float), needed - read_count, stdin);
            if (n == 0) {
                if (feof(stdin) || ferror(stdin)) {
                    fprintf(stderr, "Unexpected EOF or error reading stdin: got %zu, need %zu\n", read_count, needed);
                    return 2;
                }
            }
            read_count += n;
        }

        float *d_in = nullptr;
        CHECK_CUDA(cudaMalloc(&d_in, count * 2 * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), count * 2 * sizeof(float), cudaMemcpyHostToDevice));

        kernel_fdividef<<<blocks, threads>>>(d_in, count, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaFree(d_in));
    } else if (fn == "__sinf") {
        kernel_sinf_fast<<<blocks, threads>>>(start, count, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    } else if (fn == "sinf") {
        kernel_sinf<<<blocks, threads>>>(start, count, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    } else if (fn == "__expf") {
        kernel_expf_fast<<<blocks, threads>>>(start, count, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    } else if (fn == "expf") {
        kernel_expf<<<blocks, threads>>>(start, count, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    } else {
        fprintf(stderr, "Unknown function: %s\n", fn.c_str());
        CHECK_CUDA(cudaFree(d_out));
        return 1;
    }

    std::vector<float> h_out(count);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, count * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_out));

    size_t written = 0;
    while (written < count) {
        size_t n = fwrite(h_out.data() + written, sizeof(float), count - written, stdout);
        if (n == 0) {
            fprintf(stderr, "Error writing stdout\n");
            return 3;
        }
        written += n;
    }
    fflush(stdout);

    return 0;
}
