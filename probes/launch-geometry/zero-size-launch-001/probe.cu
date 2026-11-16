// A launch with gridDim 0 must fail with cudaErrorInvalidConfiguration, run
// nothing, and leave the context usable for the next launch.
#include "../../common/cuport.h"
#include <vector>

__global__ void touch(uint32_t *d) { d[threadIdx.x] = 0x600DF00Du; }

int main() {
    uint32_t *d;
    CUPORT_CHECK(cudaMalloc(&d, 32 * 4));
    CUPORT_CHECK(cudaMemset(d, 0xAB, 32 * 4));

    touch<<<0, 32>>>(d);
    int32_t zero_launch = (int32_t)cudaGetLastError();
    int32_t sync_after = (int32_t)cudaDeviceSynchronize();
    std::vector<uint32_t> after_zero(32), after_one(32);
    CUPORT_CHECK(cudaMemcpy(after_zero.data(), d, 32 * 4, cudaMemcpyDeviceToHost));

    touch<<<1, 32>>>(d);
    int32_t one_launch = (int32_t)cudaGetLastError();
    CUPORT_CHECK(cudaDeviceSynchronize());
    CUPORT_CHECK(cudaMemcpy(after_one.data(), d, 32 * 4, cudaMemcpyDeviceToHost));

    int32_t codes[3] = {zero_launch, sync_after, one_launch};
    uint32_t failed = zero_launch != 0;
    cuport_emit_u32("zero_launch_failed", &failed, 1);
    cuport_emit_i32("codes", codes, 3);
    cuport_emit_u32("after_zero", after_zero.data(), 32);
    cuport_emit_u32("after_one", after_one.data(), 32);
    return cuport_done();
}
