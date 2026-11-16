// Declarations only, for `cuport check --syntax`. See cuda_runtime.h.
#pragma once
#include "cuda_runtime.h"

namespace cooperative_groups {
struct thread_group {
    __device__ void sync() const;
    __device__ unsigned thread_rank() const;
    __device__ unsigned size() const;
};
struct grid_group : thread_group {
    __device__ bool is_valid() const;
};
struct thread_block : thread_group {};
__device__ grid_group this_grid();
__device__ thread_block this_thread_block();
}
