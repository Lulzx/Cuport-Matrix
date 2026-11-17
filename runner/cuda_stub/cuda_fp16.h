// Declarations only, for `cuport check --syntax`. See cuda_runtime.h.
#pragma once
#include "cuda_runtime.h"

struct __half { unsigned short x; };
__host__ __device__ __half __float2half_rn(float f);
__host__ __device__ __half __float2half_rz(float f);
__host__ __device__ __half __float2half_ru(float f);
__host__ __device__ __half __float2half_rd(float f);
__host__ __device__ float __half2float(__half h);
__device__ unsigned short __half_as_ushort(__half h);
__device__ __half __ushort_as_half(unsigned short u);
__device__ __half __hadd(__half a, __half b);
__device__ __half __hmul(__half a, __half b);
