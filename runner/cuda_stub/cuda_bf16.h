// Declarations only, for `cuport check --syntax`. See cuda_runtime.h.
#pragma once
#include "cuda_runtime.h"

struct __nv_bfloat16 { unsigned short x; };
__host__ __device__ __nv_bfloat16 __float2bfloat16_rn(float f);
__host__ __device__ __nv_bfloat16 __float2bfloat16_rz(float f);
__host__ __device__ __nv_bfloat16 __float2bfloat16_ru(float f);
__host__ __device__ __nv_bfloat16 __float2bfloat16_rd(float f);
__host__ __device__ float __bfloat162float(__nv_bfloat16 b);
__device__ unsigned short __bfloat16_as_ushort(__nv_bfloat16 b);
__device__ __nv_bfloat16 __ushort_as_bfloat16(unsigned short u);
