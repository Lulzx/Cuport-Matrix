// Declarations only, for `cuport check --syntax`: lets clang's CUDA mode parse
// probes (host and device side) on a machine with no CUDA toolkit. Nothing
// here is implemented and nothing compiled against it can run. Add a
// declaration when a probe needs one; never add tool-specific extensions.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __forceinline__ __inline__ __attribute__((always_inline))

#include <__clang_cuda_builtin_vars.h>

struct dim3 {
    unsigned x, y, z;
    __host__ __device__ constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorNotReady = 600,
    cudaErrorAssert = 710,
    cudaErrorLaunchFailure = 719,
    cudaErrorNotSupported = 801,
} cudaError_t;

enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8 };
enum cudaDeviceAttr { cudaDevAttrMaxSharedMemoryPerBlock = 8, cudaDevAttrMaxSharedMemoryPerBlockOptin = 97, cudaDevAttrWarpSize = 10, cudaDevAttrCooperativeLaunch = 95 };
enum cudaLimit { cudaLimitStackSize, cudaLimitPrintfFifoSize, cudaLimitMallocHeapSize };

typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;
#define cudaStreamNonBlocking 0x01
#define cudaEventDefault 0x00
#define cudaEventDisableTiming 0x02

// ---- texture objects (CUDA 12 object API; references were removed) --------
typedef unsigned long long cudaTextureObject_t;
enum cudaChannelFormatKind { cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned, cudaChannelFormatKindFloat };
struct cudaChannelFormatDesc { int x, y, z, w; cudaChannelFormatKind f; };
template <class T> cudaChannelFormatDesc cudaCreateChannelDesc(void);
enum cudaResourceType { cudaResourceTypeArray, cudaResourceTypeMipmappedArray, cudaResourceTypeLinear, cudaResourceTypePitch2D };
struct cudaResourceDesc {
    cudaResourceType resType;
    union {
        struct { void *devPtr; cudaChannelFormatDesc desc; size_t sizeInBytes; } linear;
        struct { void *devPtr; cudaChannelFormatDesc desc; size_t width, height, pitchInBytes; } pitch2D;
    } res;
};
enum cudaTextureAddressMode { cudaAddressModeWrap, cudaAddressModeClamp, cudaAddressModeBorder, cudaAddressModeMirror };
enum cudaTextureFilterMode { cudaFilterModePoint, cudaFilterModeLinear };
enum cudaTextureReadMode { cudaReadModeElementType, cudaReadModeNormalizedFloat };
struct cudaTextureDesc {
    cudaTextureAddressMode addressMode[3];
    cudaTextureFilterMode filterMode;
    cudaTextureReadMode readMode;
    int sRGB, normalizedCoords;
    float borderColor[4];
};
struct cudaResourceViewDesc;

struct cudaDeviceProp {
    char name[256];
    size_t sharedMemPerBlock, sharedMemPerBlockOptin, totalGlobalMem;
    int warpSize, maxThreadsPerBlock, maxThreadsDim[3], maxGridSize[3], major, minor, multiProcessorCount, cooperativeLaunch;
};

extern "C" {
unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0);
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0);
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
cudaError_t cudaMalloc(void **p, size_t n);
cudaError_t cudaMallocHost(void **p, size_t n);
cudaError_t cudaMallocManaged(void **p, size_t n, unsigned flags = 1);
cudaError_t cudaFree(void *p);
cudaError_t cudaFreeHost(void *p);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t n, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t n, cudaMemcpyKind kind, cudaStream_t s = 0);
cudaError_t cudaMemset(void *p, int v, size_t n);
cudaError_t cudaMemsetAsync(void *p, int v, size_t n, cudaStream_t s = 0);
cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t n, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char *cudaGetErrorString(cudaError_t e);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device);
cudaError_t cudaFuncSetAttribute(const void *func, cudaFuncAttribute attr, int value);
cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value);
cudaError_t cudaStreamCreate(cudaStream_t *s);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *s, unsigned flags);
cudaError_t cudaStreamDestroy(cudaStream_t s);
cudaError_t cudaStreamSynchronize(cudaStream_t s);
cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned flags = 0);
cudaError_t cudaEventCreate(cudaEvent_t *e);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *e, unsigned flags);
cudaError_t cudaEventDestroy(cudaEvent_t e);
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t e);
cudaError_t cudaEventQuery(cudaEvent_t e);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaCreateTextureObject(cudaTextureObject_t *tex, const cudaResourceDesc *res, const cudaTextureDesc *desc, const cudaResourceViewDesc *view);
cudaError_t cudaDestroyTextureObject(cudaTextureObject_t tex);
}

// Device-side launch (dynamic parallelism): what nvcc lowers <<<>>> to in
// device code when -rdc=true and cudadevrt are given.
extern "C" {
__device__ void *cudaGetParameterBufferV2(void *func, dim3 gridDim, dim3 blockDim, unsigned sharedMem);
__device__ cudaError_t cudaLaunchDeviceV2(void *buf, cudaStream_t stream);
__device__ cudaError_t cudaGetParameterBuffer(size_t alignment, size_t size, void **buf);
__device__ cudaError_t cudaLaunchDevice(void *func, void *buf, dim3 gridDim, dim3 blockDim, unsigned sharedMem, cudaStream_t stream);
__device__ unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
}

template <class T> static inline cudaError_t cudaMalloc(T **p, size_t n) { return cudaMalloc((void **)p, n); }
template <class T> static inline cudaError_t cudaMallocHost(T **p, size_t n) { return cudaMallocHost((void **)p, n); }
template <class T> static inline cudaError_t cudaMallocManaged(T **p, size_t n, unsigned f = 1) { return cudaMallocManaged((void **)p, n, f); }
template <class T> static inline cudaError_t cudaFuncSetAttribute(T *f, cudaFuncAttribute a, int v) { return cudaFuncSetAttribute((const void *)f, a, v); }
template <class T> static inline cudaError_t cudaMemcpyToSymbol(const T &sym, const void *src, size_t n, size_t off = 0, cudaMemcpyKind k = cudaMemcpyHostToDevice) { return cudaMemcpyToSymbol((const void *)&sym, src, n, off, k); }

// ---- device functions ------------------------------------------------------
__device__ void __syncthreads(void);
__device__ int __syncthreads_count(int pred);
__device__ int __syncthreads_and(int pred);
__device__ int __syncthreads_or(int pred);
__device__ void __syncwarp(unsigned mask = 0xffffffffu);
__device__ void __threadfence(void);
__device__ void __threadfence_block(void);
__device__ unsigned __activemask(void);
__device__ unsigned __ballot_sync(unsigned mask, int pred);
__device__ int __any_sync(unsigned mask, int pred);
__device__ int __all_sync(unsigned mask, int pred);
__device__ unsigned __match_any_sync(unsigned mask, unsigned v);
__device__ unsigned __match_any_sync(unsigned mask, int v);
__device__ unsigned __match_any_sync(unsigned mask, unsigned long long v);
template <class T> __device__ T __shfl_sync(unsigned mask, T v, int src, int width = 32);
template <class T> __device__ T __shfl_up_sync(unsigned mask, T v, unsigned delta, int width = 32);
template <class T> __device__ T __shfl_down_sync(unsigned mask, T v, unsigned delta, int width = 32);
template <class T> __device__ T __shfl_xor_sync(unsigned mask, T v, int lane_mask, int width = 32);
__device__ int __popc(unsigned v);
__device__ int __ffs(int v);
__device__ int __clz(int v);

__device__ int atomicAdd(int *a, int v);
__device__ unsigned atomicAdd(unsigned *a, unsigned v);
__device__ unsigned long long atomicAdd(unsigned long long *a, unsigned long long v);
__device__ float atomicAdd(float *a, float v);
__device__ double atomicAdd(double *a, double v);
__device__ int atomicSub(int *a, int v);
__device__ int atomicExch(int *a, int v);
__device__ unsigned atomicExch(unsigned *a, unsigned v);
__device__ int atomicCAS(int *a, int cmp, int v);
__device__ unsigned atomicCAS(unsigned *a, unsigned cmp, unsigned v);
__device__ unsigned long long atomicCAS(unsigned long long *a, unsigned long long cmp, unsigned long long v);
__device__ int atomicMin(int *a, int v);
__device__ unsigned atomicMin(unsigned *a, unsigned v);
__device__ unsigned long long atomicMin(unsigned long long *a, unsigned long long v);
__device__ int atomicMax(int *a, int v);
__device__ unsigned atomicMax(unsigned *a, unsigned v);
__device__ unsigned long long atomicMax(unsigned long long *a, unsigned long long v);
__device__ unsigned atomicInc(unsigned *a, unsigned v);
__device__ int atomicAnd(int *a, int v);
__device__ int atomicOr(int *a, int v);

template <class T> __device__ T tex1Dfetch(cudaTextureObject_t tex, int x);
__device__ long long clock64(void);
__device__ long long clock(void);
__device__ unsigned long long __globaltimer(void);
extern "C" __device__ int printf(const char *, ...);
extern "C" __device__ void *malloc(size_t);
extern "C" __device__ void free(void *);

__device__ float __fmaf_rn(float a, float b, float c);
__device__ float __fmul_rn(float a, float b);
__device__ float __fadd_rn(float a, float b);
__device__ float __fadd_rz(float a, float b);
__device__ float __fadd_ru(float a, float b);
__device__ float __fadd_rd(float a, float b);
__device__ float __fdiv_rn(float a, float b);
__device__ float __fdiv_rz(float a, float b);
__device__ float __fdiv_ru(float a, float b);
__device__ float __fdiv_rd(float a, float b);
__device__ float __fsqrt_rn(float a);
__device__ float __fsqrt_rz(float a);
__device__ float __fsqrt_ru(float a);
__device__ float __fsqrt_rd(float a);
__device__ float __fmul_rz(float a, float b);
__device__ float __fmul_ru(float a, float b);
__device__ float __fmul_rd(float a, float b);
__device__ float __fmaf_rz(float a, float b, float c);
__device__ float __fmaf_ru(float a, float b, float c);
__device__ float __fmaf_rd(float a, float b, float c);
__device__ float __sinf(float x);
__device__ float __cosf(float x);
__device__ float __expf(float x);
__device__ float __logf(float x);
__device__ float __fdividef(float a, float b);
__device__ float sinf(float x);
__device__ float expf(float x);
__device__ float logf(float x);
__device__ float fmaf(float a, float b, float c);
__device__ float sqrtf(float x);
__device__ float fabsf(float x);
__device__ double fma(double a, double b, double c);
__device__ double __dadd_rn(double a, double b);
__device__ double __dmul_rn(double a, double b);
__device__ double sqrt(double x);
__device__ int __float2int_rn(float x);
__device__ int __float2int_rz(float x);
__device__ int __float2int_ru(float x);
__device__ int __float2int_rd(float x);
__device__ float __int_as_float(int x);
__device__ int __float_as_int(float x);
__device__ unsigned __float_as_uint(float x);
__device__ float __uint_as_float(unsigned x);
__device__ long long __double_as_longlong(double x);
__device__ double __longlong_as_double(long long x);
