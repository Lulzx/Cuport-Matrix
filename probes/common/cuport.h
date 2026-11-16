// Shared host-side helpers for Cuport-Matrix probes. Plain C++17 plus the
// CUDA runtime API; no tool-specific extensions. Mirrors cuport_oracle.py.
//
// Output protocol (stdout), one line per emitted buffer:
//     CUPORT <name> <dtype> <count> <hex> <hex> ...
// dtype is u16, u32, i32, u64, f32 or f64; floats are emitted as raw bits.
// A probe ends with cuport_done(), which prints CUPORT-DONE. A run without
// that line is treated as a crash. CUDA errors print
//     CUPORT-ERROR <expression> <code> <message>
// and exit with status 3. Any other stdout line (device printf) is text.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef CUPORT_NO_CUDA
#include <cuda_runtime.h>
#endif

static inline uint64_t cuport_fnv1a64(const char *s) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (; *s; ++s) {
        h ^= (uint8_t)*s;
        h *= 0x100000001b3ull;
    }
    return h;
}

// x_{n+1} = 6364136223846793005 * x_n + 1442695040888963407, x_0 = FNV-1a(id).
struct CuportLcg {
    uint64_t x;
    explicit CuportLcg(const char *probe_id) : x(cuport_fnv1a64(probe_id)) {}
    uint32_t next_u32() {
        x = 6364136223846793005ull * x + 1442695040888963407ull;
        return (uint32_t)(x >> 32);
    }
    // Uniform in [0, 1) with 24 random bits; exact in float.
    float next_unit_f32() { return (float)(next_u32() >> 8) * (1.0f / 16777216.0f); }
};

static inline void cuport_fill_u32(const char *probe_id, uint32_t *dst, size_t n) {
    CuportLcg g(probe_id);
    for (size_t i = 0; i < n; ++i) dst[i] = g.next_u32();
}

static inline void cuport_emit_bits_(const char *name, const char *dtype, size_t n,
                                     const void *data, size_t elem) {
    printf("CUPORT %s %s %zu", name, dtype, n);
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < n; ++i) {
        uint64_t v = 0;
        memcpy(&v, p + i * elem, elem);  // little-endian host
        printf(" %0*llx", (int)(elem * 2), (unsigned long long)v);
    }
    printf("\n");
    fflush(stdout);
}

static inline void cuport_emit_u16(const char *n_, const uint16_t *v, size_t n) { cuport_emit_bits_(n_, "u16", n, v, 2); }
static inline void cuport_emit_u32(const char *n_, const uint32_t *v, size_t n) { cuport_emit_bits_(n_, "u32", n, v, 4); }
static inline void cuport_emit_i32(const char *n_, const int32_t *v, size_t n)  { cuport_emit_bits_(n_, "i32", n, v, 4); }
static inline void cuport_emit_u64(const char *n_, const uint64_t *v, size_t n) { cuport_emit_bits_(n_, "u64", n, v, 8); }
static inline void cuport_emit_f32(const char *n_, const float *v, size_t n)    { cuport_emit_bits_(n_, "f32", n, v, 4); }
static inline void cuport_emit_f64(const char *n_, const double *v, size_t n)   { cuport_emit_bits_(n_, "f64", n, v, 8); }

static inline int cuport_done(void) {
    printf("CUPORT-DONE\n");
    fflush(stdout);
    return 0;
}

#ifndef CUPORT_NO_CUDA
static inline void cuport_check(cudaError_t e, const char *expr) {
    if (e != cudaSuccess) {
        printf("CUPORT-ERROR %s %d %s\n", expr, (int)e, cudaGetErrorString(e));
        fflush(stdout);
        exit(3);
    }
}
#define CUPORT_CHECK(x) cuport_check((x), #x)
#endif
