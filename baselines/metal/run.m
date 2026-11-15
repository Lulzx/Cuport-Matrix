// Native Metal baseline runner for Cuport-Matrix.
//
// Compiles each kernel in kernels/ at runtime with the Metal framework (no
// Xcode or offline `metal` compiler needed), feeds it the deterministic LCG
// stream from matrix/schema.md, and compares the output bit-for-bit with a
// CPU reference that encodes the CUDA semantics (32-wide warps).
//
// Usage: run [--selftest] [--kernels <dir>]
// Exit status is 0 only if every baseline is bit-exact.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// ---- Deterministic inputs (matrix/schema.md) -------------------------------

static uint64_t fnv1a64(const char *s) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (; *s; ++s) {
        h ^= (uint8_t)*s;
        h *= 0x100000001b3ull;
    }
    return h;
}

// x_{n+1} = 6364136223846793005 * x_n + 1442695040888963407, x_0 = seed.
// Each input word is the high 32 bits of x_1, x_2, ...
static void lcg_fill(uint64_t seed, uint32_t *dst, size_t n) {
    uint64_t x = seed;
    for (size_t i = 0; i < n; ++i) {
        x = 6364136223846793005ull * x + 1442695040888963407ull;
        dst[i] = (uint32_t)(x >> 32);
    }
}

// ---- Baseline table ---------------------------------------------------------

#define CUDA_WARP 32u
#define MAX_OUTS 2

typedef void (*reference_fn)(const uint32_t *in, size_t n, uint32_t **expect);

typedef struct {
    const char *id;          // also the FNV-1a seed string
    const char *feature;     // matrix/features.yaml id this baseline mirrors
    const char *file;        // kernels/<file>
    const char *function;
    size_t threads;
    size_t threadgroup;
    size_t out_words[MAX_OUTS];  // 0 terminates
    reference_fn reference;
} baseline;

static void ref_shuffle_xor(const uint32_t *in, size_t n, uint32_t **expect) {
    for (size_t g = 0; g < n; ++g) {
        size_t base = g - g % CUDA_WARP;
        size_t lane = g % CUDA_WARP;
        for (unsigned k = 0; k < 5; ++k) {
            expect[0][g * 5 + k] = in[base + (lane ^ (1u << k))];
        }
    }
    for (size_t w = 0; w < n / CUDA_WARP; ++w) {
        expect[1][w] = CUDA_WARP;
    }
}

static void ref_ballot(const uint32_t *in, size_t n, uint32_t **expect) {
    for (size_t base = 0; base < n; base += CUDA_WARP) {
        uint32_t mask = 0;
        for (unsigned l = 0; l < CUDA_WARP; ++l) {
            if (in[base + l] & 0xFu) mask |= 1u << l;
        }
        for (unsigned l = 0; l < CUDA_WARP; ++l) {
            uint32_t *o = &expect[0][(base + l) * 3];
            o[0] = mask;
            o[1] = mask != 0;
            o[2] = mask == 0xFFFFFFFFu;
        }
    }
}

static void ref_reduce(const uint32_t *in, size_t n, uint32_t **expect) {
    for (size_t tg = 0; tg < n / 256; ++tg) {
        uint32_t sum = 0;
        for (size_t i = 0; i < 256; ++i) sum += in[tg * 256 + i];
        expect[0][tg] = sum;
    }
}

static const baseline BASELINES[] = {
    {"metal/simd-shuffle-xor-001", "warp-primitives.shfl_xor_width",
     "simd_shuffle_xor.metal", "simd_shuffle_xor_baseline",
     4096, 256, {4096 * 5, 4096 / CUDA_WARP}, ref_shuffle_xor},
    {"metal/simd-ballot-001", "warp-primitives.ballot_any_all",
     "simd_ballot.metal", "simd_ballot_baseline",
     4096, 256, {4096 * 3, 0}, ref_ballot},
    {"metal/threadgroup-reduce-001", "shared-memory-barriers.syncthreads_reduction",
     "threadgroup_reduce.metal", "threadgroup_reduce_baseline",
     65536, 256, {65536 / 256, 0}, ref_reduce},
};

// ---- Self-test --------------------------------------------------------------

static int selftest(void) {
    struct { const char *s; uint64_t h; } v[] = {
        {"", 0xcbf29ce484222325ull},
        {"a", 0xaf63dc4c8601ec8cull},
        {"foobar", 0x85944171f73967e8ull},
    };
    int bad = 0;
    for (size_t i = 0; i < sizeof v / sizeof v[0]; ++i) {
        uint64_t got = fnv1a64(v[i].s);
        if (got != v[i].h) {
            printf("FAIL fnv1a64(\"%s\") = %016llx, want %016llx\n", v[i].s,
                   (unsigned long long)got, (unsigned long long)v[i].h);
            bad = 1;
        }
    }
    uint32_t w[2];
    lcg_fill(0, w, 2);
    // From x_0 = 0: x_1 is the increment, 0x14057B7EF767814F.
    uint64_t x1 = 1442695040888963407ull;
    uint64_t x2 = 6364136223846793005ull * x1 + 1442695040888963407ull;
    if (w[0] != 0x14057B7Eu || w[1] != (uint32_t)(x2 >> 32)) {
        printf("FAIL lcg_fill(0) = %08x %08x\n", w[0], w[1]);
        bad = 1;
    }
    printf("selftest: %s\n", bad ? "FAIL" : "ok");
    return bad;
}

// ---- Runner -----------------------------------------------------------------

static int run_baseline(id<MTLDevice> dev, id<MTLCommandQueue> q,
                        NSString *dir, const baseline *b) {
    NSString *path = [dir stringByAppendingPathComponent:@(b->file)];
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:path
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (!src) {
        printf("%-30s crash-hang  cannot read %s\n", b->id, path.UTF8String);
        return 1;
    }
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        printf("%-30s refused     compile error: %s\n", b->id,
               err.localizedDescription.UTF8String);
        return 1;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@(b->function)];
    id<MTLComputePipelineState> pso =
        fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
    if (!pso) {
        printf("%-30s refused     pipeline error: %s\n", b->id,
               err ? err.localizedDescription.UTF8String : "function not found");
        return 1;
    }

    uint64_t seed = fnv1a64(b->id);
    size_t n = b->threads;
    id<MTLBuffer> inbuf = [dev newBufferWithLength:n * 4
                                           options:MTLResourceStorageModeShared];
    lcg_fill(seed, (uint32_t *)inbuf.contents, n);

    id<MTLBuffer> outs[MAX_OUTS] = {nil};
    uint32_t *expect[MAX_OUTS] = {NULL};
    size_t nouts = 0;
    for (; nouts < MAX_OUTS && b->out_words[nouts]; ++nouts) {
        size_t bytes = b->out_words[nouts] * 4;
        outs[nouts] = [dev newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared];
        memset(outs[nouts].contents, 0xAB, bytes);  // poison unwritten words
        expect[nouts] = calloc(b->out_words[nouts], 4);
    }

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:inbuf offset:0 atIndex:0];
    for (size_t i = 0; i < nouts; ++i) [enc setBuffer:outs[i] offset:0 atIndex:1 + i];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(b->threadgroup, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        printf("%-30s crash-hang  %s\n", b->id, cb.error.localizedDescription.UTF8String);
        return 1;
    }

    b->reference((const uint32_t *)inbuf.contents, n, expect);

    size_t total = 0, mismatched = 0;
    for (size_t i = 0; i < nouts; ++i) {
        const uint32_t *got = (const uint32_t *)outs[i].contents;
        for (size_t j = 0; j < b->out_words[i]; ++j) {
            ++total;
            if (got[j] != expect[i][j]) {
                if (mismatched < 3) {
                    printf("    out%zu[%zu] = %08x, want %08x\n", i, j, got[j], expect[i][j]);
                }
                ++mismatched;
            }
        }
        free(expect[i]);
    }
    printf("%-30s %-11s %zu/%zu words  feature=%s seed=%016llx threads=%zu tg=%zu simd_width=%lu\n",
           b->id, mismatched ? "silently-wrong" : "bit-exact", total - mismatched, total,
           b->feature, (unsigned long long)seed, n, b->threadgroup,
           (unsigned long)pso.threadExecutionWidth);
    return mismatched != 0;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        NSString *dir = @"kernels";
        for (int i = 1; i < argc; ++i) {
            if (!strcmp(argv[i], "--selftest")) return selftest();
            if (!strcmp(argv[i], "--kernels") && i + 1 < argc) dir = @(argv[++i]);
        }
        if (selftest()) return 1;

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            printf("no Metal device\n");
            return 1;
        }
        printf("device: %s\nmacos: %s\n", dev.name.UTF8String,
               NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
        id<MTLCommandQueue> q = [dev newCommandQueue];

        int failed = 0;
        size_t count = sizeof BASELINES / sizeof BASELINES[0];
        for (size_t i = 0; i < count; ++i) failed += run_baseline(dev, q, dir, &BASELINES[i]);
        printf("%zu/%zu baselines bit-exact\n", count - failed, count);
        return failed != 0;
    }
}
