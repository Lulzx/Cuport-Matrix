# shared-memory-barriers/static-shared-001

64 blocks each stage 256 input words in a static `__shared__ uint32_t[256]`,
synchronise, and write the tile back reversed. Thread t reads the word thread
255 - t wrote, which lives in a different SIMD group for most t, so a missing
barrier or per-warp (rather than per-block) shared storage gives wrong words.
