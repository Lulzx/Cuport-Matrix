# shared-memory-barriers/shared-32kb-001

Two launches that fill all of their dynamic shared memory: 8192 words
(32 KB, the threadgroup limit on Apple GPUs and the paravirtual device) and
8193 words (32 KB + 4 B). The kernel has no static shared memory, so these are the
block's entire shared allocation. CUDA accepts
both without opt-in (limit 48 KB), so the oracle expects correct folds for
both. On Apple hardware the second launch has to be refused with a
diagnostic or served from a fallback (mvcc's README documents a
device-memory pool for launches over 32 KB). A wrong fold here means shared
memory was silently truncated or aliased.
