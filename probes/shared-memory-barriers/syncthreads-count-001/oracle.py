# CUDA C++ Programming Guide, "Synchronization Functions": __syncthreads_count
# returns the number of threads whose predicate is non-zero; __syncthreads_and
# is non-zero iff it is non-zero for all threads; __syncthreads_or iff for any.
from cuport_oracle import Buffer, lcg_u32

ID = "shared-memory-barriers/syncthreads-count-001"
BLOCKS, THREADS = 256, 128


def expected():
    data = lcg_u32(ID, BLOCKS * THREADS)
    count, all_, any_ = [], [], []
    for b in range(BLOCKS):
        block = data[b * THREADS:(b + 1) * THREADS]
        count.append(sum((v & 7) == 0 for v in block))
        all_.append(int(all((v & 0xFF) != 0 for v in block)))
        any_.append(int(any((v & 0xFFF) == 0 for v in block)))
    return {"count": Buffer("u32", count), "all": Buffer("u32", all_), "any": Buffer("u32", any_)}
