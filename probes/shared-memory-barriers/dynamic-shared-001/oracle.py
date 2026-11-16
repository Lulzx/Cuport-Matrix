# CUDA C++ Programming Guide, "__shared__": all extern __shared__ arrays start
# at the same address, so several arrays in one allocation are carved by
# offset. Doubling a float is exact, so fout is bit-exact.
from cuport_oracle import Buffer, Lcg

ID = "shared-memory-barriers/dynamic-shared-001"
BLOCKS, THREADS = 32, 192


def expected():
    n = BLOCKS * THREADS
    g = Lcg(ID)
    fin = [g.next_unit_f32() for _ in range(n)]
    uin = [g.next_u32() for _ in range(n)]
    fout, uout = [], []
    for b in range(BLOCKS):
        base = b * THREADS
        for t in range(THREADS):
            fout.append(fin[base + (t + 1) % THREADS] * 2.0)
            uout.append(uin[base + (t + 7) % THREADS])
    return {"fout": Buffer("f32", fout), "uout": Buffer("u32", uout)}
