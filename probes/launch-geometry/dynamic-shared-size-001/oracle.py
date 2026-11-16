# CUDA C++ Programming Guide, "Execution Configuration": Ns is the number of
# bytes of shared memory dynamically allocated per block, used by extern
# __shared__ arrays. 16 KB is below the 48 KB available without opt-in.
from cuport_oracle import Buffer, lcg_u32

ID = "launch-geometry/dynamic-shared-size-001"
SIZES, BLOCKS = (64, 1024, 4096), 8


def expected():
    data = lcg_u32(ID, 4096)
    folds = []
    for n in SIZES:
        for b in range(BLOCKS):
            acc = 0
            for i in range(n):
                acc = (acc * 0x01000193 + (data[n - 1 - i] ^ b)) & 0xFFFFFFFF
            folds.append(acc)
    return {"folds": Buffer("u32", folds)}
