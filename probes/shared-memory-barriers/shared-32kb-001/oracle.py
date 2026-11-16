# CUDA C++ Programming Guide, "Compute Capabilities" / "Shared Memory": a block
# may use up to 48 KB of shared memory without cudaFuncSetAttribute opt-in, so
# both 32768 B and 32772 B of dynamic shared memory (no static shared) are valid.
from cuport_oracle import Buffer, lcg_u32

ID = "shared-memory-barriers/shared-32kb-001"
SIZES, BLOCKS = (8192, 8193), 4
M = 0xFFFFFFFF


def expected():
    data = lcg_u32(ID, 8193)
    folds = []
    for n in SIZES:
        for b in range(BLOCKS):
            key = (b * 0x9E3779B9) & M
            total = 0
            for i in range(n):
                total = (total + (data[n - 1 - i] ^ key) * (i | 1)) & M
            folds.append(total)
    return {"folds": Buffer("u32", folds)}
