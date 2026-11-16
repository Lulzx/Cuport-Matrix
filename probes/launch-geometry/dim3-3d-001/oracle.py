# CUDA C++ Programming Guide, "Thread Hierarchy": for a block of size
# (Dx, Dy, Dz) the thread at (x, y, z) has index x + y*Dx + z*Dx*Dy.
from cuport_oracle import Buffer

GRID, BLOCK = (3, 2, 4), (8, 4, 2)


def expected():
    packed = []
    for bz in range(GRID[2]):
        for by in range(GRID[1]):
            for bx in range(GRID[0]):
                for tz in range(BLOCK[2]):
                    for ty in range(BLOCK[1]):
                        for tx in range(BLOCK[0]):
                            packed.append(bx | by << 4 | bz << 8 | tx << 12 | ty << 16 | tz << 20)
    return {"packed": Buffer("u32", packed), "dims": Buffer("u32", [*BLOCK, *GRID])}
