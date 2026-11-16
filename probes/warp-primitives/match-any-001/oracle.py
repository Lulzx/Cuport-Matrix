# CUDA C++ Programming Guide, "Warp Match Functions": __match_any_sync returns
# the mask of threads in `mask` that have the same value of `value`.
from cuport_oracle import Buffer, lcg_u32

ID = "warp-primitives/match-any-001"
N = 16 * 128


def expected():
    keys = [x % 5 for x in lcg_u32(ID, N)]
    masks = []
    for g in range(N):
        base = g - g % 32
        masks.append(sum(1 << l for l in range(32) if keys[base + l] == keys[g]))
    return {"masks": Buffer("u32", masks)}
