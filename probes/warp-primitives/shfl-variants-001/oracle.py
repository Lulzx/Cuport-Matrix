# CUDA C++ Programming Guide, "Warp Shuffle Functions": __shfl_up_sync returns
# the caller's own value when laneId - delta < 0; __shfl_down_sync when
# laneId + delta >= width (32 here); __shfl_sync reads srcLane % width.
from cuport_oracle import Buffer, lcg_u32

ID = "warp-primitives/shfl-variants-001"
N = 16 * 128


def expected():
    v = lcg_u32(ID, N)
    idx, up, down = [], [], []
    for g in range(N):
        base, lane = g - g % 32, g % 32
        idx.append(v[base + (lane * 7 + 3) % 32])
        up.append(v[g] if lane < 5 else v[g - 5])
        down.append(v[g] if lane + 3 >= 32 else v[g + 3])
    return {"idx": Buffer("u32", idx), "up": Buffer("u32", up), "down": Buffer("u32", down)}
