# CUDA C++ Programming Guide, "Warp Vote Functions" / "Warp Shuffle Functions":
# with a member mask naming exactly the lanes that exist, __ballot_sync sets
# only those lanes' bits and __shfl_sync reads among them. A block of 48
# threads has a full first warp and a 16-lane second warp.
from cuport_oracle import Buffer, lcg_u32

ID = "warp-primitives/partial-warp-001"
BLOCKS, THREADS = 8, 48


def expected():
    v = lcg_u32(ID, BLOCKS * THREADS)
    mask_out, ballot, shfl = [], [], []
    for b in range(BLOCKS):
        for t in range(THREADS):
            g, warp, lane = b * THREADS + t, t // 32, t % 32
            members = min(32, THREADS - warp * 32)
            base = b * THREADS + warp * 32
            mask_out.append(0xFFFFFFFF if members == 32 else (1 << members) - 1)
            ballot.append(sum(1 << l for l in range(members) if v[base + l] & 1))
            shfl.append(v[base + (lane + 1) % members])
    return {"mask": Buffer("u32", mask_out), "ballot": Buffer("u32", ballot), "shfl": Buffer("u32", shfl)}
