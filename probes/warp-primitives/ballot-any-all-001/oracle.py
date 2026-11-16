# CUDA C++ Programming Guide, "Warp Vote Functions": __ballot_sync sets bit N
# iff lane N's predicate is non-zero; __any_sync / __all_sync are non-zero iff
# any / all participating lanes' predicates are non-zero.
from cuport_oracle import Buffer, lcg_u32

ID = "warp-primitives/ballot-any-all-001"
N = 64 * 128


def expected():
    v = lcg_u32(ID, N)
    ballot, any_, all_ = [], [], []
    for base in range(0, N, 32):
        warp = v[base:base + 32]
        ballot.append(sum(1 << l for l, x in enumerate(warp) if x & 0xF))
        any_.append(int(any((x & 0x3FF) == 0 for x in warp)))
        all_.append(int(all(x & 0xF for x in warp)))
    return {"ballot": Buffer("u32", ballot), "any": Buffer("u32", any_), "all": Buffer("u32", all_)}
