# warp-primitives/ballot-any-all-001

`__ballot_sync`, `__any_sync` and `__all_sync` over 256 full warps. The
ballot/all predicate is true with probability 15/16 per lane, so roughly one
warp in eight is all-true; the `any` predicate is true with probability
1/1024 per lane, so only a few warps are any-true. A ballot with bits in the
wrong order, or votes taken over the wrong lanes, show up per warp.
`baselines/metal/kernels/simd_ballot.metal` is the native Metal counterpart.
