# warp-primitives/shfl-xor-width-001

All 25 combinations of butterfly mask (1, 2, 4, 8, 16) and `width` (32, 16,
8, 4, 2) for `__shfl_xor_sync`. When the mask is at least the width, the
partner lane is in another group: a partner in an earlier group is read
normally, a partner in a later group gives the caller its own value. Metal's
`simd_shuffle_xor` has no width parameter, so a tool has to emulate this
asymmetric rule. Ignoring `width` gives wrong values for exactly the
mask >= width cases. `baselines/metal/kernels/simd_shuffle_xor.metal` is the
width-32 case in native Metal.
