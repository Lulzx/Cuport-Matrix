# warp-primitives/shfl-variants-001

Indexed `__shfl_sync` with a lane permutation (`(lane * 7 + 3) % 32`),
`__shfl_up_sync` by 5 and `__shfl_down_sync` by 3, over 64 full warps. The
boundary lanes (0-4 for up, 29-31 for down) must return their own value; a
translation that wraps around, or returns zero, fails at those lanes only.
