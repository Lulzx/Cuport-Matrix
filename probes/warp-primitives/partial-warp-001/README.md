# warp-primitives/partial-warp-001

Blocks of 48 threads, so every second warp has 16 lanes. Each warp computes
its member mask explicitly (`0xFFFFFFFF` or `0x0000FFFF`) and passes it to
`__ballot_sync` and to a rotate-by-one `__shfl_sync` within its members.
Bits for missing lanes must stay clear, and the 16-lane rotation must wrap at
lane 15 rather than read a phantom lane 16. The probe avoids `__activemask()`
on purpose, because CUDA does not guarantee convergence at any given point.
