# atomics/atomic-64bit-001

1024 threads apply 64-bit `atomicAdd`, `atomicMax` and an `atomicCAS` XOR
loop to 4 buckets. Apple GPUs have no native 64-bit atomics, so tools
emulate them (mvcc's README documents hashed spinlocks). Torn updates show
up as carries lost between the 32-bit halves of `sum`, or as a `max` built
from halves of different values. The README also warns that a spin lock
contended within one warp can hang, which this probe reports as `crash-hang`
via its timeout.
