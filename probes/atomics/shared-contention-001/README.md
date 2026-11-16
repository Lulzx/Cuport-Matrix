# atomics/shared-contention-001

All 512 threads of each block `atomicAdd` into one `__shared__` counter,
then into an 8-bin `__shared__` histogram, so each address sees 64-512
contending updates from 16 SIMD groups. 64 blocks run at once. Lost updates
under contention, or atomics that are only atomic within a SIMD group,
show up as low totals or bins that don't sum to 512.
