# atomics/cas-loop-001

2048 threads update 8 buckets through the standard `atomicCAS` retry loop.
Each bucket must end at the sum of its addends. The value each thread gets
back from its successful CAS is a different intermediate total, so within a
bucket all returned values must be distinct. The host counts the distinct
values, and the count must equal the number of threads in that bucket. A CAS
that reports success without storing, or that stores without being atomic,
repeats an intermediate value or loses an update.

One caveat: the "distinct" argument assumes no two different intermediate
totals collide mod 2^32. The addends are hashed (`v * 2654435761`), and with
about 256 updates per bucket a collision is astronomically unlikely, but it
is not impossible.
