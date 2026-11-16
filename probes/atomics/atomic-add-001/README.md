# atomics/atomic-add-001

4096 threads `atomicAdd` into 16 `int` buckets (signed addends that wrap)
and 16 `float` buckets. The float addends are integers 0-255 and no bucket
reaches 2^24, so the float totals are exact whatever order the hardware
serialises the adds in. A lost update or a non-atomic read-modify-write
shows up as a wrong total in the affected bucket.
