# device-services/malloc-free-001

256 threads each `malloc` between 16 and 256 bytes (size from the LCG),
fill the block with a per-thread byte pattern, fold it into a weighted sum,
and `free` it. The host sets `cudaLimitMallocHeapSize` to 8 MB first, which
is part of the feature under test.

`nulls` reports allocation failures without dereferencing; `sums` checks
that the allocated memory is real, private to the thread, and readable
after the writes. A tool with no device heap should refuse at compile or
run time; one that hands every thread the same scratch pointer produces
wrong sums under contention.
