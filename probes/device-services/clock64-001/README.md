# device-services/clock64-001

Each of 128 threads reads `clock64()`, runs a 4096-step dependent LCG, reads
`clock64()` again and reports whether the second value is strictly larger.
The LCG result is emitted and checked so the work between the reads cannot
be optimised away.

The probe does not check how many cycles elapsed, only that the counter is
64-bit, moves, and moves forward. A tool that stubs `clock64()` as a
constant, truncates it to 32 bits in a way that wraps during the loop, or
maps it to a timer coarse enough to return the same value twice fails
`increasing` or `any_zero_delta`.
