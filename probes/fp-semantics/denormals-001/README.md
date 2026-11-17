# fp-semantics/denormals-001

Checks that subnormal (denormal) values survive single- and double-precision
arithmetic. Stock `nvcc` defaults to `--ftz=false`, so float subnormals are
flushed neither on input nor on output; double precision never flushes.

| Buffer | Operation | What a flush-to-zero tool returns |
|---|---|---|
| `half_tiny` | smallest normal floats (biased exponent 1 or 2) `* 0.5f` | ±0 where the result is subnormal |
| `sub_add` | subnormal `+` subnormal (exact) | ±0 |
| `sub_scale` | subnormal `* 2^24` (exact, normal result) | ±0 |
| `d_half` | smallest normal doubles `* 0.5` | ±0 where the result is subnormal |
| `d_scale` | subnormal double `* 2^54` | ±0 |

2048 elements per buffer, random signs, all inputs built from LCG bit
patterns. Every result is the exact value rounded once to nearest even, so the
tolerance is bit-exact. `half_tiny` and `d_half` include ties (odd significands
halved), which also exercises round-to-even in the subnormal range.

Metal may flush float subnormals, depending on how a kernel is compiled. A
flushed result is a divergence from `nvcc`'s default whether or not the tool
documents it; triage decides which kind.
