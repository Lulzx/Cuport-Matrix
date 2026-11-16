# fp-semantics/fast-expf-001

Validates the CUDA `__expf(x)` intrinsic over the domain `|x| < 0.85`.

Cites: CUDA C++ Programming Guide, "Intrinsic Functions" table, which specifies
an error bound of `2 + floor(1.173 * |x|)` ulp for `__expf(x)`. When `|x| < 0.85`,
`1.173 * |x| < 1.0`, so `floor(1.173 * |x|) == 0` and the documented bound is exactly 2 ulp.

Inputs consist of edge cases 0.0 and -0.0, followed by 4096 LCG pseudorandom values
scaled to `[-0.84, 0.84]`. The cpu-reference oracle evaluates `math.exp(x)` rounded
to float32 with tolerance `ulp:2`.
