# fp-semantics/fast-fdividef-001

Validates the CUDA `__fdividef(x, y)` intrinsic over the domain `|y| in [2^-126, 2^126]`
with quotients restricted to finite normal floating-point numbers.

Cites: CUDA C++ Programming Guide, "Intrinsic Functions" table, which specifies
that `__fdividef(x, y)` has a 2 ulp maximum error for `|y| in [2^-126, 2^126]`
provided the quotient is a finite normal number. Inputs near the `2^126` boundary
and overflowing quotients are excluded from the probe domain per specification.

Inputs consist of 4096 `(x, y)` pairs generated from LCG pseudorandom draws with
exponents constrained to `[70, 184]`, ensuring `|y|` is in `[2^-57, 2^58]` and
every quotient `x / y` is strictly a finite normal float32. The cpu-reference
oracle evaluates IEEE single-precision division with tolerance `ulp:2`.
