# fp-semantics/fast-sinf-001

Validates the CUDA `__sinf(x)` intrinsic over the domain `[-pi, pi]`.

Cites: CUDA C++ Programming Guide, "Intrinsic Functions" table, which specifies
a maximum absolute error of 2^-21.41 for `__sinf(x)` when `x` is in `[-pi, pi]`.

Inputs consist of 4 edge cases (0.0, -0.0, pi_f, -pi_f) followed by 4096 LCG
pseudorandom values scaled into `[-pi_f, pi_f]`, where `pi_f` is single-precision
pi rounded down (`0x40490fda`, approximately 3.1415925f) to ensure all inputs stay
strictly inside `[-pi, pi]`. The cpu-reference oracle evaluates `math.sin(x)`
rounded to float32 with tolerance `abs:3.9e-7` (2^-21.41 documented bound plus
2^-25 for oracle rounding).
