# fp-semantics/rounding-modes-001

Checks CUDA's directed-rounding single-precision intrinsics: five operations,
each in four rounding modes, 20 buffers of 1024 elements.

| Operation | Intrinsics |
|---|---|
| add | `__fadd_rn`, `__fadd_rz`, `__fadd_ru`, `__fadd_rd` |
| mul | `__fmul_rn`, `__fmul_rz`, `__fmul_ru`, `__fmul_rd` |
| div | `__fdiv_rn`, `__fdiv_rz`, `__fdiv_ru`, `__fdiv_rd` |
| sqrt | `__fsqrt_rn`, `__fsqrt_rz`, `__fsqrt_ru`, `__fsqrt_rd`, of `fabsf(a)` |
| fma | `__fmaf_rn`, `__fmaf_rz`, `__fmaf_ru`, `__fmaf_rd` |

Buffers are named `<op>_<mode>`, e.g. `div_ru`. Inputs `a`, `b`, `c` have a
random sign, a biased exponent in `127 + [-10, 10]` and random mantissas, so
every result is a normal float. Most results are inexact, so `_ru` and `_rd`
differ in 890 of 1024 sums (the rest are exact), all 1024 products, quotients
and fused results, and 1023 of 1024 square roots.

Tolerance is bit-exact. The oracle computes each exact rational result (an
exact integer square-root comparison for `sqrt`) and rounds it once in the
requested direction with `cuport_oracle`.

A tool whose target has no directed-rounding arithmetic has to emulate these
modes in software or refuse them. A tool that maps every mode to
round-to-nearest passes the `_rn` buffers and fails the rest.
