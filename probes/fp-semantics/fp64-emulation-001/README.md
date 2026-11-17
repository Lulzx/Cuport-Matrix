# fp-semantics/fp64-emulation-001

Checks that `double` in device code is real IEEE binary64. Apple GPUs have no
FP64 hardware, so a CUDA-on-Metal tool has three choices: emulate it, quietly
demote it to `float`, or refuse it. This probe tells the three apart.

Five correctly rounded operations over 2048 elements:

| Buffer | Operation | CUDA's stated error |
|---|---|---|
| `add` | `a + b` | IEEE round to nearest even |
| `mul` | `a * b` | IEEE round to nearest even |
| `div` | `a / b` | 0 ulp |
| `sqrt` | `sqrt(b)` | 0 ulp |
| `fma` | `fma(a, b, c)` | 0 ulp |

`a` has a random sign and `b` is positive; both have a biased exponent in
`1023 + [-8, 7]` and 52 random mantissa bits. For even elements
`c = -(a*b)` rounded, so `fma` returns the product's rounding error, which only
a genuinely fused 106-bit product gets right. Odd elements use a random `c`.

Tolerance is bit-exact: CUDA documents every one of these as correctly
rounded. A demoted-to-float tool fails every buffer; an emulation that is
close but not correctly rounded fails a few elements of `div`, `sqrt` or
`fma`. The oracle rounds exact rationals with `cuport_oracle`.

Apple GPUs have no FP64 hardware; CuMetal emulates it with VF64 in
`fast48`, `wide48` or `ieee64` mode (`docs/survey-2026-11-15.md`, section 5).
Only `ieee64` can be expected to pass bit-exact, so a CuMetal cell records the
active mode in its notes.
