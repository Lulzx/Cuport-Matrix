# fp-semantics/fma-contraction-001

Checks that `a*b + c`, written as plain CUDA C++, is contracted into a fused
multiply-add, as stock `nvcc` does by default (`--fmad=true`).

Each of 4096 elements is computed three ways in one kernel:

| Buffer | Expression | Roundings |
|---|---|---|
| `expr` | `a*b + c` | one, because the default contracts it |
| `fused` | `fmaf(a, b, c)` | one, always |
| `split` | `__fadd_rn(__fmul_rn(a, b), c)` | two; these intrinsics are never fused |

`a` and `b` are in `[1, 2)`. For even elements `c = -(a*b)` rounded to float,
so the fused result is the exact rounding error of the product and the split
result is zero; this makes the two cases differ in almost every even element.
Odd elements use `c` in `[-4, -2)`.

Tolerance is bit-exact. The oracle rounds the exact rational results once
(`fused`, `expr`) or twice (`split`) with `cuport_oracle.round_value`.

A tool that never contracts produces `split` values in `expr`; one that
contracts the intrinsics produces `fused` values in `split`. The first is a
difference in compiler defaults, the second breaks a documented guarantee.
Contraction is permitted rather than required by C++, which is why the
`--fmad` default, not the language, is the cited rule.
