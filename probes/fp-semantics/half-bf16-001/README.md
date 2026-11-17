# fp-semantics/half-bf16-001

Checks `float` to `half` and `float` to `bfloat16` conversions in every rounding
mode CUDA exposes for them, and the exact widening back to `float`.

| Buffer | Conversion | Input |
|---|---|---|
| `h_rn`, `h_rz`, `h_ru`, `h_rd` | `__float2half_{rn,rz,ru,rd}(x)`, raw half bits | `x` |
| `h_back` | `__half2float(__float2half_rn(x))` | `x` |
| `bf_rn`, `bf_rz` | `__float2bfloat16_{rn,rz}(z)`, raw bf16 bits | `z` |
| `bf_back` | `__bfloat162float(__float2bfloat16_rn(z))` | `z` |

The first 24 elements of both inputs are edge cases: signed zeros,
infinities, 65504 (the largest half), the 65520 tie that rounds to infinity,
the smallest half subnormal and the tie below it, half and bf16 round-to-even
ties, the float maximum and the smallest float subnormals. The remaining 2024
are random: `x` spans biased float exponents 101 to 144, from below the
smallest half subnormal to above the largest half; `z` is any finite float.

Tolerance is bit-exact. The oracle rounds each exact input once with
`cuport_oracle.round_bits` in the requested mode, which follows IEEE 754:
overflow gives infinity, or the largest finite value when the mode rounds
toward zero.

The hosted `paravirtual` device has no bf16 (`matrix/schema.md`), so a tool
that maps `__nv_bfloat16` to Metal's `bfloat` may refuse there and pass on
real silicon; the two lanes are kept as separate rows for that reason.
