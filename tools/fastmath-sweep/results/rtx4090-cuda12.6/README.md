# RTX 4090, CUDA 12.6 sweep

Run on 2026-11-16 on an RTX 4090 (sm_89), driver 595.99.02, nvcc 12.6 (V12.6.20). The sweep code is
`tools/fastmath-sweep/` as of commit `0b0af08`. The unary functions cover every 32-bit pattern. `__fdividef`
uses 2^30 random bit-pattern pairs drawn from numpy `PCG64(20261116)`.

```bash
/usr/local/cuda-12.6/bin/nvcc -std=c++17 -arch=sm_89 -o build/fastmath-sweep/sweep tools/fastmath-sweep/sweep.cu
for fn in __sinf __expf __fdividef sinf expf; do
  python3 tools/fastmath-sweep/sweep.py --bin build/fastmath-sweep/sweep --fn $fn --chunk 2^26 \
    --gpu "NVIDIA GeForce RTX 4090 (sm_89)" --driver "595.99.02 (nvidia-smi reports CUDA 13.2)" \
    --nvcc "$(/usr/local/cuda-12.6/bin/nvcc --version | grep release)" \
    --out tools/fastmath-sweep/results/rtx4090-cuda12.6/$fn.txt
done
```

No input inside a documented domain broke its bound:

| fn | in-domain inputs | worst in-domain error | violations |
|---|---|---|---|
| `__sinf` | 2,157,060,022 | 3.474486e-07 abs (bound 2^-21.41) | 0 |
| `__expf` | 2,237,668,968 | 64 ulp (bound 2 + floor(1.173·\|x\|)) | 0 |
| `__fdividef` | 790,620,651 | 2 ulp | 0 |
| `sinf` | 4,278,190,080 | 1 ulp | 0 |
| `expf` | 2,237,668,968 | 2 ulp | 0 |

Outside the documented domains, `__sinf` exceeds the bound on 2,093,410,226 inputs (|x| > π) and `__fdividef` on
4,873,158 pairs. CUDA documents no bound for those inputs, so the counts record hardware behaviour rather than
conformance failures. The out-of-domain inputs for `__expf`, `sinf` and `expf` are NaN or infinity, or give a
result that is not a finite normal float. None of them exceed the bound, and no NaN or infinity result mismatches.

numpy prints `RuntimeWarning: invalid value encountered in cast` during the `__expf` and `expf` runs. Widening
signalling-NaN inputs to float64 triggers it, and every file reports `nonfinite_mismatch: 0`.
