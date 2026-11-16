# tools/fastmath-sweep

Full-range intrinsic error sweep tool for CUDA fast-math functions against documented bounds.

Cites: CUDA C++ Programming Guide, "Intrinsic Functions" table.

## Architecture

The tool consists of:
- `sweep.cu`: A compiled CUDA program invoked as `sweep <fn> <start> <count>`.
  Supported functions: `__sinf`, `__expf`, `__fdividef`, `sinf`, `expf`.
  For unary functions, inputs are consecutive 32-bit patterns starting from `start`
  through `start + count - 1`. For `__fdividef`, `count` `(x, y)` 32-bit float pairs
  are streamed from `stdin`. Outputs are emitted as raw little-endian float32 bytes
  to `stdout`.
- `sweep.py`: Driver script orchestrating chunked execution and metric accumulation.
  Computes reference values via numpy (float64 for transcendentals rounded to float32,
  and native float32 division for `__fdividef`).

## Documented Bounds

- `__sinf`: absolute error <= 2^-21.41 for `x in [-pi, pi]` (using float pi rounded down, `0x40490fda`).
- `__expf`: error <= `2 + floor(1.173 * |x|)` ULP when reference is a finite normal float.
- `__fdividef`: error <= 2 ULP for `|y| in [2^-126, 2^126]` where quotient is a finite normal float.
- `sinf`: error <= 2 ULP full range for finite inputs.
- `expf`: error <= 2 ULP full range when reference is a finite normal float.

Inputs matching these conditions are classified as in-domain; inputs outside are
evaluated and recorded separately as out-of-domain.

## Metrics Recorded

For both in-domain and out-of-domain partitions:
- `count`: Total inputs processed.
- `max_ulp`: Maximum ULP distance observed.
- `max_abs`: Maximum absolute error observed.
- `violations`: Count of inputs exceeding documented bounds.
- `histogram`: ULP distance frequencies partitioned across bins `0`, `1`, `2`, `3`, `4-7`, `8-15`, and `16+`.
- `nonfinite_mismatch`: Count of NaN vs non-NaN or infinity mismatches.

## Usage

Build the CUDA driver:
```bash
/usr/local/cuda-12.6/bin/nvcc -std=c++17 -arch=sm_89 -o /tmp/cuport-agy-fm/sweep tools/fastmath-sweep/sweep.cu
```

Run a sweep via the Python driver:
```bash
python3 tools/fastmath-sweep/sweep.py --fn __sinf --chunk 2^26 --limit 1048576 --out summary_sinf.txt
```
