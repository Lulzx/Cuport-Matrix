"""Fast-math full-range sweep driver and metric evaluation.

Cites: CUDA C++ Programming Guide, "Intrinsic Functions" table.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time
import numpy as np

PI_F = np.uint32(0x40490FDA).view(np.float32)  # 3.1415925f, pi rounded down
HISTOGRAM_BINS = ["0", "1", "2", "3", "4-7", "8-15", "16+"]


def ulp_distance_f32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute ULP distance between float32 values.

    +0.0 and -0.0 have distance 0.
    Smallest positive subnormal vs smallest negative subnormal is 2.
    NaN vs NaN is 0.
    """
    u_a = a.view(np.uint32)
    u_b = b.view(np.uint32)

    mag_a = (u_a & np.uint32(0x7FFFFFFF)).astype(np.int64)
    mag_b = (u_b & np.uint32(0x7FFFFFFF)).astype(np.int64)

    ord_a = np.where((u_a & np.uint32(0x80000000)) != 0, -mag_a, mag_a)
    ord_b = np.where((u_b & np.uint32(0x80000000)) != 0, -mag_b, mag_b)

    dist = np.abs(ord_a - ord_b)
    both_nan = np.isnan(a) & np.isnan(b)
    return np.where(both_nan, np.int64(0), dist)


def in_domain(fn: str, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
    """Determine which inputs are within the documented bound domain.

    Documented bounds:
    - __sinf: abs 2^-21.41 for x in [-pi, pi] (float pi rounded down).
    - __expf: 2 + floor(1.173*|x|) ulp, where reference is finite normal.
    - __fdividef: 2 ulp for |y| in [2^-126, 2^126] and quotient finite normal.
    - sinf: 2 ulp full range (finite input).
    - expf: 2 ulp full range (reference is finite normal).
    """
    if fn == "__sinf":
        return (x >= -PI_F) & (x <= PI_F)
    elif fn in ("__expf", "expf"):
        # Reference is finite normal
        ref = np.exp(x.astype(np.float64))
        abs_ref = np.abs(ref)
        return np.isfinite(ref) & (abs_ref >= 2.0 ** -126) & (abs_ref < 2.0 ** 128)
    elif fn == "__fdividef":
        if y is None:
            raise ValueError("__fdividef requires y")
        abs_y = np.abs(y.astype(np.float64))
        y_in = (abs_y >= 2.0 ** -126) & (abs_y <= 2.0 ** 126)
        q = x.astype(np.float64) / y.astype(np.float64)
        abs_q = np.abs(q)
        q_normal = np.isfinite(q) & (abs_q >= 2.0 ** -126) & (abs_q < 2.0 ** 128)
        return y_in & q_normal
    elif fn == "sinf":
        return np.isfinite(x)
    else:
        raise ValueError(f"unknown function {fn!r}")


def bound_ok(
    fn: str,
    x: np.ndarray,
    got: np.ndarray,
    want: np.ndarray,
    y: np.ndarray = None,
) -> np.ndarray:
    """Check if got matches want within the documented bound for fn."""
    nan_got = np.isnan(got)
    nan_want = np.isnan(want)
    inf_got = np.isinf(got)
    inf_want = np.isinf(want)

    both_nan = nan_got & nan_want
    both_inf = inf_got & inf_want & (got == want)
    nonfinite_match = both_nan | both_inf
    nonfinite_mismatch = (nan_got ^ nan_want) | (inf_got ^ inf_want) | (inf_got & inf_want & (got != want))

    if fn == "__sinf":
        abs_err = np.abs(got.astype(np.float64) - want.astype(np.float64))
        bound = 2.0 ** -21.41
        ok = abs_err <= bound
    elif fn == "__expf":
        ulps = ulp_distance_f32(got, want)
        bound = 2 + np.floor(1.173 * np.abs(x.astype(np.float64)))
        ok = ulps <= bound
    elif fn in ("__fdividef", "sinf", "expf"):
        ulps = ulp_distance_f32(got, want)
        ok = ulps <= 2
    else:
        raise ValueError(f"unknown function {fn!r}")

    return np.where(nonfinite_mismatch, False, np.where(nonfinite_match, True, ok))


def make_stats() -> dict:
    return {
        "in_domain": {
            "count": 0,
            "max_ulp": 0,
            "max_abs": 0.0,
            "violations": 0,
            "histogram": [0, 0, 0, 0, 0, 0, 0],
            "nonfinite_mismatch": 0,
        },
        "out_domain": {
            "count": 0,
            "max_ulp": 0,
            "max_abs": 0.0,
            "violations": 0,
            "histogram": [0, 0, 0, 0, 0, 0, 0],
            "nonfinite_mismatch": 0,
        },
    }


def accumulate(stats: dict, *args, **kwargs) -> None:
    """Accumulate comparison metrics into stats.

    Signature can be:
      accumulate(stats, got, want, in_domain_mask, ...) or
      accumulate(stats, fn, got, want, in_domain_mask, ...)
    """
    if len(args) >= 4 and isinstance(args[0], str):
        fn = args[0]
        got, want, in_dom = args[1], args[2], args[3]
        x = kwargs.get("x", None)
        y = kwargs.get("y", None)
    elif len(args) >= 3:
        got, want, in_dom = args[0], args[1], args[2]
        fn = kwargs.get("fn", None)
        x = kwargs.get("x", None)
        y = kwargs.get("y", None)
    else:
        raise ValueError("insufficient arguments to accumulate")

    nan_got = np.isnan(got)
    nan_want = np.isnan(want)
    inf_got = np.isinf(got)
    inf_want = np.isinf(want)

    mismatch = (nan_got ^ nan_want) | (inf_got ^ inf_want) | (inf_got & inf_want & (got != want))
    ulps = ulp_distance_f32(got, want)
    with np.errstate(invalid="ignore"):
        abs_err = np.abs(got.astype(np.float64) - want.astype(np.float64))
    abs_err = np.where(np.isnan(abs_err), 0.0, abs_err)

    if fn is not None and x is not None:
        ok = bound_ok(fn, x, got, want, y=y)
        violation = ~ok
    elif fn == "__sinf":
        violation = mismatch | (abs_err > 2.0 ** -21.41)
    elif fn is not None:
        # ulp bound of 2 default if x not given
        violation = mismatch | (ulps > 2)
    else:
        violation = mismatch | (ulps > 2)

    for domain_name, mask in [("in_domain", in_dom), ("out_domain", ~in_dom)]:
        part_count = int(np.count_nonzero(mask))
        if part_count == 0:
            continue
        d = stats[domain_name]
        d["count"] += part_count

        part_mismatch = int(np.count_nonzero(mask & mismatch))
        d["nonfinite_mismatch"] += part_mismatch

        part_violations = int(np.count_nonzero(mask & violation))
        d["violations"] += part_violations

        # For max metrics and histogram, exclude mismatches
        valid = mask & ~mismatch
        if np.any(valid):
            part_max_ulp = int(np.max(ulps[valid]))
            if part_max_ulp > d["max_ulp"]:
                d["max_ulp"] = part_max_ulp

            part_max_abs = float(np.max(abs_err[valid]))
            if part_max_abs > d["max_abs"]:
                d["max_abs"] = part_max_abs

            v_ulps = ulps[valid]
            h = d["histogram"]
            h[0] += int(np.count_nonzero(v_ulps == 0))
            h[1] += int(np.count_nonzero(v_ulps == 1))
            h[2] += int(np.count_nonzero(v_ulps == 2))
            h[3] += int(np.count_nonzero(v_ulps == 3))
            h[4] += int(np.count_nonzero((v_ulps >= 4) & (v_ulps <= 7)))
            h[5] += int(np.count_nonzero((v_ulps >= 8) & (v_ulps <= 15)))
            h[6] += int(np.count_nonzero(v_ulps >= 16))


def parse_size(s: str) -> int:
    """Parse size string, allowing expressions like 2^26 or integer."""
    s = str(s).strip()
    if "^" in s:
        base, exp = s.split("^", 1)
        return int(base) ** int(exp)
    return int(s)


def run_sweep(
    fn: str,
    chunk_size: int = 1 << 26,
    limit: int | None = None,
    seed: int = 20261116,
    bin_path: str = "/tmp/cuport-agy-fm/sweep",
    out_path: str | None = None,
    nvcc_info: str = "",
    driver_info: str = "",
    gpu_info: str = "",
) -> dict:
    is_fdivide = fn == "__fdividef"
    total_total = (1 << 30) if is_fdivide else (1 << 32)
    total = total_total if limit is None else min(limit, total_total)

    stats = make_stats()
    start_time = time.monotonic()

    rng = np.random.Generator(np.random.PCG64(seed)) if is_fdivide else None

    # Wrapper invocation
    runbin = "/home/ccbd/cuport-nv/runbin"

    done = 0
    while done < total:
        n = min(chunk_size, total - done)
        if not is_fdivide:
            # Unary fn: call sweep <fn> <start> <count>
            cmd = [runbin, bin_path, fn, str(done), str(n)]
            proc = subprocess.run(cmd, capture_output=True, check=True)
            got = np.frombuffer(proc.stdout, dtype="<f4")

            bits = np.arange(done, done + n, dtype=np.uint32)
            x = bits.view(np.float32)

            if fn in ("__sinf", "sinf"):
                want = np.sin(x.astype(np.float64)).astype(np.float32)
            else:
                want = np.exp(x.astype(np.float64)).astype(np.float32)

            in_dom = in_domain(fn, x)
            accumulate(stats, fn, got, want, in_dom, x=x)
        else:
            # Binary fn __fdividef: draw random 32-bit pairs and stream over stdin
            u = rng.integers(0, 2**32, size=2 * n, dtype=np.uint32)
            pairs = u.view(np.float32)

            cmd = [runbin, bin_path, fn, "0", str(n)]
            proc = subprocess.run(cmd, input=pairs.tobytes(), capture_output=True, check=True)
            got = np.frombuffer(proc.stdout, dtype="<f4")

            x = pairs[0::2]
            y = pairs[1::2]
            want = x.astype(np.float32) / y.astype(np.float32)

            in_dom = in_domain(fn, x, y=y)
            accumulate(stats, fn, got, want, in_dom, x=x, y=y)

        done += n

    wall_time = time.monotonic() - start_time

    summary = {
        "fn": fn,
        "input_range": f"0..{total - 1}",
        "total_inputs": total,
        "chunk_size": chunk_size,
        "seed": seed if is_fdivide else None,
        "nvcc": nvcc_info,
        "driver": driver_info,
        "gpu": gpu_info,
        "wall_time_s": wall_time,
        "stats": stats,
    }

    if out_path:
        p = pathlib.Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"fn: {fn}",
            f"input_range: 0..{total - 1}",
            f"total_inputs: {total}",
            f"chunk_size: {chunk_size}",
            f"seed: {seed if is_fdivide else 'n/a'}",
            f"nvcc: {nvcc_info}",
            f"driver: {driver_info}",
            f"gpu: {gpu_info}",
            f"wall_time_s: {wall_time:.3f}",
            "",
            "in_domain:",
            f"  count: {stats['in_domain']['count']}",
            f"  max_ulp: {stats['in_domain']['max_ulp']}",
            f"  max_abs: {stats['in_domain']['max_abs']:.6e}",
            f"  violations: {stats['in_domain']['violations']}",
            f"  histogram: {stats['in_domain']['histogram']}",
            f"  nonfinite_mismatch: {stats['in_domain']['nonfinite_mismatch']}",
            "",
            "out_domain:",
            f"  count: {stats['out_domain']['count']}",
            f"  max_ulp: {stats['out_domain']['max_ulp']}",
            f"  max_abs: {stats['out_domain']['max_abs']:.6e}",
            f"  violations: {stats['out_domain']['violations']}",
            f"  histogram: {stats['out_domain']['histogram']}",
            f"  nonfinite_mismatch: {stats['out_domain']['nonfinite_mismatch']}",
        ]
        p.write_text("\n".join(lines) + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Full-range fastmath sweep tool")
    parser.add_argument("--fn", required=True, choices=["__sinf", "__expf", "__fdividef", "sinf", "expf"])
    parser.add_argument("--chunk", default="2^26", help="Chunk size (e.g. 2^26)")
    parser.add_argument("--limit", default=None, help="Limit total inputs tested")
    parser.add_argument("--seed", type=int, default=20261116, help="RNG seed for __fdividef")
    parser.add_argument("--out", default=None, help="Output summary text file")
    parser.add_argument("--bin", default="/tmp/cuport-agy-fm/sweep", help="Path to sweep binary")
    parser.add_argument("--nvcc", default="", help="nvcc version string")
    parser.add_argument("--driver", default="", help="driver version string")
    parser.add_argument("--gpu", default="", help="GPU model string")

    args = parser.parse_args()
    chunk = parse_size(args.chunk)
    limit = parse_size(args.limit) if args.limit else None

    summary = run_sweep(
        fn=args.fn,
        chunk_size=chunk,
        limit=limit,
        seed=args.seed,
        bin_path=args.bin,
        out_path=args.out,
        nvcc_info=args.nvcc,
        driver_info=args.driver,
        gpu_info=args.gpu,
    )
    print(f"Sweep complete for {args.fn}: {summary['total_inputs']} inputs in {summary['wall_time_s']:.2f}s")


if __name__ == "__main__":
    main()
