"""Tests for fastmath sweep metric functions."""

import pathlib
import sys
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "fastmath-sweep"))

import sweep  # noqa: E402


def test_ulp_distance_zero():
    # ulp distance across +0/-0 is 0
    p0 = np.array([0.0], dtype=np.float32)
    m0 = np.array([-0.0], dtype=np.float32)
    assert sweep.ulp_distance_f32(p0, m0)[0] == 0
    assert sweep.ulp_distance_f32(p0, p0)[0] == 0


def test_ulp_distance_adjacent():
    # adjacent floats distance is 1
    a = np.array([1.0], dtype=np.float32)
    b = np.nextafter(a, np.float32(2.0))
    assert sweep.ulp_distance_f32(a, b)[0] == 1
    assert sweep.ulp_distance_f32(b, a)[0] == 1


def test_ulp_distance_across_zero():
    # across zero (smallest +denormal vs smallest -denormal) is 2
    pos_denorm = np.array([1], dtype=np.uint32).view(np.float32)
    neg_denorm = np.array([0x80000001], dtype=np.uint32).view(np.float32)
    assert sweep.ulp_distance_f32(pos_denorm, neg_denorm)[0] == 2
    assert sweep.ulp_distance_f32(pos_denorm, np.array([0.0], dtype=np.float32))[0] == 1
    assert sweep.ulp_distance_f32(neg_denorm, np.array([-0.0], dtype=np.float32))[0] == 1


def test_ulp_distance_nan():
    # NaN/NaN handled as 0
    nan1 = np.array([np.nan], dtype=np.float32)
    nan2 = np.array([np.nan], dtype=np.float32)
    assert sweep.ulp_distance_f32(nan1, nan2)[0] == 0


def test_bound_ok_expf_086():
    # __expf bound at x=0.86 is 3 ulp: 2 + floor(1.173 * 0.86) = 2 + 1 = 3 ulp
    x = np.array([0.86], dtype=np.float32)
    want = np.exp(x.astype(np.float64)).astype(np.float32)

    # 3 ulps away should be ok
    # Step 3 ulps from want
    got3 = want.copy()
    for _ in range(3):
        got3 = np.nextafter(got3, np.float32(np.inf))
    assert sweep.bound_ok("__expf", x, got3, want)[0]

    # 4 ulps away should violate the bound
    got4 = np.nextafter(got3, np.float32(np.inf))
    assert not sweep.bound_ok("__expf", x, got4, want)[0]


def test_bound_ok_sinf():
    # __sinf bound is abs 2^-21.41
    x = np.array([1.0], dtype=np.float32)
    want = np.sin(x.astype(np.float64)).astype(np.float32)
    bound = 2.0 ** -21.41
    got_ok = (want.astype(np.float64) + bound * 0.9).astype(np.float32)
    got_bad = (want.astype(np.float64) + bound * 1.5).astype(np.float32)
    assert sweep.bound_ok("__sinf", x, got_ok, want)[0]
    assert not sweep.bound_ok("__sinf", x, got_bad, want)[0]


def test_bound_ok_fdividef():
    x = np.array([5.0], dtype=np.float32)
    y = np.array([2.0], dtype=np.float32)
    want = (x / y).astype(np.float32)
    got2 = want.copy()
    for _ in range(2):
        got2 = np.nextafter(got2, np.float32(np.inf))
    got3 = np.nextafter(got2, np.float32(np.inf))
    assert sweep.bound_ok("__fdividef", x, got2, want, y=y)[0]
    assert not sweep.bound_ok("__fdividef", x, got3, want, y=y)[0]


def test_in_domain():
    # __sinf: [-pi_f, pi_f]
    pi_f = np.uint32(0x40490FDA).view(np.float32)
    assert sweep.in_domain("__sinf", np.array([0.0, float(pi_f), -float(pi_f)], dtype=np.float32)).all()
    assert not sweep.in_domain("__sinf", np.array([3.15, -3.15], dtype=np.float32)).any()

    # __expf: normal finite output
    assert sweep.in_domain("__expf", np.array([0.0, 1.0, -10.0], dtype=np.float32)).all()
    # overflows or underflows to non-normal
    assert not sweep.in_domain("__expf", np.array([100.0, -150.0], dtype=np.float32)).any()

    # __fdividef: |y| in [2^-126, 2^126] and quotient finite normal
    x_ok = np.array([1.0], dtype=np.float32)
    y_ok = np.array([2.0], dtype=np.float32)
    assert sweep.in_domain("__fdividef", x_ok, y=y_ok)[0]

    # y out of range
    y_tiny = np.array([1e-40], dtype=np.float32)  # subnormal (< 2^-126)
    assert not sweep.in_domain("__fdividef", x_ok, y=y_tiny)[0]


def test_accumulate_histogram_bins():
    stats = sweep.make_stats()
    # Bins: 0, 1, 2, 3, 4-7, 8-15, 16+
    ulps = [0, 1, 2, 3, 4, 7, 8, 15, 16, 25]
    want_arr = np.array([1.0] * len(ulps), dtype=np.float32)
    got_arr = []
    for u in ulps:
        val = np.float32(1.0)
        for _ in range(u):
            val = np.nextafter(val, np.float32(np.inf))
        got_arr.append(val)
    got_arr = np.array(got_arr, dtype=np.float32)
    in_domain = np.ones(len(ulps), dtype=bool)

    sweep.accumulate(stats, "expf", got_arr, want_arr, in_domain)
    # expected counts per bin:
    # 0: 1
    # 1: 1
    # 2: 1
    # 3: 1
    # 4-7: 2 (4 and 7)
    # 8-15: 2 (8 and 15)
    # 16+: 2 (16 and 25)
    assert stats["in_domain"]["histogram"] == [1, 1, 1, 1, 2, 2, 2]
    assert stats["in_domain"]["count"] == 10
    assert stats["in_domain"]["max_ulp"] == 25


def test_accumulate_nonfinite_mismatch():
    stats = sweep.make_stats()
    # Mismatches: NaN vs finite, Inf vs finite, +Inf vs -Inf
    got = np.array([np.nan, np.inf, np.inf, np.inf, np.nan], dtype=np.float32)
    want = np.array([1.0, 1.0, -np.inf, np.inf, np.nan], dtype=np.float32)
    in_domain = np.ones(len(got), dtype=bool)

    sweep.accumulate(stats, "expf", got, want, in_domain)
    # got[0] is NaN vs 1.0: mismatch
    # got[1] is Inf vs 1.0: mismatch
    # got[2] is +Inf vs -Inf: mismatch
    # got[3] is +Inf vs +Inf: match
    # got[4] is NaN vs NaN: match
    assert stats["in_domain"]["nonfinite_mismatch"] == 3


def test_fast_expf_oracle_scale_matches_probe_float():
    # probe.cu scales by the float constant 0.84f; the oracle must use the same value.
    import importlib.util
    import pathlib
    import struct

    path = pathlib.Path(__file__).resolve().parents[2] / "probes/fp-semantics/fast-expf-001/oracle.py"
    spec = importlib.util.spec_from_file_location("fast_expf_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SCALE == struct.unpack("<f", struct.pack("<f", 0.84))[0]

