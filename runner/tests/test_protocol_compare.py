import math
import struct

import pytest

from cuport.compare import compare_buffer, parse_tolerance, ulp_distance
from cuport.protocol import parse_output

import cuport_oracle as co


def test_parse_buffers_text_and_done():
    out = (
        "hello from block 0\n"
        "CUPORT a u32 3 00000001 ffffffff 0000000a\n"
        "CUPORT b f32 1 3f800000\n"
        "CUPORT-DONE\n"
    )
    p = parse_output(out)
    assert p.buffers == {"a": ("u32", [1, 0xFFFFFFFF, 10]), "b": ("f32", [0x3F800000])}
    assert p.text == ["hello from block 0"]
    assert p.done and not p.errors and not p.malformed


def test_parse_error_line_and_missing_done():
    p = parse_output("CUPORT-ERROR cudaMalloc(&d,n) 2 out of memory\n")
    assert p.errors == ["cudaMalloc(&d,n) 2 out of memory"]
    assert not p.done


@pytest.mark.parametrize("line", [
    "CUPORT a u32 3 00000001 00000002",       # count mismatch
    "CUPORT a q32 1 00000001",                # unknown dtype
    "CUPORT a u32 1 zz",                      # not hex
    "CUPORT a u32 1 100000000",               # wider than dtype
])
def test_malformed_lines_are_reported(line):
    p = parse_output(line + "\nCUPORT-DONE\n")
    assert p.malformed == [line]
    assert "a" not in p.buffers


def test_duplicate_buffer_name_is_malformed():
    p = parse_output("CUPORT a u32 1 00000001\nCUPORT a u32 1 00000002\nCUPORT-DONE\n")
    assert p.malformed == ["CUPORT a u32 1 00000002"]


def test_parse_tolerance():
    assert parse_tolerance("bit-exact") == ("bit-exact", 0)
    assert parse_tolerance("ulp:2") == ("ulp", 2)
    assert parse_tolerance("abs:1e-6") == ("abs", 1e-6)
    assert parse_tolerance("rel:0.5") == ("rel", 0.5)
    with pytest.raises(ValueError):
        parse_tolerance("ulp:-1")
    with pytest.raises(ValueError):
        parse_tolerance("close-enough")


def test_ulp_distance_f32_crosses_zero():
    assert ulp_distance("f32", 0x00000000, 0x80000000) == 0      # +0 vs -0
    assert ulp_distance("f32", 0x00000001, 0x80000001) == 2
    assert ulp_distance("f32", co.f32_bits(1.0), co.f32_bits(1.0) + 3) == 3


def test_integer_buffers_exact_or_wrong():
    exp = co.Buffer("u32", [1, 2, 3])
    assert compare_buffer(exp, "u32", [1, 2, 3], "bit-exact").status == "bit-exact"
    r = compare_buffer(exp, "u32", [1, 2, 4], "bit-exact")
    assert r.status == "mismatch" and r.mismatches == 1 and "[2]" in r.detail


def test_i32_expected_negative_values_compare_to_twos_complement_bits():
    exp = co.Buffer("i32", [-1, 5])
    assert compare_buffer(exp, "i32", [0xFFFFFFFF, 5], "bit-exact").status == "bit-exact"


def test_float_tolerances():
    one = co.f32_bits(1.0)
    exp = co.Buffer("f32", [1.0])
    assert compare_buffer(exp, "f32", [one], "ulp:1").status == "bit-exact"
    assert compare_buffer(exp, "f32", [one + 1], "ulp:1").status == "within-tolerance"
    assert compare_buffer(exp, "f32", [one + 2], "ulp:1").status == "mismatch"
    assert compare_buffer(exp, "f32", [co.f32_bits(1.0 + 1e-7)], "abs:1e-6").status == "within-tolerance"
    assert compare_buffer(exp, "f32", [co.f32_bits(1.1)], "rel:0.05").status == "mismatch"


def test_nan_matches_any_nan_only_within_tolerance():
    exp = co.Buffer("f32", [math.nan])
    canonical = co.f32_bits(math.nan)
    other_nan = 0x7FC00001
    assert compare_buffer(exp, "f32", [canonical], "ulp:0").status == "bit-exact"
    assert compare_buffer(exp, "f32", [other_nan], "ulp:0").status == "within-tolerance"
    assert compare_buffer(exp, "f32", [other_nan], "bit-exact").status == "mismatch"
    assert compare_buffer(exp, "f32", [one_bits := co.f32_bits(1.0)], "abs:10").status == "mismatch"
    assert one_bits


def test_f64_and_u16():
    exp = co.Buffer("f64", [0.1])
    bits = struct.unpack("<Q", struct.pack("<d", 0.1))[0]
    assert compare_buffer(exp, "f64", [bits], "bit-exact").status == "bit-exact"
    assert compare_buffer(co.Buffer("u16", [0x3C00]), "u16", [0x3C00], "bit-exact").status == "bit-exact"


def test_shape_and_type_mismatches():
    exp = co.Buffer("u32", [1, 2])
    assert compare_buffer(exp, "u32", [1], "bit-exact").status == "mismatch"
    assert compare_buffer(exp, "i32", [1, 2], "bit-exact").status == "mismatch"
