"""Cross-checks for probes/common: the Python oracle helpers and cuport.h must
produce identical inputs and identical output encodings."""

import os
import pathlib
import struct
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "probes" / "common"))

import cuport_oracle as co  # noqa: E402


def test_fnv1a64_reference_vectors():
    assert co.fnv1a64("") == 0xCBF29CE484222325
    assert co.fnv1a64("a") == 0xAF63DC4C8601EC8C
    assert co.fnv1a64("foobar") == 0x85944171F73967E8


def test_fnv1a64_matches_metal_baseline_seeds():
    # Seeds printed by baselines/metal/run.m on the M6.
    assert co.fnv1a64("metal/simd-ballot-001") == 0x9BAC5E46517E9B9B
    assert co.fnv1a64("metal/threadgroup-reduce-001") == 0x16C76FD69FA71E23


def test_lcg_from_zero_seed():
    lcg = co.Lcg.from_seed(0)
    x1 = 1442695040888963407
    x2 = (6364136223846793005 * x1 + 1442695040888963407) % 2**64
    assert lcg.next_u32() == x1 >> 32 == 0x14057B7E
    assert lcg.next_u32() == x2 >> 32


def test_lcg_u32_is_the_probe_stream():
    a = co.lcg_u32("x/y-001", 5)
    lcg = co.Lcg("x/y-001")
    assert a == [lcg.next_u32() for _ in range(5)]


def test_unit_f32_is_exact_and_in_range():
    lcg = co.Lcg.from_seed(0)
    u = lcg.next_u32()
    v = co.Lcg.from_seed(0).next_unit_f32()
    assert v == (u >> 8) / 16777216.0
    assert 0.0 <= v < 1.0
    assert co.f32(v) == v


def test_f32_rounding_and_bits():
    assert co.f32(0.1) == struct.unpack("<f", struct.pack("<f", 0.1))[0]
    assert co.f32_bits(1.0) == 0x3F800000
    assert co.bits_f32(0x3F800000) == 1.0
    assert co.f64_bits(1.0) == 0x3FF0000000000000


@pytest.mark.skipif(os.name == "nt", reason="builds the header with a POSIX c++")
def test_header_matches_python(tmp_path):
    src = tmp_path / "t.cpp"
    src.write_text(
        '#define CUPORT_NO_CUDA\n'
        f'#include "{ROOT}/probes/common/cuport.h"\n'
        "int main() {\n"
        '  CuportLcg g("x/y-001");\n'
        "  uint32_t v[3] = {g.next_u32(), g.next_u32(), g.next_u32()};\n"
        '  cuport_emit_u32("a", v, 3);\n'
        "  float f[2] = {g.next_unit_f32(), -0.0f};\n"
        '  cuport_emit_f32("b", f, 2);\n'
        "  uint64_t w[1] = {cuport_fnv1a64(\"foobar\")};\n"
        '  cuport_emit_u64("c", w, 1);\n'
        "  return cuport_done();\n"
        "}\n"
    )
    exe = tmp_path / "t"
    subprocess.run(["clang++", "-std=c++17", "-Wall", "-Werror", str(src), "-o", str(exe)], check=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout

    lcg = co.Lcg("x/y-001")
    a = [lcg.next_u32() for _ in range(3)]
    b0 = co.f32_bits(lcg.next_unit_f32())
    assert out.splitlines() == [
        "CUPORT a u32 3 " + " ".join(f"{x:08x}" for x in a),
        f"CUPORT b f32 2 {b0:08x} 80000000",
        "CUPORT c u64 1 85944171f73967e8",
        "CUPORT-DONE",
    ]


@pytest.mark.parametrize("dtype", ["u32", "i32", "u64", "f32", "f64", "u16"])
def test_every_dtype_has_a_width(dtype):
    assert co.DTYPE_HEX_WIDTH[dtype] in (4, 8, 16)
