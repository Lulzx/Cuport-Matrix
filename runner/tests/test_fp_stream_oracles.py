"""Oracles of the fp-semantics and streams-events probes added for v0: metadata,
hand-checked edge values, and invariants that hold whatever the inputs."""

import pathlib

import pytest

import cuport_oracle as co
from cuport.probe import load_features, load_oracle, load_probe

ROOT = pathlib.Path(__file__).resolve().parents[2]
FEATURES = load_features(ROOT / "matrix" / "features.yaml")


def oracle(probe_id):
    probe = load_probe(ROOT / "probes" / probe_id, FEATURES)
    expected, text = load_oracle(probe)
    assert text is None
    return probe, expected


@pytest.mark.parametrize("probe_id,feature", [
    ("fp-semantics/fma-contraction-001", "fma_contraction_default"),
    ("fp-semantics/denormals-001", "denormal_handling"),
    ("fp-semantics/fp64-emulation-001", "fp64_emulation_presence"),
    ("fp-semantics/half-bf16-001", "half_bf16_conversions"),
    ("fp-semantics/rounding-modes-001", "rounding_mode_intrinsics"),
    ("streams-events/event-timing-001", "event_elapsed_time"),
    ("streams-events/default-stream-sync-001", "default_stream_sync"),
])
def test_metadata(probe_id, feature):
    probe, _ = oracle(probe_id)
    assert probe.features == (feature,)
    assert (probe.oracle, probe.tolerance, probe.expect) == ("spec", "bit-exact", "output")


def test_fma_contraction_separates_fused_from_split():
    _, e = oracle("fp-semantics/fma-contraction-001")
    fused, split = e["fused"].values, e["split"].values
    assert e["expr"].values == fused
    even = [i for i in range(0, len(fused), 2) if fused[i] != split[i]]
    assert len(even) > 0.9 * len(fused) / 2
    assert all(split[i] == 0.0 for i in range(0, len(split), 2))  # c = -(a*b rounded)


def test_denormals_produce_subnormals_and_not_zeros():
    _, e = oracle("fp-semantics/denormals-001")
    bits = [co.f32_bits(v) & 0x7FFFFFFF for v in e["half_tiny"].values]
    assert sum(0 < b < 0x00800000 for b in bits) > 0.3 * len(bits)
    assert all(v != 0.0 for v in e["sub_scale"].values)
    assert all(v != 0.0 for v in e["d_scale"].values)


def test_fp64_results_are_not_float_representable():
    _, e = oracle("fp-semantics/fp64-emulation-001")
    for name in ("add", "mul", "div", "sqrt"):
        vals = e[name].values
        assert sum(co.f32(v) != v for v in vals) > 0.99 * len(vals), name


def test_half_bf16_edges():
    _, e = oracle("fp-semantics/half-bf16-001")
    h = {m: e[f"h_{m}"].values for m in co.ROUNDING_MODES}
    assert [h["rn"][i] for i in range(4)] == [0x0000, 0x8000, 0x7C00, 0xFC00]  # +-0, +-inf
    assert h["rn"][4] == 0x7BFF                         # 65504
    assert h["rn"][5] == 0x7C00 and h["rz"][5] == 0x7BFF  # 65520 tie: inf, or max toward zero
    assert h["rn"][6] == 0x7BFF                         # just below the tie
    assert h["rn"][7] == 0xFC00 and h["ru"][7] == 0xFBFF  # -65520
    assert h["rn"][8] == 0x0001 and h["rn"][9] == 0x0000  # 2^-24; 2^-25 ties to even
    assert h["rn"][13] == 0x3C00 and h["rn"][14] == 0x3C02  # half ties to even
    bf = e["bf_rn"].values
    assert bf[15] == 0x7F80 and e["bf_rz"].values[15] == 0x7F7F  # float max
    assert bf[16] == 0x7F80                             # tie above the largest bf16, odd: up
    assert bf[20] == 0x3F80 and bf[21] == 0x3F82        # bf16 ties to even


def test_rounding_modes_bracket_nearest():
    _, e = oracle("fp-semantics/rounding-modes-001")
    for op in ("add", "mul", "div", "sqrt", "fma"):
        rn, rz, ru, rd = (e[f"{op}_{m}"].values for m in co.ROUNDING_MODES)
        for i in range(len(rn)):
            assert rd[i] <= rn[i] <= ru[i]
            assert rz[i] == (rd[i] if rn[i] > 0 else ru[i])
        assert sum(u != d for u, d in zip(ru, rd)) > (0.8 if op == "add" else 0.99) * len(rn), op


def test_event_checks():
    _, e = oracle("streams-events/event-timing-001")
    assert e["checks"].values == [0, 0, 1, 1, 1, 1, 400, 400]


def test_default_stream_model_matches_scalar_reference():
    _, e = oracle("streams-events/default-stream-sync-001")
    M = 0xFFFFFFFF

    def mix(x, rounds):
        for r in range(rounds):
            x = (((x ^ (x >> 15)) * 0x2C1B3C6D) + r) & M
        return x

    inp = co.lcg_u32("streams-events/default-stream-sync-001", 3)
    for i in range(3):
        x = mix(inp[i], 4096)
        y = mix(x ^ 0x9E3779B9, 4096)
        assert (e["x"].values[i], e["y"].values[i], e["z"].values[i]) == (x, y, y ^ 0xA5A5A5A5)
