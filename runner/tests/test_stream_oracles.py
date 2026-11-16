"""Tests for streams-events probe metadata and oracles."""

import pathlib
import pytest

from cuport.probe import load_features, load_oracle, load_probe

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_stream_ordering_probe_and_oracle():
    features = load_features(ROOT / "matrix" / "features.yaml")
    probe_dir = ROOT / "probes" / "streams-events" / "stream-ordering-001"
    probe = load_probe(probe_dir, features)
    assert probe.id == "streams-events/stream-ordering-001"
    assert probe.family == "streams-events"
    assert probe.features == ("stream_ordering",)
    assert probe.oracle == "spec"
    assert probe.tolerance == "bit-exact"
    assert probe.expect == "output"

    expected, text = load_oracle(probe)
    assert text is None
    assert "out" in expected
    buf = expected["out"]
    assert buf.dtype == "u32"
    assert len(buf.values) == 65536

    # Verify first element mathematically:
    import cuport_oracle as co
    lcg = co.Lcg("streams-events/stream-ordering-001")
    first_in = lcg.next_u32()
    # sum_{k=0..63} k * 0x9E3779B9 = 2016 * 0x9E3779B9 = 0xf4de90e0 (mod 2^32)
    expected_first = (first_in + 0xF4DE90E0) & 0xFFFFFFFF
    assert buf.values[0] == expected_first
