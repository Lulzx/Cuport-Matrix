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

    # Each launch k maps x -> (x ^ (k * 0x9E3779B9)) * 0x01000193 (mod 2^32), which
    # does not commute across k, so a reordered chain gives a different result.
    import cuport_oracle as co
    M = 0xFFFFFFFF

    def chain(x, order):
        for k in order:
            x = ((x ^ ((k * 0x9E3779B9) & M)) * 0x01000193) & M
        return x

    first_in = co.Lcg("streams-events/stream-ordering-001").next_u32()
    assert buf.values[0] == chain(first_in, range(64))
    assert buf.values[0] != chain(first_in, [1, 0, *range(2, 64)])
    assert buf.values[0] != chain(first_in, reversed(range(64)))


def test_async_memcpy_overlap_probe_and_oracle():
    features = load_features(ROOT / "matrix" / "features.yaml")
    probe_dir = ROOT / "probes" / "streams-events" / "async-memcpy-overlap-001"
    probe = load_probe(probe_dir, features)
    assert probe.id == "streams-events/async-memcpy-overlap-001"
    assert probe.family == "streams-events"
    assert probe.features == ("async_memcpy_overlap",)
    assert probe.oracle == "spec"
    assert probe.tolerance == "bit-exact"
    assert probe.expect == "output"

    expected, text = load_oracle(probe)
    assert text is None
    assert "sample" in expected
    assert "count_wrong" in expected

    wrong = expected["count_wrong"]
    assert wrong.dtype == "u32"
    assert list(wrong.values) == [0]

    sample = expected["sample"]
    assert sample.dtype == "u32"
    assert len(sample.values) == 4096

    import cuport_oracle as co
    lcg = co.Lcg("streams-events/async-memcpy-overlap-001")
    first_in = lcg.next_u32()
    # For index 0, chunk is 0, y = x * 2654435761 + 0
    expected_first = (first_in * 2654435761) & 0xFFFFFFFF
    assert sample.values[0] == expected_first

    # The oracle skips ahead 1024 LCG draws per sample; check against plain draws.
    g = co.Lcg("streams-events/async-memcpy-overlap-001")
    draws = [g.next_u32() for _ in range(3 * 1024 + 1)]
    for j in (1, 2, 3):
        assert sample.values[j] == (draws[j * 1024] * 2654435761) & 0xFFFFFFFF
