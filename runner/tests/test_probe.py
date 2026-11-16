import pytest

from cuport.probe import ProbeError, discover, load_oracle, load_probe
from helpers import FEATURES, make_probe


def test_load_probe_defaults(tmp_path):
    p = load_probe(make_probe(tmp_path), FEATURES)
    assert p.id == "warp-primitives/shfl-xor-001"
    assert p.features == ("shfl_xor_width",)
    assert p.outputs == {} and p.observe_only is False and p.timeout_s == 5
    assert p.tolerance_for("a") == "bit-exact"


def test_output_tolerance_override(tmp_path):
    p = load_probe(make_probe(tmp_path, yaml_extra='outputs:\n  a: {tolerance: "ulp:2"}\n'), FEATURES)
    assert p.tolerance_for("a") == "ulp:2"


@pytest.mark.parametrize("yaml_extra,fragment", [
    ("oracle: guess\n", "oracle"),
    ("expect: maybe\n", "expect"),
    ("tolerance: roughly\n", "tolerance"),
    ("bogus_key: 1\n", "bogus_key"),
    ("timeout_s: 0\n", "timeout_s"),
])
def test_probe_yaml_validation(tmp_path, yaml_extra, fragment):
    d = make_probe(tmp_path)
    key = yaml_extra.split(":")[0]
    lines = [l for l in (d / "probe.yaml").read_text().splitlines() if not l.startswith(key + ":")]
    (d / "probe.yaml").write_text("\n".join(lines) + "\n" + yaml_extra)
    with pytest.raises(ProbeError, match=fragment):
        load_probe(d, FEATURES)


def test_probe_id_must_match_directory(tmp_path):
    d = make_probe(tmp_path)
    (d / "probe.yaml").write_text((d / "probe.yaml").read_text().replace("shfl-xor-001", "other-001"))
    with pytest.raises(ProbeError, match="directory"):
        load_probe(d, FEATURES)


def test_unknown_feature_rejected(tmp_path):
    with pytest.raises(ProbeError, match="feature"):
        load_probe(make_probe(tmp_path, features="[nope]"), FEATURES)


def test_discover_skips_common_and_collects_errors(tmp_path):
    make_probe(tmp_path)
    make_probe(tmp_path, "atomics/cas-loop-001", features="[atomicCAS_loop]")
    (tmp_path / "common").mkdir()
    bad = make_probe(tmp_path, "atomics/bad-001", features="[atomicCAS_loop]")
    (bad / "probe.cu").unlink()
    probes, errors = discover(tmp_path, FEATURES)
    assert [p.id for p in probes] == ["atomics/cas-loop-001", "warp-primitives/shfl-xor-001"]
    assert len(errors) == 1 and "probe.cu" in errors[0]


def test_load_oracle_with_text_lines(tmp_path):
    d = make_probe(tmp_path, oracle="TEXT_LINES = ['x']\ndef expected():\n    return {}\n")
    buffers, text = load_oracle(load_probe(d, FEATURES))
    assert buffers == {} and text == ["x"]
