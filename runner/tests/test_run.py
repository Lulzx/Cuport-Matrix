import os

import pytest

from cuport.adapters import execute
from cuport.probe import discover
from cuport.run import run_matrix
from helpers import FEATURES, GOOD, make_probe, step


class FakeAdapter:
    id = "fake"
    version = "v0"

    def available(self):
        return True, "fake"

    def describe(self):
        return {"id": "fake", "version": "v0", "commit": "0" * 40}

    def compile(self, probe_dir, out_dir, timeout_s):
        if (probe_dir / "refuse").exists():
            return step(rc=1, err="error: unsupported\n"), None
        exe = out_dir / "probe"
        exe.write_text(f"#!/bin/sh\ncat '{probe_dir / 'fake_stdout.txt'}'\n")
        exe.chmod(0o755)
        return step(), exe

    def run(self, executable, timeout_s):
        return execute([str(executable)], timeout_s)


@pytest.mark.skipif(os.name == "nt", reason="fake tools are POSIX shell scripts")
def test_run_matrix_end_to_end(tmp_path):
    root = tmp_path / "probes"
    make_probe(root, stdout=GOOD)
    make_probe(root, "warp-primitives/ballot-001", features="[ballot_any_all]",
               stdout="CUPORT a u32 2 00000001 00000009\nCUPORT-DONE\n")
    r = make_probe(root, "atomics/cas-loop-001", features="[atomicCAS_loop]", stdout="")
    (r / "refuse").touch()
    probes, errors = discover(root, FEATURES)
    assert not errors
    seen = []
    cells = run_matrix(FakeAdapter(), probes, tmp_path / "build", jobs=4, gpu_jobs=2, triage=[],
                       progress=lambda p, c: seen.append(p.id))
    got = {c["probe"]: c["class"] for c in cells}
    assert got == {"warp-primitives/shfl-xor-001": "bit-exact",
                   "warp-primitives/ballot-001": "silently-wrong",
                   "atomics/cas-loop-001": "refused-with-diagnostic"}
    assert [c["probe"] for c in cells] == sorted(got)
    assert sorted(seen) == sorted(got)


@pytest.mark.skipif(os.name == "nt", reason="fake tools are POSIX shell scripts")
def test_triage_is_applied_by_tool_id(tmp_path):
    root = tmp_path / "probes"
    make_probe(root, stdout="CUPORT a u32 2 00000001 00000009\nCUPORT-DONE\n")
    probes, _ = discover(root, FEATURES)
    triage = [{"tool": "fake", "probe": "warp-primitives/shfl-xor-001", "raw_class": "silently-wrong",
               "class": "documented-divergence", "reference": "https://example.org", "upstream": None,
               "notes": "documented"}]
    cells = run_matrix(FakeAdapter(), probes, tmp_path / "build", jobs=1, gpu_jobs=1, triage=triage)
    assert cells[0]["class"] == "documented-divergence" and cells[0]["reference"] == "https://example.org"
