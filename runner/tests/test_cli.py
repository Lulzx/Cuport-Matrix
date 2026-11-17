import json
import os
import sys
import types

import pytest

import cuport.__main__ as cli
from helpers import GOOD, make_probe
from test_run import FakeAdapter

FEATURES_YAML = """\
schema: 1
families:
  - id: warp-primitives
    description: x
    features: [shfl_xor_width, ballot_any_all]
"""


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "matrix").mkdir()
    (tmp_path / "matrix" / "features.yaml").write_text(FEATURES_YAML)
    make_probe(tmp_path / "probes", stdout=GOOD)
    monkeypatch.setattr(cli, "collect_hardware", lambda hw: {"id": hw, "macos": "27.2 (26B123)", "profiler": "Apple M6"})
    module = types.ModuleType("cuport.adapters.fake")
    module.ADAPTER = FakeAdapter()
    monkeypatch.setitem(sys.modules, "cuport.adapters.fake", module)
    return tmp_path


def test_check_ok_and_failure(repo, capsys):
    assert cli.main(["--root", str(repo), "check"]) == 0
    assert "1 probes" in capsys.readouterr().out
    (repo / "probes" / "warp-primitives" / "shfl-xor-001" / "oracle.py").write_text("def expected(:\n")
    assert cli.main(["--root", str(repo), "check"]) == 1


@pytest.mark.skipif(os.name == "nt", reason="fake tools are POSIX shell scripts")
def test_run_writes_valid_row_and_refuses_overwrite(repo, capsys):
    args = ["--root", str(repo), "run", "--tool", "fake", "--hardware", "m6-12c", "--date", "2026-11-16",
            "--jobs", "2", "--gpu-jobs", "1"]
    assert cli.main(args) == 0
    path = repo / "results" / "fake" / "m6-12c" / "2026-11-16.json"
    row = json.loads(path.read_text())
    assert row["cells"][0]["class"] == "bit-exact"
    cfg = row["runner"]["config"]
    assert cfg["jobs"] == 2 and cfg["gpu_jobs"] == 1 and "FNV-1a" in cfg["seed"]
    assert cli.main(["--root", str(repo), "validate", str(path)]) == 0
    assert cli.main(args) == 2  # exists
    assert cli.main(args + ["--force"]) == 0


def test_run_with_unknown_probe_filter_fails(repo):
    assert cli.main(["--root", str(repo), "run", "--tool", "fake", "--hardware", "m6-12c",
                     "--probe", "atomics/"]) == 1


VALID_CU = """#include "../../common/cuport.h"
__global__ void k(uint32_t *o) { o[threadIdx.x] = blockIdx.x; __syncthreads(); }
int main() {
    uint32_t *d;
    CUPORT_CHECK(cudaMalloc(&d, 128));
    k<<<1, 32>>>(d);
    CUPORT_CHECK(cudaDeviceSynchronize());
    return cuport_done();
}
"""


@pytest.mark.skipif(os.name == "nt", reason="needs clang++ and symlinks")
def test_check_syntax(repo, capsys):
    d = repo / "probes" / "warp-primitives" / "shfl-xor-001"
    (repo / "probes" / "common").symlink_to(cli.ROOT / "probes" / "common")
    (d / "probe.cu").write_text(VALID_CU)
    assert cli.main(["--root", str(repo), "check", "--syntax"]) == 0
    capsys.readouterr()
    (d / "probe.cu").write_text(VALID_CU.replace("blockIdx.x;", "blockIdx.q;"))
    assert cli.main(["--root", str(repo), "check", "--syntax"]) == 1
    assert "no member named 'q'" in capsys.readouterr().out
