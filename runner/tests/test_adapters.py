import pathlib

import pytest

import cuport.adapters as ad
from cuport.adapters import COMMON_DIR, StepResult, load_adapter


@pytest.fixture
def tools(tmp_path, monkeypatch):
    monkeypatch.setattr(ad, "TOOLS_ROOT", tmp_path)
    calls = []

    def fake_execute(cmd, timeout_s, cwd=None, env=None):
        calls.append(cmd)
        return StepResult(cmd, 0, "", "", 0.0, False)

    monkeypatch.setattr(ad, "execute", fake_execute)
    return tmp_path, calls


@pytest.mark.parametrize("tool", ["mvcc", "cumetal"])
def test_unavailable_when_not_installed(tools, tool):
    ok, detail = load_adapter(tool).available()
    assert not ok and "not installed" in detail


def test_mvcc_compile_command(tools, tmp_path):
    root, calls = tools
    probe = tmp_path / "p"
    out = tmp_path / "o"
    step, exe = load_adapter("mvcc").compile(probe, out, 10)
    assert exe == out / "probe"
    assert calls[-1] == [str(root / "mvcc-v1.3.0" / "bin" / "nvcc"), "-std=c++17", "-I", str(COMMON_DIR),
                         "-o", str(out / "probe"), str(probe / "probe.cu")]


def test_cumetal_compile_command(tools, tmp_path):
    root, calls = tools
    probe = tmp_path / "p"
    out = tmp_path / "o"
    load_adapter("cumetal").compile(probe, out, 10)
    assert calls[-1] == [str(root / "cumetal-v0.9.1" / "bin" / "cumetalc"), str(probe / "probe.cu"),
                         "-I", str(COMMON_DIR), "-o", str(out / "probe")]


def test_failed_compile_returns_no_executable(tools, tmp_path, monkeypatch):
    monkeypatch.setattr(ad, "execute", lambda cmd, t, cwd=None, env=None: StepResult(cmd, 1, "", "error: x", 0.0, False))
    step, exe = load_adapter("mvcc").compile(tmp_path / "p", tmp_path / "o", 10)
    assert exe is None and step.returncode == 1


def test_run_invokes_executable(tools):
    _, calls = tools
    load_adapter("mvcc").run(pathlib.Path("/x/probe"), 5)
    assert calls[-1] == [str(pathlib.Path("/x/probe"))]


def test_describe_has_pinned_version(tools):
    for tool, version in (("mvcc", "v1.3.0"), ("cumetal", "v0.9.1")):
        d = load_adapter(tool).describe()
        assert d["id"] == tool and d["version"] == version and "commit" in d
