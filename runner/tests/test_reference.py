import os
import pathlib
import subprocess
import pytest

import cuport.adapters as ad
from cuport.adapters import COMMON_DIR, StepResult, load_adapter


def test_nvcc_adapter_compile_flags(monkeypatch, tmp_path):
    calls = []

    def fake_execute(cmd, timeout_s, cwd=None, env=None):
        calls.append(cmd)
        return StepResult(cmd, 0, "", "", 0.0, False)

    monkeypatch.setattr(ad, "execute", fake_execute)
    monkeypatch.setattr(ad, "WINDOWS", False)
    monkeypatch.setattr(ad, "EXE_SUFFIX", "")
    monkeypatch.setenv("CUPORT_NVCC", "/custom/nvcc")
    monkeypatch.setenv("CUPORT_NVCC_ARCH", "sm_89")

    adapter = load_adapter("nvcc")
    assert adapter.id == "nvcc"

    # Regular probe
    regular_probe = tmp_path / "warp-primitives" / "shfl-xor-001"
    regular_probe.mkdir(parents=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)

    step, exe = adapter.compile(regular_probe, out_dir, 10.0)
    assert exe == out_dir / "probe"
    assert calls[-1] == [
        "/custom/nvcc",
        "-std=c++17",
        "-arch=sm_89",
        "-I",
        str(COMMON_DIR),
        "-o",
        str(out_dir / "probe"),
        str(regular_probe / "probe.cu"),
    ]

    # dynamic-parallelism-001 probe: should include -rdc=true -lcudadevrt
    dp_probe = tmp_path / "limits-refusals" / "dynamic-parallelism-001"
    dp_probe.mkdir(parents=True)

    step, exe = adapter.compile(dp_probe, out_dir, 10.0)
    assert calls[-1] == [
        "/custom/nvcc",
        "-std=c++17",
        "-arch=sm_89",
        "-I",
        str(COMMON_DIR),
        "-rdc=true",
        "-lcudadevrt",
        "-o",
        str(out_dir / "probe"),
        str(dp_probe / "probe.cu"),
    ]


def test_nvcc_adapter_windows_uses_msvc(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ad, "execute", lambda cmd, t, cwd=None, env=None: calls.append(cmd) or
                        StepResult(cmd, 0, "", "", 0.0, False))
    monkeypatch.setattr(ad, "WINDOWS", True)
    monkeypatch.setattr(ad, "EXE_SUFFIX", ".exe")
    monkeypatch.setenv("CUPORT_NVCC", "nvcc")
    monkeypatch.setenv("CUPORT_NVCC_ARCH", "sm_89")
    monkeypatch.setenv("CUPORT_NVCC_CCBIN", "C:/VC/bin/Hostx64/x64")
    probe = tmp_path / "limits-refusals" / "cooperative-grid-sync-001"
    probe.mkdir(parents=True)
    step, exe = load_adapter("nvcc").compile(probe, tmp_path / "out", 10.0)
    assert exe == tmp_path / "out" / "probe.exe"
    assert calls[-1][:6] == ["nvcc", "-ccbin", str(pathlib.Path("C:/VC/bin/Hostx64/x64")),
                             "-Xcompiler", "/Zc:preprocessor", "-std=c++17"]


def test_nvcc_adapter_arch_detection(monkeypatch, tmp_path):
    calls = []

    def fake_execute(cmd, timeout_s, cwd=None, env=None):
        calls.append(cmd)
        return StepResult(cmd, 0, "", "", 0.0, False)

    monkeypatch.setattr(ad, "execute", fake_execute)
    monkeypatch.delenv("CUPORT_NVCC_ARCH", raising=False)

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["nvidia-smi", "--query-gpu=compute_cap"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="8.9\n8.9\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = load_adapter("nvcc")
    probe = tmp_path / "warp-primitives" / "shfl-xor-001"
    probe.mkdir(parents=True)
    out = tmp_path / "out"

    adapter.compile(probe, out, 10.0)
    assert "-arch=sm_89" in calls[-1]


def test_write_drop_creates_files_and_ordered_meta(tmp_path):
    import cuport.reference as ref

    drop_dir = tmp_path / "drops" / "p1"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    compile_cmd = [
        "nvcc",
        "-std=c++17",
        "-arch=sm_89",
        "-I",
        str(repo_root / "probes" / "common"),
        "-o",
        str(repo_root / "build" / "probe"),
        str(repo_root / "probes" / "family" / "p1" / "probe.cu"),
    ]
    compile_step = StepResult(compile_cmd, 0, "build out\n", "build err\n", 0.1, False)
    run_step = StepResult(["probe"], 0, "run out\n", "run err\n", 0.05, False)

    env = {
        "gpu": "NVIDIA GeForce RTX 4090 (sm_89)",
        "driver": "595.99 (nvidia-smi reports CUDA 12.6)",
        "nvcc": "Cuda compilation tools, release 12.6, V12.6.20",
        "host": "gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0",
        "repo_commit": "abcdef1234567890",
        "date": "2026-11-16",
        "root": repo_root,
    }

    ref.write_drop(drop_dir, "family/p1", compile_step, run_step, env)

    assert (drop_dir / "output.txt").read_text() == "run out\n"
    assert (drop_dir / "stderr.txt").read_text() == "run err\n"
    assert (drop_dir / "build.log").read_text() == "build out\nbuild err\n"

    meta_lines = (drop_dir / "meta.txt").read_text().splitlines()
    expected_keys = [
        "probe:",
        "gpu:",
        "driver:",
        "nvcc:",
        "host compiler:",
        "command:",
        "build exit:",
        "run exit:",
        "date:",
        "repo commit:",
        "seed:",
    ]
    actual_keys = [line.split(":", 1)[0] + ":" for line in meta_lines]
    assert actual_keys == expected_keys

    meta_dict = dict(line.split(": ", 1) for line in meta_lines)
    assert meta_dict["probe"] == "family/p1"
    assert meta_dict["gpu"] == "NVIDIA GeForce RTX 4090 (sm_89)"
    assert meta_dict["driver"] == "595.99 (nvidia-smi reports CUDA 12.6)"
    assert meta_dict["nvcc"] == "Cuda compilation tools, release 12.6, V12.6.20"
    assert meta_dict["host compiler"] == "gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
    assert "probes/common" in meta_dict["command"]
    assert str(repo_root) not in meta_dict["command"]  # must be repo-relative
    assert meta_dict["build exit"] == "0"
    assert meta_dict["run exit"] == "0"
    assert meta_dict["date"] == "2026-11-16"
    assert meta_dict["repo commit"] == "abcdef1234567890"
    assert "FNV-1a" in meta_dict["seed"]


def test_write_drop_without_run_step(tmp_path):
    import cuport.reference as ref

    drop_dir = tmp_path / "drops" / "failed_compile"
    compile_step = StepResult(["nvcc", "p.cu"], 1, "", "syntax error\n", 0.1, False)

    env = {
        "gpu": "NVIDIA RTX 4500 Ada (sm_89)",
        "driver": "596.71 (nvidia-smi reports CUDA 13.2)",
        "nvcc": "Cuda compilation tools, release 13.5",
        "host": "MSVC 14.44",
        "repo_commit": "123456",
        "date": "2026-11-16",
    }

    ref.write_drop(drop_dir, "family/p2", compile_step, None, env)

    assert (drop_dir / "output.txt").read_text() == ""
    assert (drop_dir / "stderr.txt").read_text() == ""
    assert (drop_dir / "build.log").read_text() == "syntax error\n"

    meta_dict = dict(line.split(": ", 1) for line in (drop_dir / "meta.txt").read_text().splitlines())
    assert meta_dict["build exit"] == "1"
    assert meta_dict["run exit"] == "not run"


def _make_fake_drop(drop_dir, stdout="CUPORT a u32 2 00000001 00000002\nCUPORT-DONE\n",
                    stderr="", build_exit=0, run_exit=0):
    drop_dir.mkdir(parents=True, exist_ok=True)
    (drop_dir / "output.txt").write_text(stdout)
    (drop_dir / "stderr.txt").write_text(stderr)
    (drop_dir / "build.log").write_text("build log\n")
    meta = [
        "probe: family/p1",
        "gpu: NVIDIA RTX 4090",
        "driver: 595.99",
        "nvcc: release 12.6",
        "host compiler: gcc",
        "command: nvcc probe.cu",
        f"build exit: {build_exit}",
        f"run exit: {run_exit}",
        "date: 2026-11-16",
        "repo commit: 123456",
        "seed: default",
    ]
    (drop_dir / "meta.txt").write_text("\n".join(meta) + "\n")


def test_compare_drops_identical(tmp_path):
    import cuport.reference as ref

    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_fake_drop(a)
    _make_fake_drop(b)

    cmp = ref.compare_drops(a, b)
    assert cmp.agree is True
    assert cmp.problems == []
    assert cmp.notes == []


def test_compare_drops_bit_flip(tmp_path):
    import cuport.reference as ref

    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_fake_drop(a, stdout="CUPORT y u32 2 00000001 00000002\nCUPORT-DONE\n")
    _make_fake_drop(b, stdout="CUPORT y u32 2 00000001 00000003\nCUPORT-DONE\n")

    cmp = ref.compare_drops(a, b)
    assert cmp.agree is False
    assert any("buffer y" in p for p in cmp.problems)


def test_compare_drops_printf_order(tmp_path):
    import cuport.reference as ref

    a = tmp_path / "a"
    b = tmp_path / "b"
    stdout_a = "line 1\nline 2\nCUPORT a u32 1 00000001\nCUPORT-DONE\n"
    stdout_b = "line 2\nline 1\nCUPORT a u32 1 00000001\nCUPORT-DONE\n"
    _make_fake_drop(a, stdout=stdout_a)
    _make_fake_drop(b, stdout=stdout_b)

    cmp = ref.compare_drops(a, b)
    assert cmp.agree is True
    assert cmp.problems == []


def test_compare_drops_timing_lines(tmp_path):
    import cuport.reference as ref

    a = tmp_path / "a"
    b = tmp_path / "b"
    stdout_a = "timing chain64_ms median=1.23\nCUPORT a u32 1 00000001\nCUPORT-DONE\n"
    stdout_b = "timing chain64_ms median=4.56\nCUPORT a u32 1 00000001\nCUPORT-DONE\n"
    _make_fake_drop(a, stdout=stdout_a)
    _make_fake_drop(b, stdout=stdout_b)

    cmp = ref.compare_drops(a, b)
    assert cmp.agree is True
    assert cmp.problems == []


def test_compare_drops_stderr_note(tmp_path):
    import cuport.reference as ref

    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_fake_drop(a, stderr="warning: host compiler A\n")
    _make_fake_drop(b, stderr="warning: host compiler B\n")

    cmp = ref.compare_drops(a, b)
    assert cmp.agree is True
    assert cmp.problems == []
    assert cmp.notes == ["stderr differs"]


def test_compare_drops_run_exit(tmp_path):
    import cuport.reference as ref

    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_fake_drop(a, run_exit=0)
    _make_fake_drop(b, run_exit=1)

    cmp = ref.compare_drops(a, b)
    assert cmp.agree is False
    assert any("run exit" in p for p in cmp.problems)


def test_compare_drops_missing_done(tmp_path):
    import cuport.reference as ref

    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_fake_drop(a, stdout="CUPORT a u32 1 00000001\nCUPORT-DONE\n")
    _make_fake_drop(b, stdout="CUPORT a u32 1 00000001\n")

    cmp = ref.compare_drops(a, b)
    assert cmp.agree is False
    assert any("CUPORT-DONE" in p for p in cmp.problems)


def test_refdrop_cli_refuses_without_force(tmp_path, monkeypatch):
    import cuport.__main__ as cli
    from helpers import GOOD, make_probe
    from test_cli import FEATURES_YAML

    repo = tmp_path / "repo"
    (repo / "matrix").mkdir(parents=True)
    (repo / "matrix" / "features.yaml").write_text(FEATURES_YAML)
    probe_dir = make_probe(repo / "probes", stdout=GOOD)

    existing_drop = probe_dir / "reference" / "tag1"
    existing_drop.mkdir(parents=True)

    assert cli.main(["--root", str(repo), "refdrop", "--tag", "tag1"]) == 2


def test_refdrop_cli_runs_and_writes_drop(tmp_path, monkeypatch, capsys):
    import cuport.__main__ as cli
    from helpers import GOOD, make_probe
    from test_cli import FEATURES_YAML

    repo = tmp_path / "repo"
    (repo / "matrix").mkdir(parents=True)
    (repo / "matrix" / "features.yaml").write_text(FEATURES_YAML)
    probe_dir = make_probe(repo / "probes", stdout=GOOD)

    calls = []

    def fake_execute(cmd, timeout_s, cwd=None, env=None):
        calls.append(cmd)
        if cmd[0] == "nvcc":
            return StepResult(cmd, 0, "compile out\n", "", 0.1, False)
        # executable run
        return StepResult(cmd, 0, GOOD, "", 0.05, False)

    monkeypatch.setattr(ad, "execute", fake_execute)
    monkeypatch.setenv("CUPORT_NVCC_ARCH", "sm_89")

    # Mock nvcc availability & describe
    nvcc_adapter = load_adapter("nvcc")
    monkeypatch.setattr(nvcc_adapter, "available", lambda: (True, "mocked nvcc"))
    import cuport.reference as ref
    monkeypatch.setattr(ref, "collect_env", lambda root=None: {
        "gpu": "NVIDIA RTX 4090 (sm_89)",
        "driver": "595.99 (nvidia-smi reports CUDA 12.6)",
        "nvcc": "Cuda compilation tools, release 12.6",
        "host": "gcc 11.4.0",
        "repo_commit": "abcdef",
        "date": "2026-11-16",
        "root": repo,
    })

    rc = cli.main(["--root", str(repo), "refdrop", "--tag", "tag1"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "warp-primitives/shfl-xor-001" in out
    assert "bit-exact" in out

    drop_dir = probe_dir / "reference" / "tag1"
    assert (drop_dir / "output.txt").read_text() == GOOD
    assert (drop_dir / "meta.txt").exists()

    # With existing drop and --force, should overwrite and succeed
    rc_force = cli.main(["--root", str(repo), "refdrop", "--tag", "tag1", "--force"])
    assert rc_force == 0


def test_refdrop_cli_fails_when_probe_fails(tmp_path, monkeypatch, capsys):
    import cuport.__main__ as cli
    from helpers import make_probe
    from test_cli import FEATURES_YAML

    repo = tmp_path / "repo"
    (repo / "matrix").mkdir(parents=True)
    (repo / "matrix" / "features.yaml").write_text(FEATURES_YAML)
    # Output that does not match expected oracle
    bad_output = "CUPORT a u32 2 00000001 00000099\nCUPORT-DONE\n"
    make_probe(repo / "probes", stdout=bad_output)

    def fake_execute(cmd, timeout_s, cwd=None, env=None):
        if cmd[0] == "nvcc":
            return StepResult(cmd, 0, "", "", 0.1, False)
        return StepResult(cmd, 0, bad_output, "", 0.05, False)

    monkeypatch.setattr(ad, "execute", fake_execute)
    monkeypatch.setenv("CUPORT_NVCC_ARCH", "sm_89")

    nvcc_adapter = load_adapter("nvcc")
    monkeypatch.setattr(nvcc_adapter, "available", lambda: (True, "mocked nvcc"))
    import cuport.reference as ref
    monkeypatch.setattr(ref, "collect_env", lambda root=None: {
        "gpu": "sm_89", "driver": "595", "nvcc": "12.6", "host": "gcc",
        "repo_commit": "123", "date": "2026-11-16", "root": repo,
    })

    rc = cli.main(["--root", str(repo), "refdrop", "--tag", "tag1"])
    assert rc == 1
    assert "silently-wrong" in capsys.readouterr().out


def test_refcompare_cli(tmp_path, capsys):
    import cuport.__main__ as cli
    from helpers import GOOD, make_probe
    from test_cli import FEATURES_YAML

    repo = tmp_path / "repo"
    (repo / "matrix").mkdir(parents=True)
    (repo / "matrix" / "features.yaml").write_text(FEATURES_YAML)
    probe_dir = make_probe(repo / "probes", stdout=GOOD)

    drop_a = probe_dir / "reference" / "tagA"
    drop_b = probe_dir / "reference" / "tagB"

    # None compared -> exit 2
    rc_none = cli.main(["--root", str(repo), "refcompare", "--a", "tagA", "--b", "tagB"])
    assert rc_none == 2

    # Both drops exist and match -> exit 0
    _make_fake_drop(drop_a, stdout=GOOD)
    _make_fake_drop(drop_b, stdout=GOOD)
    capsys.readouterr()

    rc_agree = cli.main(["--root", str(repo), "refcompare", "--a", "tagA", "--b", "tagB"])
    assert rc_agree == 0
    out_agree = capsys.readouterr().out
    assert "agree" in out_agree
    assert "1 compared, 0 differ" in out_agree

    # One drop differs -> exit 1
    _make_fake_drop(drop_b, stdout="CUPORT a u32 2 00000001 00000099\nCUPORT-DONE\n")
    rc_differ = cli.main(["--root", str(repo), "refcompare", "--a", "tagA", "--b", "tagB"])
    assert rc_differ == 1
    out_differ = capsys.readouterr().out
    assert "DIFFER" in out_differ
    assert "1 compared, 1 differ" in out_differ



