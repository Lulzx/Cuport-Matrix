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
