"""Reference drops capture and comparison against real NVIDIA hardware."""

from __future__ import annotations

import datetime
import os
import pathlib
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Optional

from cuport.adapters import StepResult

ROOT = pathlib.Path(__file__).resolve().parents[2]
SEED_RULE = "per probe: x0 = FNV-1a 64 of the probe id; LCG and draw rule in matrix/schema.md"


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def collect_env(root: Optional[pathlib.Path] = None) -> dict:
    repo_root = root or ROOT
    gpu = ""
    driver = ""
    nvcc_line = ""
    host = ""

    # 1. GPU
    gpu_csv = _sh(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"])
    if gpu_csv:
        first_line = gpu_csv.splitlines()[0]
        if "," in first_line:
            parts = [p.strip() for p in first_line.split(",", 1)]
            name = parts[0]
            cap = parts[1].replace(".", "")
            gpu = f"{name} (sm_{cap})"

    # 2. Driver
    drv_csv = _sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    drv_ver = drv_csv.splitlines()[0].strip() if drv_csv else ""
    smi_out = _sh(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*(\S+)", smi_out)
    cuda_ver = m.group(1) if m else ""
    if drv_ver and cuda_ver:
        driver = f"{drv_ver} (nvidia-smi reports CUDA {cuda_ver})"
    elif drv_ver:
        driver = drv_ver

    # 3. nvcc
    nvcc_bin = os.environ.get("CUPORT_NVCC", "nvcc")
    nvcc_out = _sh([nvcc_bin, "--version"])
    for line in nvcc_out.splitlines():
        if line.strip().startswith("Cuda compilation tools, release"):
            nvcc_line = line.strip()
            break

    # 4. Host compiler
    gcc_out = _sh(["gcc", "--version"])
    if gcc_out:
        host = gcc_out.splitlines()[0].strip()

    # 5. Repo commit & date
    commit = _sh(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    date = datetime.datetime.now(datetime.timezone.utc).date().isoformat()

    return {
        "gpu": gpu,
        "driver": driver,
        "nvcc": nvcc_line,
        "host": host,
        "repo_commit": commit,
        "date": date,
        "root": repo_root,
    }


def _make_relative(arg: str, root: pathlib.Path) -> str:
    try:
        p = pathlib.Path(arg)
        if p.is_absolute() and p.is_relative_to(root):
            return str(p.relative_to(root))
    except (ValueError, TypeError):
        pass
    return arg


def write_drop(
    drop_dir: pathlib.Path,
    probe_id: str,
    compile_step: StepResult,
    run_step: Optional[StepResult],
    env: dict,
) -> None:
    drop_dir.mkdir(parents=True, exist_ok=True)

    repo_root = env.get("root") or ROOT
    cmd_rel = [_make_relative(arg, repo_root) for arg in compile_step.cmd]
    cmd_str = shlex.join(cmd_rel)

    run_exit_str = "not run" if run_step is None else str(run_step.returncode)
    date_str = str(env.get("date") or datetime.datetime.now(datetime.timezone.utc).date().isoformat())
    commit_str = str(env.get("repo_commit") or env.get("commit") or _sh(["git", "-C", str(repo_root), "rev-parse", "HEAD"]))
    host_str = str(env.get("host") or env.get("host_compiler") or "")

    meta_lines = [
        f"probe: {probe_id}",
        f"gpu: {env.get('gpu', '')}",
        f"driver: {env.get('driver', '')}",
        f"nvcc: {env.get('nvcc', '')}",
        f"host compiler: {host_str}",
        f"command: {cmd_str}",
        f"build exit: {compile_step.returncode}",
        f"run exit: {run_exit_str}",
        f"date: {date_str}",
        f"repo commit: {commit_str}",
        f"seed: {SEED_RULE}",
    ]

    out_text = run_step.stdout if run_step else ""
    err_text = run_step.stderr if run_step else ""
    build_log = (compile_step.stdout or "") + (compile_step.stderr or "")

    (drop_dir / "output.txt").write_text(out_text)
    (drop_dir / "stderr.txt").write_text(err_text)
    (drop_dir / "build.log").write_text(build_log)
    (drop_dir / "meta.txt").write_text("\n".join(meta_lines) + "\n")
