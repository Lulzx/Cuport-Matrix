"""Reference drops capture and comparison against real NVIDIA hardware."""

from __future__ import annotations

import datetime
import os
import pathlib
import platform
import re
import shlex
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from cuport.adapters import StepResult
from cuport.protocol import parse_output

ROOT = pathlib.Path(__file__).resolve().parents[2]
SEED_RULE = "per probe: x0 = FNV-1a 64 of the probe id; LCG and draw rule in matrix/schema.md"


@dataclass
class DropComparison:
    agree: bool
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)



def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _msvc_host() -> str:
    """'MSVC 14.44.35207 (cl 19.44.35207), Windows 10.0.26300' from the cl.exe nvcc will use."""
    from cuport.adapters.nvcc import msvc_bin

    ccbin = msvc_bin()
    cl = str(ccbin / "cl.exe") if ccbin else shutil.which("cl")
    if not cl:
        return ""
    toolset = re.search(r"MSVC[\\/](\d+\.\d+\.\d+)", cl)
    try:
        banner = subprocess.run([cl], capture_output=True, text=True, timeout=60).stderr
    except (OSError, subprocess.SubprocessError):
        banner = ""
    ver = re.search(r"Version (\d+\.\d+\.\d+)", banner)
    parts = [f"MSVC {toolset.group(1)}" if toolset else "MSVC"]
    if ver:
        parts.append(f"(cl {ver.group(1)})")
    return " ".join(parts) + f", Windows {platform.version()}"


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

    # 4. Host compiler: MSVC on Windows (nvcc's only supported host compiler there), gcc elsewhere
    host = _msvc_host() if os.name == "nt" else ""
    if not host:
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
            return p.relative_to(root).as_posix()
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

    # LF on every platform, so drops written on Windows compare byte-for-byte.
    (drop_dir / "output.txt").write_text(out_text, newline="\n")
    (drop_dir / "stderr.txt").write_text(err_text, newline="\n")
    (drop_dir / "build.log").write_text(build_log, newline="\n")
    (drop_dir / "meta.txt").write_text("\n".join(meta_lines) + "\n", newline="\n")


def _read_meta(drop_dir: pathlib.Path) -> dict[str, str]:
    meta_path = drop_dir / "meta.txt"
    if not meta_path.is_file():
        return {}
    res = {}
    for line in meta_path.read_text().splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            res[k.strip()] = v.strip()
    return res


def compare_drops(a: pathlib.Path, b: pathlib.Path) -> DropComparison:
    problems: list[str] = []
    notes: list[str] = []

    meta_a = _read_meta(a)
    meta_b = _read_meta(b)

    # 1. build exit equal
    build_exit_a = meta_a.get("build exit")
    build_exit_b = meta_b.get("build exit")
    if build_exit_a != build_exit_b:
        problems.append(f"build exit differs: {build_exit_a} vs {build_exit_b}")

    # 2. run exit equal
    run_exit_a = meta_a.get("run exit")
    run_exit_b = meta_b.get("run exit")
    if run_exit_a != run_exit_b:
        problems.append(f"run exit differs: {run_exit_a} vs {run_exit_b}")

    # 3. parse stdout
    out_a = (a / "output.txt").read_text() if (a / "output.txt").is_file() else ""
    out_b = (b / "output.txt").read_text() if (b / "output.txt").is_file() else ""
    pa = parse_output(out_a)
    pb = parse_output(out_b)

    # 4. both or neither have CUPORT-DONE
    if pa.done != pb.done:
        problems.append(f"CUPORT-DONE differs: {pa.done} vs {pb.done}")

    # 5. identical CUPORT-ERROR lists
    if pa.errors != pb.errors:
        problems.append(f"CUPORT-ERROR differs: {pa.errors} vs {pb.errors}")

    # 6. identical buffers (same names, dtypes, raw bits) via protocol.parse_output
    if set(pa.buffers) != set(pb.buffers):
        problems.append(f"buffer names differ: {sorted(pa.buffers)} vs {sorted(pb.buffers)}")
    for name in sorted(set(pa.buffers) & set(pb.buffers)):
        dtype_a, bits_a = pa.buffers[name]
        dtype_b, bits_b = pb.buffers[name]
        if dtype_a != dtype_b:
            problems.append(f"buffer {name} dtype differs: {dtype_a} vs {dtype_b}")
        elif bits_a != bits_b:
            problems.append(f"buffer {name} bits differ")

    # 7. printf text equal as a multiset after dropping lines starting with "timing "
    text_a = [line for line in pa.text if not line.startswith("timing ")]
    text_b = [line for line in pb.text if not line.startswith("timing ")]
    if Counter(text_a) != Counter(text_b):
        problems.append("printf text differs")

    # 8. stderr differences go to notes, never problems
    err_a = (a / "stderr.txt").read_text() if (a / "stderr.txt").is_file() else ""
    err_b = (b / "stderr.txt").read_text() if (b / "stderr.txt").is_file() else ""
    if err_a != err_b:
        notes.append("stderr differs")

    agree = len(problems) == 0
    return DropComparison(agree=agree, problems=problems, notes=notes)

