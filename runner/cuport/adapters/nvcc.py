"""Stock nvcc adapter for reference drops."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
from typing import Optional

import cuport.adapters as ad

EXTRA_FLAGS: dict[str, list[str]] = {
    "limits-refusals/dynamic-parallelism-001": ["-rdc=true", "-lcudadevrt"],
}

VSWHERE = pathlib.Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
    "Microsoft Visual Studio" / "Installer" / "vswhere.exe"


def msvc_bin() -> Optional[pathlib.Path]:
    """Directory holding the x64 cl.exe nvcc should use on Windows: $CUPORT_NVCC_CCBIN,
    else the newest Visual Studio install vswhere reports. None when not on Windows,
    or when cl.exe is already on PATH (a vcvars shell)."""
    if not ad.WINDOWS:
        return None
    if os.environ.get("CUPORT_NVCC_CCBIN"):
        return pathlib.Path(os.environ["CUPORT_NVCC_CCBIN"])
    if shutil.which("cl"):
        return None
    try:
        r = subprocess.run([str(VSWHERE), "-latest", "-products", "*", "-find",
                            r"VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"], capture_output=True, text=True)
    except OSError:
        return None
    found = sorted(l.strip() for l in r.stdout.splitlines() if l.strip())
    return pathlib.Path(found[-1]).parent if found else None


def host_flags() -> list[str]:
    """Host-compiler flags. With MSVC, /Zc:preprocessor is required by CCCL 13
    (cooperative_groups) and harmless everywhere else."""
    if not ad.WINDOWS:
        return []
    ccbin = msvc_bin()
    return [*(["-ccbin", str(ccbin)] if ccbin else []), "-Xcompiler", "/Zc:preprocessor"]


class Nvcc:
    id = "nvcc"
    version = "stock"

    def _nvcc(self) -> str:
        return os.environ.get("CUPORT_NVCC", "nvcc")

    def _arch(self) -> str:
        if "CUPORT_NVCC_ARCH" in os.environ:
            return os.environ["CUPORT_NVCC_ARCH"]
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            cap = r.stdout.strip().splitlines()[0].strip().replace(".", "")
            return f"sm_{cap}"
        except Exception as e:
            raise RuntimeError(f"failed to determine GPU compute capability via nvidia-smi: {e}")

    def available(self) -> tuple[bool, str]:
        nvcc = self._nvcc()
        try:
            r = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
            if r.returncode != 0:
                return False, f"{nvcc} failed: {r.stderr.strip()}"
            return True, r.stdout.strip().splitlines()[-1]
        except OSError as e:
            return False, f"{nvcc} is not available: {e}"

    def describe(self) -> dict:
        nvcc = self._nvcc()
        try:
            r = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
            ver = r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "stock"
        except Exception:
            ver = "stock"
        return {"id": self.id, "version": ver, "commit": "stock"}

    def compile(
        self, probe_dir: pathlib.Path, out_dir: pathlib.Path, timeout_s: float
    ) -> tuple[ad.StepResult, Optional[pathlib.Path]]:
        probe_id = f"{probe_dir.parent.name}/{probe_dir.name}"
        exe = out_dir / f"probe{ad.EXE_SUFFIX}"
        cmd = [
            self._nvcc(),
            *host_flags(),
            "-std=c++17",
            f"-arch={self._arch()}",
            "-I",
            str(ad.COMMON_DIR),
            *EXTRA_FLAGS.get(probe_id, []),
            "-o",
            str(exe),
            str(probe_dir / "probe.cu"),
        ]
        step = ad.execute(cmd, timeout_s)
        return step, (exe if step.returncode == 0 else None)

    def run(self, executable: pathlib.Path, timeout_s: float) -> ad.StepResult:
        return ad.execute([str(executable)], timeout_s)


ADAPTER = Nvcc()
