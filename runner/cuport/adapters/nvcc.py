"""Stock nvcc adapter for reference drops."""

from __future__ import annotations

import os
import pathlib
import subprocess
from typing import Optional

import cuport.adapters as ad

EXTRA_FLAGS: dict[str, list[str]] = {
    "limits-refusals/dynamic-parallelism-001": ["-rdc=true", "-lcudadevrt"],
}


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
        exe = out_dir / "probe"
        cmd = [
            self._nvcc(),
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
