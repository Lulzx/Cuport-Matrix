"""mvcc v1.3.0 (doximity/mvcc): an nvcc-compatible driver.

Built from the pinned tag; the toolkit is installed at
$CUPORT_TOOLS/mvcc-v1.3.0 and the checkout kept at $CUPORT_TOOLS/src/mvcc."""

from __future__ import annotations

import subprocess

import cuport.adapters as ad

VERSION = "v1.3.0"
VERSION_MARKER = "mvcc_1.3.0"


class Mvcc:
    id = "mvcc"
    version = VERSION

    def _nvcc(self):
        return ad.TOOLS_ROOT / f"mvcc-{VERSION}" / "bin" / "nvcc"

    def available(self):
        nvcc = self._nvcc()
        if not nvcc.exists():
            return False, f"not installed at {nvcc}"
        out = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True).stdout
        if VERSION_MARKER not in out:
            return False, f"{nvcc} is not {VERSION}: {out.strip()[-80:]}"
        return True, out.strip().splitlines()[-1]

    def describe(self):
        src = ad.TOOLS_ROOT / "src" / "mvcc"
        r = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
        return {"id": self.id, "version": VERSION, "commit": r.stdout.strip() or "unknown"}

    def compile(self, probe_dir, out_dir, timeout_s):
        exe = out_dir / "probe"
        step = ad.execute([str(self._nvcc()), "-std=c++17", "-I", str(ad.COMMON_DIR), "-o", str(exe),
                           str(probe_dir / "probe.cu")], timeout_s)
        return step, (exe if step.returncode == 0 else None)

    def run(self, executable, timeout_s):
        return ad.execute([str(executable)], timeout_s)


ADAPTER = Mvcc()
