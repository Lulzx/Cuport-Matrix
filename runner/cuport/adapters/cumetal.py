"""CuMetal v0.9.1 (Lulzx/cuda-metal): cumetalc compiles CUDA source to a
linked executable.

Built from the pinned tag and installed at $CUPORT_TOOLS/cumetal-v0.9.1.
cumetalc packages kernels into a metallib with Apple's offline Metal
compiler (`xcrun metal`), which ships with full Xcode, not the Command Line
Tools; without it the tool is reported unavailable rather than run."""

from __future__ import annotations

import subprocess

import cuport.adapters as ad

VERSION = "v0.9.1"


class CuMetal:
    id = "cumetal"
    version = VERSION

    def _cc(self):
        return ad.TOOLS_ROOT / f"cumetal-{VERSION}" / "bin" / "cumetalc"

    def available(self):
        cc = self._cc()
        if not cc.exists():
            return False, f"not installed at {cc}"
        if subprocess.run(["xcrun", "--find", "metal"], capture_output=True).returncode != 0:
            return False, "Metal toolchain missing: `xcrun --find metal` fails (needs full Xcode)"
        return True, str(cc)

    def describe(self):
        src = ad.TOOLS_ROOT / "src" / "cuda-metal"
        r = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
        return {"id": self.id, "version": VERSION, "commit": r.stdout.strip() or "unknown"}

    def compile(self, probe_dir, out_dir, timeout_s):
        exe = out_dir / "probe"
        step = ad.execute([str(self._cc()), str(probe_dir / "probe.cu"), "-I", str(ad.COMMON_DIR),
                           "-o", str(exe)], timeout_s)
        return step, (exe if step.returncode == 0 else None)

    def run(self, executable, timeout_s):
        return ad.execute([str(executable)], timeout_s)


ADAPTER = CuMetal()
