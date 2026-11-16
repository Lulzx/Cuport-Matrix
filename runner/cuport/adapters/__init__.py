"""Tool adapters. One module per tool id in matrix/tools.yaml
(cuport/adapters/<id>.py), each exposing a module-level ADAPTER object.

Tools are installed outside the repo, under
    $CUPORT_TOOLS (default ~/.local/cuport-tools)/<id>-<version>/
"""

from __future__ import annotations

import importlib
import os
import pathlib
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Protocol

TOOLS_ROOT = pathlib.Path(os.environ.get("CUPORT_TOOLS", pathlib.Path.home() / ".local" / "cuport-tools"))
COMMON_DIR = pathlib.Path(__file__).resolve().parents[3] / "probes" / "common"


@dataclass
class StepResult:
    cmd: list[str]
    returncode: Optional[int]  # None when timed out
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool


def execute(cmd: list[str], timeout_s: float, cwd: Optional[pathlib.Path] = None,
            env: Optional[dict] = None) -> StepResult:
    """Run cmd, capture output, kill the whole process group on timeout."""
    start = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, errors="replace", start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout_s)
        return StepResult(cmd, proc.returncode, out, err, time.monotonic() - start, False)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, 9)
        out, err = proc.communicate()
        return StepResult(cmd, None, out, err, time.monotonic() - start, True)


class Adapter(Protocol):
    id: str
    version: str

    def available(self) -> tuple[bool, str]:
        """(True, detail) if the pinned version is installed and runnable."""

    def describe(self) -> dict:
        """The result row's "tool" object: {"id", "version", "commit"}."""

    def compile(self, probe_dir: pathlib.Path, out_dir: pathlib.Path,
                timeout_s: float) -> tuple[StepResult, Optional[pathlib.Path]]:
        """Compile probe_dir/probe.cu with -I COMMON_DIR into out_dir.
        Returns the step and the executable path, or None if no executable."""

    def run(self, executable: pathlib.Path, timeout_s: float) -> StepResult:
        """Run the compiled probe with the tool's runtime environment."""


def load_adapter(tool_id: str) -> Adapter:
    module = importlib.import_module(f"cuport.adapters.{tool_id.replace('-', '_')}")
    return module.ADAPTER
