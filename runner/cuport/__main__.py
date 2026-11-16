"""Command line: PYTHONPATH=runner python -m cuport <command>.

    list                      list probes
    check                     validate every probe.yaml and evaluate every oracle
    run --tool T --hardware H write results/<T>/<H>/<date>.json
    validate FILE...          validate result rows
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import pathlib
import platform
import shlex
import subprocess
import sys
import time

import yaml

from cuport.adapters import load_adapter
from cuport.probe import discover, load_features, load_oracle
from cuport.rows import build_row, validate_row
from cuport.run import run_matrix

ROOT = pathlib.Path(__file__).resolve().parents[2]
SEED_RULE = "per probe: x0 = FNV-1a 64 of the probe id; LCG and draw rule in matrix/schema.md"


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def collect_hardware(hw_id: str) -> dict:
    macos = f"{_sh(['sw_vers', '-productVersion'])} ({_sh(['sw_vers', '-buildVersion'])})"
    keep = ("Chipset Model", "Total Number of Cores", "Metal Support")
    lines = [l.strip() for l in _sh(["system_profiler", "SPDisplaysDataType"]).splitlines()
             if l.strip().startswith(keep)]
    return {"id": hw_id, "macos": macos, "profiler": "; ".join(lines)}


def _repo_state(root: pathlib.Path) -> dict:
    commit = _sh(["git", "-C", str(root), "rev-parse", "HEAD"])
    dirty = bool(_sh(["git", "-C", str(root), "status", "--porcelain", "--", "probes", "runner", "matrix"]))
    return {"repo_commit": commit, "repo_dirty": dirty}


def _load(root: pathlib.Path, prefixes: list[str] | None = None):
    features = load_features(root / "matrix" / "features.yaml")
    probes, errors = discover(root / "probes", features)
    if prefixes:
        probes = [p for p in probes if any(p.id.startswith(x) for x in prefixes)]
    return features, probes, errors


def cmd_list(args) -> int:
    _, probes, errors = _load(args.root)
    for p in probes:
        print(f"{p.id:45} {','.join(p.features):30} {p.oracle:16} {p.tolerance}")
    for e in errors:
        print("ERROR", e)
    return 1 if errors else 0


STUB_DIR = pathlib.Path(__file__).resolve().parents[1] / "cuda_stub"


def syntax_errors(probe) -> list[str]:
    """Parse probe.cu with clang's CUDA mode against runner/cuda_stub, host side
    and device side. Catches typos and missing declarations; runs nothing."""
    base = ["clang++", "-x", "cuda", "-std=c++17", "-nocudainc", "-nocudalib", "--cuda-gpu-arch=sm_80",
            "-fsyntax-only", "-Wno-unknown-cuda-version", "-I", str(STUB_DIR), str(probe.dir / "probe.cu")]
    out = []
    for side in ("--cuda-host-only", "--cuda-device-only"):
        r = subprocess.run(base + [side], capture_output=True, text=True)
        if r.returncode != 0:
            first = [l for l in r.stderr.splitlines() if "error:" in l][:3]
            out.append(f"{probe.id}: syntax ({side[7:]}): " + " | ".join(first or [r.stderr.strip()[:300]]))
    return out


def cmd_check(args) -> int:
    _, probes, errors = _load(args.root)
    if args.syntax:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(os.cpu_count()) as pool:
            for errs in pool.map(syntax_errors, probes):
                errors.extend(errs)
    for p in probes:
        try:
            expected, _ = load_oracle(p)
            for name, buf in expected.items():
                if len(buf.values) == 0:
                    errors.append(f"{p.id}: oracle buffer {name} is empty")
        except Exception as e:  # an oracle is user code; report any failure
            errors.append(f"{p.id}: oracle failed: {type(e).__name__}: {e}")
    for e in errors:
        print("ERROR", e)
    print(f"{len(probes)} probes, {len(errors)} errors")
    return 1 if errors else 0


def cmd_run(args) -> int:
    features, probes, errors = _load(args.root, args.probe)
    if errors:
        for e in errors:
            print("ERROR", e)
        return 1
    if not probes:
        print("no probes selected")
        return 1
    out = args.root / "results" / args.tool / args.hardware / f"{args.date}.json"
    if out.exists() and not args.force:
        print(f"{out} exists; pass --force to replace it")
        return 2
    adapter = load_adapter(args.tool)
    ok, detail = adapter.available()
    if not ok:
        print(f"{args.tool} is not available: {detail}")
        return 2
    triage_path = args.root / "matrix" / "triage.yaml"
    triage = (yaml.safe_load(triage_path.read_text()) or {}).get("entries", []) if triage_path.exists() else []

    start = time.monotonic()
    done = [0]

    def progress(probe, cell):
        done[0] += 1
        print(f"[{done[0]:>3}/{len(probes)}] {probe.id:45} {cell.klass:24} {cell.notes[:80]}", flush=True)

    cells = run_matrix(adapter, probes, args.root / "build" / args.tool, args.jobs, args.gpu_jobs, triage, progress)
    runner = {
        "command": "python -m cuport " + shlex.join(args.argv),
        "duration_s": round(time.monotonic() - start, 1),
        "config": {"jobs": args.jobs, "gpu_jobs": args.gpu_jobs, "probe_filter": args.probe or [],
                   "probes": len(probes), "seed": SEED_RULE, "python": platform.python_version(),
                   "tool_detail": detail, **_repo_state(args.root)},
    }
    row = build_row(adapter.describe(), collect_hardware(args.hardware), args.date, runner, cells)
    problems = validate_row(row, features)
    if problems:
        for p in problems:
            print("INVALID", p)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2) + "\n")
    counts = collections.Counter(c["class"] for c in cells)
    print(f"wrote {out} in {runner['duration_s']}s: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 0


def cmd_validate(args) -> int:
    features = load_features(args.root / "matrix" / "features.yaml")
    bad = 0
    for f in args.files:
        problems = validate_row(json.loads(pathlib.Path(f).read_text()), features)
        for p in problems:
            print(f"{f}: {p}")
        bad += bool(problems)
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="cuport")
    ap.add_argument("--root", type=pathlib.Path, default=ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    c = sub.add_parser("check")
    c.add_argument("--syntax", action="store_true", help="also parse probe.cu with clang against runner/cuda_stub")
    c.set_defaults(fn=cmd_check)
    r = sub.add_parser("run")
    r.add_argument("--tool", required=True)
    r.add_argument("--hardware", required=True)
    r.add_argument("--probe", action="append", help="probe id prefix; repeatable")
    r.add_argument("--jobs", type=int, default=os.cpu_count())
    r.add_argument("--gpu-jobs", type=int, default=4)
    r.add_argument("--date", default=datetime.datetime.now(datetime.timezone.utc).date().isoformat())
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_run)
    v = sub.add_parser("validate")
    v.add_argument("files", nargs="+")
    v.set_defaults(fn=cmd_validate)
    args = ap.parse_args(argv)
    args.argv = argv
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
