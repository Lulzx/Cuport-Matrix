"""Command line: PYTHONPATH=runner python -m cuport <command>.

    list                      list probes
    check                     validate every probe.yaml and evaluate every oracle
    run --tool T --hardware H write results/<T>/<H>/<date>.json
    validate FILE...          validate result rows
    refdrop --tag TAG         capture reference drops using stock nvcc
    refcompare --a A --b B    compare two sets of reference drops
    scoreboard [--check]      write docs/scoreboard.md from results/ (--check: fail if stale)
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
from cuport.classify import classify
from cuport.probe import discover, load_features, load_oracle
from cuport.protocol import parse_output
from cuport.reference import collect_env, compare_drops, write_drop
from cuport.rows import build_row, validate_row
from cuport.run import COMPILE_TIMEOUT_S, run_matrix


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


def cmd_refdrop(args) -> int:
    features, probes, errors = _load(args.root, args.probe)
    if errors:
        for e in errors:
            print("ERROR", e)
        return 1
    if not probes:
        print("no probes selected")
        return 1

    for p in probes:
        drop_dir = p.dir / "reference" / args.tag
        if drop_dir.exists() and not args.force:
            print(f"{drop_dir} exists; pass --force to replace it")
            return 2

    adapter = load_adapter("nvcc")
    ok, detail = adapter.available()
    if not ok:
        print(f"nvcc is not available: {detail}")
        return 2

    env = collect_env(args.root)
    build_root = args.root / "build" / "nvcc"
    any_bad = False

    for p in probes:
        out_dir = build_root / p.id
        out_dir.mkdir(parents=True, exist_ok=True)
        step, exe = adapter.compile(p.dir, out_dir, COMPILE_TIMEOUT_S)
        if exe is not None and step.returncode == 0:
            run_step = adapter.run(exe, p.timeout_s)
        else:
            run_step = None

        drop_dir = p.dir / "reference" / args.tag
        write_drop(drop_dir, p.id, step, run_step, env)

        expected, text_lines = load_oracle(p)
        parsed = parse_output(run_step.stdout) if run_step else None
        cell = classify(p, step, exe, run_step, parsed, expected, text_lines)
        print(f"{p.id:45} {cell.klass:24} {cell.notes}".rstrip())

        accepted = False
        if cell.klass in ("bit-exact", "within-tolerance"):
            accepted = True
        elif p.expect in ("refusal", "either") and cell.klass == "refused-with-diagnostic":
            accepted = True
        if not accepted:
            any_bad = True

    return 1 if any_bad else 0


def cmd_refcompare(args) -> int:
    features, probes, errors = _load(args.root, args.probe)
    if errors:
        for e in errors:
            print("ERROR", e)
        return 1

    compared = 0
    differ = 0
    for p in probes:
        drop_a = p.dir / "reference" / args.a
        drop_b = p.dir / "reference" / args.b
        if drop_a.is_dir() and drop_b.is_dir():
            compared += 1
            cmp = compare_drops(drop_a, drop_b)
            if cmp.agree:
                print(f"{p.id} agree")
            else:
                differ += 1
                probs = "; ".join(cmp.problems)
                print(f"{p.id} DIFFER {probs}")

    print(f"{compared} compared, {differ} differ")
    if compared == 0:
        return 2
    return 1 if differ > 0 else 0


def cmd_scoreboard(args) -> int:
    from cuport.scoreboard import build

    _, probes, errors = _load(args.root)
    if errors:
        for e in errors:
            print("ERROR", e)
        return 1
    out = args.out or args.root / "docs" / "scoreboard.md"
    text = build(args.root, [p.id for p in probes])
    if args.check:
        current = out.read_text(encoding="utf-8") if out.exists() else ""  # universal newlines: CRLF compares equal
        if current != text:
            print(f"{out} is stale; run: PYTHONPATH=runner python -m cuport scoreboard")
            return 1
        print(f"{out} is up to date")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return 0


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
    rd = sub.add_parser("refdrop")
    rd.add_argument("--tag", required=True)
    rd.add_argument("--probe", action="append", help="probe id prefix; repeatable")
    rd.add_argument("--force", action="store_true")
    rd.set_defaults(fn=cmd_refdrop)
    rc = sub.add_parser("refcompare")
    rc.add_argument("--a", required=True)
    rc.add_argument("--b", required=True)
    rc.add_argument("--probe", action="append", help="probe id prefix; repeatable")
    rc.set_defaults(fn=cmd_refcompare)
    sb = sub.add_parser("scoreboard")
    sb.add_argument("--out", type=pathlib.Path)
    sb.add_argument("--check", action="store_true", help="exit 1 if the file differs from a fresh build")
    sb.set_defaults(fn=cmd_scoreboard)
    args = ap.parse_args(argv)
    args.argv = argv
    return args.fn(args)



if __name__ == "__main__":
    sys.exit(main())
