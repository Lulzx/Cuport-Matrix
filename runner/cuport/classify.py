"""Turn one probe execution into a matrix class (matrix/schema.md)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Optional

from cuport.compare import compare_buffer

_NOTE_LIMIT = 300


@dataclass(frozen=True)
class Cell:
    klass: str
    notes: str
    reference: Optional[str] = None
    upstream: Optional[str] = None


def _first_lines(text: str) -> str:
    s = " | ".join(l.strip() for l in text.strip().splitlines() if l.strip())
    return s[:_NOTE_LIMIT]


def _classify(probe, compile_step, exe, run_step, parsed, expected, text_lines) -> Cell:
    if compile_step.timed_out:
        return Cell("crash-hang", f"compile timed out after {compile_step.duration_s:.0f}s")
    if exe is None or compile_step.returncode != 0:
        rc = compile_step.returncode
        if rc is not None and rc < 0:
            return Cell("crash-hang", f"compiler killed by signal {-rc}: {_first_lines(compile_step.stderr)}")
        diag = _first_lines(compile_step.stderr + "\n" + compile_step.stdout)
        if diag:
            return Cell("refused-with-diagnostic", f"compile: {diag}")
        return Cell("crash-hang", f"compiler exited {rc} without a diagnostic")

    if run_step.timed_out:
        return Cell("crash-hang", f"run timed out after {run_step.duration_s:.0f}s")
    if parsed.errors:
        return Cell("refused-with-diagnostic", f"runtime: {parsed.errors[0]}"[:_NOTE_LIMIT])
    rc = run_step.returncode
    if rc is not None and rc < 0:
        return Cell("crash-hang", f"killed by signal {-rc}: {_first_lines(run_step.stderr)}")
    if not parsed.done:
        return Cell("crash-hang", f"exit {rc} without CUPORT-DONE: {_first_lines(run_step.stderr)}")
    if rc != 0:
        return Cell("crash-hang", f"exit {rc} after CUPORT-DONE")
    if parsed.malformed:
        return Cell("silently-wrong", f"malformed protocol line: {parsed.malformed[0][:120]}")

    problems, within = [], []
    for name, buf in expected.items():
        if name not in parsed.buffers:
            problems.append(f"missing buffer {name}")
            continue
        dtype, bits = parsed.buffers[name]
        r = compare_buffer(buf, dtype, bits, probe.tolerance_for(name))
        if r.status == "mismatch":
            problems.append(f"{name}{r.detail if r.detail.startswith('[') else ': ' + r.detail}"
                            f" ({r.mismatches} wrong)")
        elif r.status == "within-tolerance":
            within.append(f"{name}: {r.detail}")
    for name in sorted(set(parsed.buffers) - set(expected)):
        problems.append(f"unexpected buffer {name}")
    if text_lines is not None and Counter(parsed.text) != Counter(text_lines):
        problems.append(f"printf lines differ: got {len(parsed.text)}, want {len(text_lines)}")
    if problems:
        return Cell("silently-wrong", "; ".join(problems)[:_NOTE_LIMIT])
    if within:
        return Cell("within-tolerance", "; ".join(within)[:_NOTE_LIMIT])
    return Cell("bit-exact", "")


def classify(probe, compile_step, exe, run_step, parsed, expected, text_lines) -> Cell:
    cell = _classify(probe, compile_step, exe, run_step, parsed, expected, text_lines)
    if probe.observe_only:
        cell = replace(cell, notes=("observe-only; " + cell.notes).rstrip("; "))
    return cell


def apply_triage(cell: Cell, tool: str, probe_id: str, entries: list[dict]) -> Cell:
    """matrix/triage.yaml entries reclassify a raw class after a human read the
    tool's docs. An entry applies only while the raw class is unchanged."""
    for e in entries:
        if e["tool"] == tool and e["probe"] == probe_id and e["raw_class"] == cell.klass:
            notes = f"raw {cell.klass}; {e.get('notes', '')}; {cell.notes}".strip("; ")
            return Cell(e["class"], notes[:_NOTE_LIMIT], e.get("reference"), e.get("upstream"))
    return cell
