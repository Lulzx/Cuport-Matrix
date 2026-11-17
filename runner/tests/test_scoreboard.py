import json

import cuport.__main__ as cli
from cuport.scoreboard import build, disagreement, latest_rows
from helpers import make_probe


def write_row(root, tool, hw, date, classes, commit="abc1234def"):
    cells = [{"feature": f"warp-primitives.{p.split('/')[1]}", "probe": p, "class": c, "oracle": "spec",
              "tolerance": "bit-exact", "evidence": "[reproduced]",
              "upstream": "https://example.invalid/1" if c == "undocumented-divergence" else None, "notes": ""}
             for p, c in classes.items()]
    row = {"schema": 1, "tool": {"id": tool, "version": "v1", "commit": "x"},
           "hardware": {"id": hw, "macos": "27", "profiler": ""}, "date": date,
           "runner": {"command": "c", "duration_s": 1, "config": {"repo_commit": commit}}, "cells": cells}
    f = root / "results" / tool / hw / f"{date}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(row))


P1, P2, P3 = "w/p1", "w/p2", "w/p3"


def test_latest_row_wins_and_disagreement_rate(tmp_path):
    write_row(tmp_path, "mvcc", "m6-12c", "2026-11-01", {P1: "silently-wrong"})
    write_row(tmp_path, "mvcc", "m6-12c", "2026-11-16", {P1: "bit-exact", P2: "bit-exact", P3: "bit-exact"})
    write_row(tmp_path, "mvcc", "paravirtual", "2026-11-17",
              {P1: "bit-exact", P2: "refused-with-diagnostic", P3: "not-run"})
    write_row(tmp_path, "cumetal", "m6-12c", "2026-11-16", {P1: "bit-exact"})  # no hosted row
    rows = latest_rows(tmp_path / "results")
    assert [(r.tool, r.hardware, r.date) for r in rows] == [
        ("mvcc", "m6-12c", "2026-11-16"), ("mvcc", "paravirtual", "2026-11-17"), ("cumetal", "m6-12c", "2026-11-16")]
    [d] = disagreement(rows)
    assert d.tool == "mvcc" and d.compared == 2  # P3 is not-run on the hosted lane
    assert d.differ == [(P2, "refused-with-diagnostic", "bit-exact")] and d.rate == 0.5


def test_render_counts_missing_probes_as_not_run(tmp_path):
    write_row(tmp_path, "mvcc", "m6-12c", "2026-11-16", {P1: "undocumented-divergence"})
    text = build(tmp_path, [P1, P2])
    assert "| mvcc v1 | `m6-12c` |" in text
    assert "| 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |" in text  # one undocumented, one not-run
    assert "1 of 8 (12.5 %)" in text                   # 2 probes x 2 tools x 2 lanes
    assert "not measured: no `paravirtual` row yet" in text
    assert "https://example.invalid/1" in text


def test_check_detects_stale_file(tmp_path, capsys):
    (tmp_path / "matrix").mkdir()
    (tmp_path / "matrix" / "features.yaml").write_text(
        "families:\n  - id: warp-primitives\n    features: [shfl_xor_width]\n")
    make_probe(tmp_path / "probes")
    write_row(tmp_path, "mvcc", "m6-12c", "2026-11-16", {"warp-primitives/shfl-xor-001": "bit-exact"})
    args = ["--root", str(tmp_path), "scoreboard"]
    assert cli.main(args + ["--check"]) == 1
    assert cli.main(args) == 0
    assert cli.main(args + ["--check"]) == 0
    write_row(tmp_path, "mvcc", "m6-12c", "2026-11-17", {"warp-primitives/shfl-xor-001": "crash-hang"})
    assert cli.main(args + ["--check"]) == 1
    assert "is stale" in capsys.readouterr().out
