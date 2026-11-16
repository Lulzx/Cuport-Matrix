import json

from cuport.rows import CLASSES, build_row, validate_row
from helpers import FEATURES


def _row():
    cell = {"feature": "warp-primitives.shfl_xor_width", "probe": "warp-primitives/shfl-xor-001",
            "class": "bit-exact", "oracle": "cpu-reference", "tolerance": "bit-exact",
            "evidence": "[reproduced]", "upstream": None, "notes": ""}
    return build_row({"id": "t", "version": "v1", "commit": "abc"},
                     {"id": "m6-12c", "macos": "27.2 (26B123)", "profiler": "Apple M6"},
                     "2026-11-16", {"command": "x", "duration_s": 1.0}, [cell])


def test_valid_row():
    assert validate_row(_row(), FEATURES) == []
    assert "not-run" in CLASSES and len(CLASSES) == 8


def test_bad_cells_reported_individually():
    bad = json.loads(json.dumps(_row()))
    bad["cells"][0]["class"] = "fine"
    bad["cells"][0]["feature"] = "warp-primitives.nope"
    bad["cells"][0]["evidence"] = "[self-reported]"
    assert len(validate_row(bad, FEATURES)) == 3


def test_undocumented_divergence_needs_upstream():
    und = _row()
    und["cells"][0]["class"] = "undocumented-divergence"
    assert any("upstream" in e for e in validate_row(und, FEATURES))


def test_bad_header():
    row = _row()
    row["date"] = "16/11/2026"
    del row["tool"]["commit"]
    assert len(validate_row(row, FEATURES)) == 2
