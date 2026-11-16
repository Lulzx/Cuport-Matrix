"""Build and validate result rows (matrix/schema.md)."""

from __future__ import annotations

import re

CLASSES = ("bit-exact", "within-tolerance", "documented-divergence", "undocumented-divergence",
           "refused-with-diagnostic", "silently-wrong", "crash-hang", "not-run")
_CELL_KEYS = ("feature", "probe", "class", "oracle", "tolerance", "evidence", "upstream", "notes")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def build_row(tool: dict, hardware: dict, date: str, runner: dict, cells: list[dict]) -> dict:
    return {"schema": 1, "tool": tool, "hardware": hardware, "date": date, "runner": runner,
            "cells": sorted(cells, key=lambda c: (c["probe"], c["feature"]))}


def validate_row(row: dict, features: dict[str, set[str]]) -> list[str]:
    errors = []
    if row.get("schema") != 1:
        errors.append("schema must be 1")
    for key, sub in (("tool", ("id", "version", "commit")), ("hardware", ("id", "macos", "profiler")),
                     ("runner", ("command", "duration_s"))):
        if not isinstance(row.get(key), dict) or any(k not in row[key] for k in sub):
            errors.append(f"{key} needs {sub}")
    if not _DATE.match(str(row.get("date", ""))):
        errors.append("date must be YYYY-MM-DD")
    for i, c in enumerate(row.get("cells", [])):
        where = f"cells[{i}] {c.get('probe', '?')}"
        missing = [k for k in _CELL_KEYS if k not in c]
        if missing:
            errors.append(f"{where}: missing {missing}")
            continue
        family, _, feat = c["feature"].partition(".")
        if feat not in features.get(family, set()):
            errors.append(f"{where}: unknown feature {c['feature']}")
        if c["class"] not in CLASSES:
            errors.append(f"{where}: unknown class {c['class']}")
        if c["evidence"] != "[reproduced]":
            errors.append(f"{where}: evidence must be [reproduced]")
        if c["class"] == "undocumented-divergence" and not c["upstream"]:
            errors.append(f"{where}: undocumented-divergence needs an upstream issue")
    return errors
