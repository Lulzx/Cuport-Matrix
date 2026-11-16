"""Load and validate probes (probes/README.md) and their oracles."""

from __future__ import annotations

import importlib.util
import pathlib
from dataclasses import dataclass, field

import yaml

from cuport.compare import parse_tolerance

ORACLE_CLASSES = ("spec", "cpu-reference", "reference-vector")
EXPECT = ("output", "refusal", "either")
_REQUIRED = ("id", "family", "features", "oracle", "tolerance")
_ALLOWED = set(_REQUIRED) | {"outputs", "expect", "timeout_s", "observe_only"}


class ProbeError(Exception):
    pass


@dataclass(frozen=True)
class Probe:
    id: str
    family: str
    dir: pathlib.Path
    features: tuple[str, ...]
    oracle: str
    tolerance: str
    outputs: dict = field(default_factory=dict)  # buffer name -> tolerance
    expect: str = "output"
    timeout_s: float = 60
    observe_only: bool = False

    def tolerance_for(self, buffer: str) -> str:
        return self.outputs.get(buffer, self.tolerance)


def load_features(path: pathlib.Path) -> dict[str, set[str]]:
    data = yaml.safe_load(path.read_text())
    return {f["id"]: set(f["features"]) for f in data["families"]}


def load_probe(d: pathlib.Path, features: dict[str, set[str]]) -> Probe:
    where = f"{d}"
    for required in ("probe.cu", "probe.yaml", "oracle.py"):
        if not (d / required).is_file():
            raise ProbeError(f"{where}: missing {required}")
    meta = yaml.safe_load((d / "probe.yaml").read_text()) or {}
    errors = [f"missing key {k}" for k in _REQUIRED if k not in meta]
    errors += [f"unknown key {k}" for k in sorted(set(meta) - _ALLOWED)]
    if errors:
        raise ProbeError(f"{where}: " + "; ".join(errors))

    family = meta["family"]
    if meta["id"] != f"{d.parent.name}/{d.name}" or family != d.parent.name:
        errors.append(f"id {meta['id']!r} / family {family!r} do not match directory {d.parent.name}/{d.name}")
    if family not in features:
        errors.append(f"unknown family {family!r}")
    feats = meta["features"]
    if not isinstance(feats, list) or not feats:
        errors.append("features must be a non-empty list")
        feats = []
    for f in feats:
        if f not in features.get(family, set()):
            errors.append(f"unknown feature {family}.{f}")
    if meta["oracle"] not in ORACLE_CLASSES:
        errors.append(f"oracle must be one of {ORACLE_CLASSES}")
    outputs = {}
    for name, spec in (meta.get("outputs") or {}).items():
        if not isinstance(spec, dict) or set(spec) != {"tolerance"}:
            errors.append(f"outputs.{name} must be {{tolerance: ...}}")
        else:
            outputs[name] = spec["tolerance"]
    for tol in [meta["tolerance"], *outputs.values()]:
        try:
            parse_tolerance(str(tol))
        except ValueError as e:
            errors.append(f"tolerance: {e}")
    expect = meta.get("expect", "output")
    if expect not in EXPECT:
        errors.append(f"expect must be one of {EXPECT}")
    timeout = meta.get("timeout_s", 60)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        errors.append("timeout_s must be a positive number")
    observe = meta.get("observe_only", False)
    if not isinstance(observe, bool):
        errors.append("observe_only must be true or false")
    if errors:
        raise ProbeError(f"{where}: " + "; ".join(errors))
    return Probe(meta["id"], family, d, tuple(feats), meta["oracle"], str(meta["tolerance"]),
                 outputs, expect, float(timeout), observe)


def discover(root: pathlib.Path, features: dict[str, set[str]]) -> tuple[list[Probe], list[str]]:
    probes, errors = [], []
    for yaml_path in sorted(root.glob("*/*/probe.yaml")):
        if yaml_path.parent.parent.name == "common":
            continue
        try:
            probes.append(load_probe(yaml_path.parent, features))
        except ProbeError as e:
            errors.append(str(e))
    for d in sorted(root.glob("*/*/")):
        if d.parent.name != "common" and d.is_dir() and not (d / "probe.yaml").exists():
            errors.append(f"{d}: missing probe.yaml")
    probes.sort(key=lambda p: p.id)
    return probes, errors


def load_oracle(probe: Probe):
    """Return (expected buffers, TEXT_LINES or None)."""
    name = "cuport_oracle_" + probe.id.replace("/", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, probe.dir / "oracle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.expected(), getattr(module, "TEXT_LINES", None)
