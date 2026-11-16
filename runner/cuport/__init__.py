"""Cuport-Matrix probe runner."""

import pathlib
import sys

_COMMON = str(pathlib.Path(__file__).resolve().parents[2] / "probes" / "common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)  # cuport_oracle, shared with probe oracles
