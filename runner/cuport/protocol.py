"""Parse a probe's stdout (protocol in probes/README.md)."""

from __future__ import annotations

from dataclasses import dataclass, field

from cuport_oracle import DTYPE_HEX_WIDTH


@dataclass
class ParsedOutput:
    buffers: dict[str, tuple[str, list[int]]] = field(default_factory=dict)  # name -> (dtype, raw bits)
    text: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)
    done: bool = False


def _parse_buffer(line: str):
    parts = line.split()
    if len(parts) < 4:
        return None
    _, name, dtype, count = parts[:4]
    if dtype not in DTYPE_HEX_WIDTH or not count.isdigit():
        return None
    words = parts[4:]
    if len(words) != int(count):
        return None
    width = DTYPE_HEX_WIDTH[dtype]
    values = []
    for w in words:
        if len(w) > width:
            return None
        try:
            values.append(int(w, 16))
        except ValueError:
            return None
    return name, dtype, values


def parse_output(stdout: str) -> ParsedOutput:
    p = ParsedOutput()
    for raw in stdout.splitlines():
        line = raw.rstrip("\r")
        if line == "CUPORT-DONE":
            p.done = True
        elif line.startswith("CUPORT-ERROR "):
            p.errors.append(line[len("CUPORT-ERROR "):])
        elif line.startswith("CUPORT "):
            parsed = _parse_buffer(line)
            if parsed is None or parsed[0] in p.buffers:
                p.malformed.append(line)
            else:
                name, dtype, values = parsed
                p.buffers[name] = (dtype, values)
        else:
            p.text.append(line)
    return p
