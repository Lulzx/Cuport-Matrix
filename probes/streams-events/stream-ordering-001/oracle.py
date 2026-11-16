# CUDA C++ Programming Guide, "Streams":
# "A stream is a sequence of commands (possibly issued by different host threads)
# that execute in order." Commands in the same stream are serialized and execute
# in issue order. 64 dependent kernel launches each add k * 0x9E3779B9 to every
# element of a 65536-element u32 buffer. With launch index k in [0, 63], the total
# added to each element is sum_{k=0}^{63} (k * 0x9E3779B9) = 2016 * 0x9E3779B9 = 0xF4DE90E0 (mod 2^32).
from __future__ import annotations

from cuport_oracle import Buffer, lcg_u32

ID = "streams-events/stream-ordering-001"
N = 65536
TOTAL_DELTA = sum((k * 0x9E3779B9) & 0xFFFFFFFF for k in range(64)) & 0xFFFFFFFF


def expected() -> dict[str, Buffer]:
    raw = lcg_u32(ID, N)
    out = [(x + TOTAL_DELTA) & 0xFFFFFFFF for x in raw]
    return {"out": Buffer("u32", out)}
