# CUDA C++ Programming Guide, "Streams": a stream is a sequence of commands that
# execute in order. 64 dependent kernel launches into one stream each map every
# element x of a 65536-element u32 buffer to (x ^ (k * 0x9E3779B9)) * 0x01000193
# (mod 2^32), k = launch index. The step does not commute across k, so launches
# that ran out of order or overlapped would change the result.
from __future__ import annotations

from cuport_oracle import Buffer, lcg_u32

ID = "streams-events/stream-ordering-001"
N = 65536
LAUNCHES = 64
M = 0xFFFFFFFF


def _chain(x: int) -> int:
    for k in range(LAUNCHES):
        x = ((x ^ ((k * 0x9E3779B9) & M)) * 0x01000193) & M
    return x


def expected() -> dict[str, Buffer]:
    return {"out": Buffer("u32", [_chain(x) for x in lcg_u32(ID, N)])}
