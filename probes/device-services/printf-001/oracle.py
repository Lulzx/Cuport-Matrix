# CUDA C++ Programming Guide, "Formatted Output": device printf supports the
# C printf conversions used here; output is flushed to the host stream at
# cudaDeviceSynchronize. Lines from different threads arrive in no defined
# order, so the runner compares them as a multiset.
from cuport_oracle import Buffer, lcg_u32

ID = "device-services/printf-001"
BLOCKS, THREADS = 4, 16


def _line(b, t, x):
    s = (x >> 8) - 0x800000
    f = (x & 0x3FFF) / 8.0
    w = "odd" if x & 1 else "even"
    return f"b{b} t{t} d={s} u={x} x={x:08x} f={f:.3f} ll={x * -3} s={w}%"


_inputs = lcg_u32(ID, BLOCKS * THREADS)
TEXT_LINES = [_line(b, t, _inputs[b * THREADS + t]) for b in range(BLOCKS) for t in range(THREADS)]


def expected():
    return {"lines": Buffer("u32", [BLOCKS * THREADS])}
