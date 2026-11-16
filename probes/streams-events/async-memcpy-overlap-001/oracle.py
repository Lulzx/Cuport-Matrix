# CUDA C++ Programming Guide, "Asynchronous Concurrent Execution":
# "Concurrent Data Transfers and Kernel Execution: Some devices capable of concurrent
# kernel execution can also perform a kernel execution concurrently with a host-to-device,
# device-to-host, or device-to-device data transfer."
# The probe pipelines 16 MiB of u32 (4,194,304 elements) in 8 chunks of 524,288 elements
# across 2 non-default streams (chunk % 2).
# For each chunk, y = x * 2654435761 + chunk (mod 2^32).
# The oracle produces:
#   count_wrong: 0 (all elements match)
#   sample: strided sample of 4096 elements (every 1024th element)
from __future__ import annotations

from cuport_oracle import Buffer, Lcg, _M64

ID = "streams-events/async-memcpy-overlap-001"
N = 4194304
CHUNK_ELEMS = 524288
SAMPLE_COUNT = 4096
STRIDE = 1024
MULTIPLIER = 2654435761

# LCG jump parameters for 1024 steps:
_A1024 = 0xAA1421B9D5CD9001
_C1024 = 0x6B00BEA8C50C1400
_LCG_MUL = 6364136223846793005
_LCG_INC = 1442695040888963407


def expected() -> dict[str, Buffer]:
    lcg = Lcg(ID)
    cur_x = lcg.x
    sample = []
    for j in range(SAMPLE_COUNT):
        chunk = (j * STRIDE) // CHUNK_ELEMS
        val_x = (_LCG_MUL * cur_x + _LCG_INC) & _M64
        val = val_x >> 32
        sample.append((val * MULTIPLIER + chunk) & 0xFFFFFFFF)
        cur_x = (_A1024 * cur_x + _C1024) & _M64

    return {
        "count_wrong": Buffer("u32", [0]),
        "sample": Buffer("u32", sample),
    }
