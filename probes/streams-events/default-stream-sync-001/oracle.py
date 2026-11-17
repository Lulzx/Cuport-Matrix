# CUDA C++ Programming Guide, "Default Stream" / Runtime API "Stream
# synchronization behavior": the legacy default stream "is an implicit stream
# which synchronizes with all other streams in the same CUcontext except for
# non-blocking streams". A kernel in stream 0 therefore waits for earlier work in
# a blocking stream, later blocking-stream work waits for it, and cudaMemcpy (in
# stream 0) waits for everything before it.
#
# x = mix(in, 4096), y = mix(x ^ 0x9E3779B9, 4096), z = y ^ 0xA5A5A5A5, where
# mix repeats x -> (x ^ (x >> 15)) * 0x2C1B3C6D + r (mod 2^32) for r = 0..rounds-1.
# A kernel that ran before its producer finished reads zeros or partial data.
import numpy as np

from cuport_oracle import Buffer, lcg_u32

ID = "streams-events/default-stream-sync-001"
N = 65536
ROUNDS = 4096


def mix(x: np.ndarray, rounds: int) -> np.ndarray:
    x = x.astype(np.uint32)
    mul = np.uint32(0x2C1B3C6D)
    for r in range(rounds):
        x = (x ^ (x >> np.uint32(15))) * mul + np.uint32(r)
    return x


def expected():
    inp = np.array(lcg_u32(ID, N), dtype=np.uint32)
    x = mix(inp, ROUNDS)
    y = mix(x ^ np.uint32(0x9E3779B9), ROUNDS)
    z = y ^ np.uint32(0xA5A5A5A5)
    return {"x": Buffer("u32", x.tolist()), "y": Buffer("u32", y.tolist()), "z": Buffer("u32", z.tolist())}
