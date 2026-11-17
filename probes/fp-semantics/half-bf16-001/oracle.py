# CUDA Math API, "Half Precision Conversion and Data Movement": __float2half_rn,
# _rz, _ru and _rd convert float to half "in round-to-nearest-even / round-towards-
# zero / round-up / round-down mode"; "Bfloat16 Precision Conversion and Data
# Movement": __float2bfloat16_rn and _rz likewise. These are IEEE 754 conversions:
# overflow gives infinity, or the largest finite value when the mode rounds toward
# zero. __half2float and __bfloat162float are exact.
import struct

from cuport_oracle import Buffer, Lcg, bits_f32, round_bits

ID = "fp-semantics/half-bf16-001"
N = 2048
EDGES = [
    0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x477FE000, 0x477FF000, 0x477FEFFF,
    0xC77FF000, 0x33800000, 0x33000000, 0x33400000, 0x387FC000, 0x38800000, 0x3F801000,
    0x3F803000, 0x7F7FFFFF, 0x7F7F8000, 0x00000001, 0x00008000, 0x00018000, 0x3F808000,
    0x3F818000, 0xFF7F8000, 0x80018000,
]


def inputs():
    g = Lcg(ID)
    x, z = [], []
    for i in range(N):
        s = g.next_u32() & 0x80000000
        e = 101 + g.next_u32() % 44
        m = g.next_u32() >> 9
        zb = g.next_u32()
        if zb & 0x7F800000 == 0x7F800000:
            zb &= 0xBFFFFFFF
        x.append(bits_f32(EDGES[i] if i < len(EDGES) else s | (e << 23) | m))
        z.append(bits_f32(EDGES[i] if i < len(EDGES) else zb))
    return x, z


def half_value(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def expected():
    x, z = inputs()
    out = {f"h_{m}": Buffer("u16", [round_bits(v, "f16", m) for v in x]) for m in ("rn", "rz", "ru", "rd")}
    out["h_back"] = Buffer("f32", [half_value(b) for b in out["h_rn"].values])
    out["bf_rn"] = Buffer("u16", [round_bits(v, "bf16", "rn") for v in z])
    out["bf_rz"] = Buffer("u16", [round_bits(v, "bf16", "rz") for v in z])
    out["bf_back"] = Buffer("f32", [bits_f32(b << 16) for b in out["bf_rn"].values])
    return out
