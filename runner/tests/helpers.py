import textwrap

from cuport.adapters import StepResult

FEATURES = {"warp-primitives": {"shfl_xor_width", "ballot_any_all"}, "atomics": {"atomicCAS_loop"}}
GOOD = "CUPORT a u32 2 00000001 00000002\nCUPORT-DONE\n"


def make_probe(root, pid="warp-primitives/shfl-xor-001", yaml_extra="", oracle=None, stdout=None,
               features="[shfl_xor_width]"):
    family, name = pid.split("/")
    d = root / family / name
    d.mkdir(parents=True)
    (d / "probe.cu").write_text("// probe\n")
    (d / "probe.yaml").write_text(textwrap.dedent(f"""\
        id: {pid}
        family: {family}
        features: {features}
        oracle: cpu-reference
        tolerance: bit-exact
        expect: output
        timeout_s: 5
        """) + yaml_extra)
    (d / "oracle.py").write_text(oracle or textwrap.dedent("""\
        from cuport_oracle import Buffer
        def expected():
            return {"a": Buffer("u32", [1, 2])}
        """))
    if stdout is not None:
        (d / "fake_stdout.txt").write_text(stdout)
    return d


def step(rc=0, out="", err="", timed_out=False):
    return StepResult(["x"], None if timed_out else rc, out, err, 0.01, timed_out)
