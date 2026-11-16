import pathlib

import cuport_oracle as co
from cuport.classify import Cell, apply_triage, classify
from cuport.probe import load_probe
from cuport.protocol import parse_output
from helpers import FEATURES, GOOD, make_probe, step

EXP = {"a": co.Buffer("u32", [1, 2])}


def cls(tmp_path, compile_step=None, exe=True, run_step=None, expected=EXP, text=None, yaml_extra=""):
    p = load_probe(make_probe(tmp_path, yaml_extra=yaml_extra), FEATURES)
    run_step = run_step if run_step is not None else step(out=GOOD)
    return classify(p, compile_step or step(), pathlib.Path("e") if exe else None, run_step,
                    parse_output(run_step.stdout), expected, text)


def test_classify_bit_exact(tmp_path):
    assert cls(tmp_path).klass == "bit-exact"


def test_classify_compile_timeout(tmp_path):
    assert cls(tmp_path, compile_step=step(timed_out=True), exe=False).klass == "crash-hang"


def test_classify_compile_refusal_with_message(tmp_path):
    c = cls(tmp_path, compile_step=step(rc=1, err="error: __match_any_sync is not supported\n"), exe=False)
    assert c.klass == "refused-with-diagnostic" and "__match_any_sync" in c.notes


def test_classify_compile_failure_without_message(tmp_path):
    assert cls(tmp_path, compile_step=step(rc=1), exe=False).klass == "crash-hang"


def test_classify_compiler_signal(tmp_path):
    assert cls(tmp_path, compile_step=step(rc=-11, err="Segmentation fault"), exe=False).klass == "crash-hang"


def test_classify_run_timeout(tmp_path):
    assert cls(tmp_path, run_step=step(timed_out=True)).klass == "crash-hang"


def test_classify_runtime_error_is_refusal(tmp_path):
    c = cls(tmp_path, run_step=step(rc=3, out="CUPORT-ERROR cudaLaunchKernel 98 invalid device function\n"))
    assert c.klass == "refused-with-diagnostic" and "invalid device function" in c.notes


def test_classify_missing_done_is_crash(tmp_path):
    assert cls(tmp_path, run_step=step(rc=0, out="CUPORT a u32 2 00000001 00000002\n")).klass == "crash-hang"


def test_classify_nonzero_exit_after_done_is_crash(tmp_path):
    assert cls(tmp_path, run_step=step(rc=134, out=GOOD)).klass == "crash-hang"


def test_classify_wrong_value(tmp_path):
    c = cls(tmp_path, run_step=step(out="CUPORT a u32 2 00000001 00000003\nCUPORT-DONE\n"))
    assert c.klass == "silently-wrong" and "a[1]" in c.notes


def test_classify_missing_and_extra_buffers(tmp_path):
    assert cls(tmp_path, run_step=step(out="CUPORT-DONE\n")).klass == "silently-wrong"
    extra = "CUPORT b u32 1 00000000\n" + GOOD
    assert cls(tmp_path / "x", run_step=step(out=extra)).klass == "silently-wrong"


def test_classify_malformed_line(tmp_path):
    c = cls(tmp_path, run_step=step(out="CUPORT a u32 3 1\n" + GOOD))
    assert c.klass == "silently-wrong" and "malformed" in c.notes


def test_classify_within_tolerance(tmp_path):
    exp = {"a": co.Buffer("f32", [1.0])}
    out = f"CUPORT a f32 1 {co.f32_bits(1.0) + 1:08x}\nCUPORT-DONE\n"
    c = cls(tmp_path, run_step=step(out=out), expected=exp, yaml_extra='outputs:\n  a: {tolerance: "ulp:1"}\n')
    assert c.klass == "within-tolerance"


def test_classify_text_lines_multiset(tmp_path):
    out = "t1\nt0\n" + GOOD
    assert cls(tmp_path, run_step=step(out=out), text=["t0", "t1"]).klass == "bit-exact"
    assert cls(tmp_path / "x", run_step=step(out=GOOD), text=["t0"]).klass == "silently-wrong"


def test_observe_only_keeps_class_and_marks_notes(tmp_path):
    c = cls(tmp_path, run_step=step(out="CUPORT-DONE\n"), yaml_extra="observe_only: true\n")
    assert c.klass == "silently-wrong" and c.notes.startswith("observe-only")


def test_triage_applies_only_to_matching_raw_class():
    entries = [{"tool": "t", "probe": "p/x-001", "raw_class": "silently-wrong",
                "class": "documented-divergence", "reference": "https://example.org/doc", "upstream": None,
                "notes": "README says so"}]
    c = apply_triage(Cell("silently-wrong", "a[0] differs"), "t", "p/x-001", entries)
    assert c.klass == "documented-divergence" and "raw silently-wrong" in c.notes and c.reference
    assert apply_triage(Cell("bit-exact", ""), "t", "p/x-001", entries).klass == "bit-exact"
    assert apply_triage(Cell("silently-wrong", ""), "other", "p/x-001", entries).klass == "silently-wrong"
