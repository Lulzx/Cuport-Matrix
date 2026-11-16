"""Compile and run probes through one tool, in parallel.

Compiles are CPU-bound and use `jobs` workers. Each finished compile is
handed straight to a separate pool of `gpu_jobs` workers that run probes on
the GPU, so compiling and running overlap."""

from __future__ import annotations

import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

from cuport.classify import apply_triage, classify
from cuport.probe import Probe, load_oracle
from cuport.protocol import parse_output

COMPILE_TIMEOUT_S = 300


def _cells_for(probe: Probe, cell) -> list[dict]:
    return [{"feature": f"{probe.family}.{f}", "probe": probe.id, "class": cell.klass,
             "oracle": probe.oracle, "tolerance": probe.tolerance, "evidence": "[reproduced]",
             "upstream": cell.upstream, "notes": cell.notes,
             **({"reference": cell.reference} if cell.reference else {})}
            for f in probe.features]


def run_matrix(adapter, probes: list[Probe], build_root: pathlib.Path, jobs: int, gpu_jobs: int,
               triage: list[dict], progress=None) -> list[dict]:
    build_root.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []
    lock = threading.Lock()
    oracles = {p.id: load_oracle(p) for p in probes}  # fail fast on a broken oracle

    def finish(probe, compile_step, exe, run_step):
        expected, text_lines = oracles[probe.id]
        parsed = parse_output(run_step.stdout) if run_step else None
        cell = classify(probe, compile_step, exe, run_step, parsed, expected, text_lines)
        cell = apply_triage(cell, adapter.id, probe.id, triage)
        with lock:
            cells.extend(_cells_for(probe, cell))
            if progress:
                progress(probe, cell)

    with ThreadPoolExecutor(gpu_jobs, thread_name_prefix="gpu") as gpu_pool, \
            ThreadPoolExecutor(jobs, thread_name_prefix="cc") as cc_pool:
        run_futures = []

        def compile_one(probe):
            out_dir = build_root / probe.id
            out_dir.mkdir(parents=True, exist_ok=True)
            step, exe = adapter.compile(probe.dir, out_dir, COMPILE_TIMEOUT_S)
            if exe is None or step.returncode != 0:
                finish(probe, step, exe, None)
                return
            with lock:
                run_futures.append(gpu_pool.submit(
                    lambda: finish(probe, step, exe, adapter.run(exe, probe.timeout_s))))

        for f in [cc_pool.submit(compile_one, p) for p in probes]:
            f.result()
        cc_pool.shutdown(wait=True)
        for f in list(run_futures):
            f.result()
    return sorted(cells, key=lambda c: (c["probe"], c["feature"]))
