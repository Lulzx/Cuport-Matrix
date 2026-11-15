# probes/

Probe corpus for the Cuport-Matrix conformance and divergence matrix.

No probes are implemented yet. This directory records the intended layout so
that the first milestone (see `docs/survey-2026-11-15.md`, section 6) has a
fixed target.

## Layout

```
probes/
  <family>/                 # one of the families in matrix/features.yaml
    <probe-id>/
      probe.cu              # the CUDA source under test (plain CUDA C++)
      oracle.cpp | oracle.py | oracle.txt
                            # CPU reference, spec citation, or committed vector
      probe.yaml            # metadata: family, feature ids, tolerance, inputs
      README.md             # what the probe checks and why it is interesting
```

## Probe rules

- Deterministic inputs. Use the fixed-seed LCG described in
  `matrix/schema.md`; never `rand()` or time-based seeds.
- One behaviour per probe. A probe that checks shuffles and atomics is two
  probes.
- Every probe states its oracle class: `spec`, `cpu-reference`, or
  `reference-vector`.
- Every probe states a tolerance: `bit-exact`, an absolute or relative
  epsilon, or `ulp:<n>`.
- Probes must compile with stock `nvcc` syntax; no tool-specific extensions.

## Families (milestone v0)

1. `launch-geometry`
2. `shared-memory-barriers`
3. `warp-primitives`
4. `atomics`
5. `fp-semantics`
6. `streams-events`
7. `device-services`
8. `limits-refusals`
