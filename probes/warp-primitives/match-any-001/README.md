# warp-primitives/match-any-001

`__match_any_sync` with keys `in % 5`, so each warp has five groups of
roughly six lanes. Metal has no single-instruction counterpart, so a tool
either emulates it (for example with a ballot per distinct key) or refuses
it. `expect: either` records that a clear refusal is acceptable, while a
wrong mask is not.
