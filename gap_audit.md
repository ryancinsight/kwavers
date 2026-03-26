# Gap Audit

## Initial Findings (Sprint Start)
- Current structure claims zero circular dependencies, but the user directive overrides: circular dependencies and cross-contamination *are* currently present and must be resolved.
- Required to validate `pykwavers` against `k-wave-python`. Needs formal suite for 1-to-1 parity mapping.
- Deep nested file structures (3-5+ levels) required but may not be fully conforming.
- `pykwavers` needs to solely represent thin `PyO3` wrappings over `kwavers`, any core fixes must be bubbled down to `kwavers` itself.

## Priority Matrix
1. [Highest] Locate and prune circular dependencies and duplicate/inconsistent implementations across core modules.
2. [High] Define the validation suite matrix for Grid, Source, Signal, Sensor, Solver.
3. [Medium] Ensure GPU support is routed correctly via BURN crate.
