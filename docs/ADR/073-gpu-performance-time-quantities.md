# ADR 073: Type GPU Performance Time Metrics

- Status: Accepted
- Date: 2026-07-31
- Driver: `KWAVERS-AEQ-MET-35`

## Decision

The GPU real-time monitor and orchestrator carry elapsed durations and
simulation time as Aequitas `Time<f64>`, throughput as `Frequency<f64>`, and
percentages as `Dimensionless<f64>`. Public names omit embedded units. Scalar
extraction is limited to arithmetic, comparisons, and formatting at the
performance boundary.

`PhysicsKernel` returns estimated execution time as `Time<f64>`. This keeps the
analytical estimate and measured wall time on the same contract while leaving
the fixed 10 TFLOP/s estimate as an existing model assumption.

Eunomia compatibility is real-valued: these metrics are elapsed real times and
do not require `Complex<T>` or an imaginary-unit physical dimension. Complex
values remain outside this telemetry contract.

## Rejected alternative

Retaining millisecond-suffixed fields and converting at each caller would
preserve unit-bearing scalar ambiguity and duplicate conversion policy across
the monitor and realtime loop.

## Verification

Value-semantic monitor, kernel-estimate, and realtime-loop tests verify unit
preservation, budget comparisons, percentile values, throughput, and temporal
state migration. Package checks, Nextest, Clippy, doctests, Rustdoc, and
formatting are required; provider compilation remains subject to the existing
peer Leto graph state.
