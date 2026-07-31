# ADR 078: Type Clinical Workflow Quantities

- Status: Accepted
- Date: 2026-07-31
- Driver: `KWAVERS-AEQ-MET-40`

## Decision

Clinical workflow latency, acquisition duration, processing duration, total
duration, and per-stage timing use Aequitas `Time<f64>` stored in SI seconds.
The workflow latency configuration is also a `Time<f64>` so the comparison
contract cannot silently mix milliseconds and seconds.

Clinical confidence and measured GPU utilization use Aequitas
`Dimensionless<f64>` and retain their percentage convention (0-100). GPU
utilization is optional because the workflow currently has no connected
telemetry provider. Memory usage is an optional byte count at the explicit
storage-instrumentation boundary; Aequitas has no information dimension.

Stage measurements are recorded for the individual stage interval rather than
the cumulative elapsed workflow time. Synthetic GPU and memory samples are
removed because generated values are not measurements.

## Eunomia compatibility

The clinical workflow metrics are real-valued durations, percentages, and byte
counts. They contain no phasor or imaginary component, so no complex unit is
introduced. Future coherent imaging results must use the existing
Eunomia-backed complex scalar support at the formula or dense-storage boundary
without changing these real-valued workflow contracts.

## Rejected alternative

Keeping `Duration`, millisecond-suffixed latency fields, and raw confidence
scalars would leave unit semantics at call sites. Reporting unavailable GPU or
memory values as zero, `NaN`, or trigonometric samples would present absence as
measurement. Assigning bytes to a physical Aequitas dimension would be
dimensionally incorrect.

## Verification

Against the delivered revision, `cargo check -p kwavers-diagnostics --tests
--offline` passes; focused workflow Nextest passes 57/57 with 136 skipped;
warning-denied all-target Clippy passes; doctests pass with 1 executable and 5
ignored; RustDoc, formatting, and diff checks pass. Shared unused-provider-patch
and linker warnings remain outside this decision.
