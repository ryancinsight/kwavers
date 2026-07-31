# ADR 076: Type Visualization Telemetry Quantities

- Status: Accepted
- Date: 2026-07-31
- Driver: `KWAVERS-AEQ-MET-38`

## Decision

Visualization target and measured frame rates use Aequitas `Frequency`.
Render, transfer, simulation-frame, stream-latency, and pipeline-budget
values use Aequitas `Time`. Quality factors and drop rates use
`Dimensionless`. Public names carry the type contract instead of a unit
suffix. The typed boundary covers visualization configuration, metrics,
engine updates, bounded streaming, synchronization, and stage-pipeline
statistics.

Transfer byte bandwidth remains a derived storage-rate instrumentation value
in GiB/s. Aequitas has no information-rate dimension in the current provider;
mapping bytes per second to `Frequency` or a physical volumetric flow would be
dimensionally false. The field remains at that explicit instrumentation
boundary until a justified provider contract exists.

Eunomia compatibility is real-only. Visualization timing and quality metrics
have no complex or imaginary physical component, so no imaginary unit or
complex dimension is introduced.

## Rejected alternative

Keeping millisecond/fps suffixes and raw scalars would retain conversion policy
at every visualization caller. Treating byte bandwidth as frequency would
hide a dimension mismatch rather than extend Aequitas correctly.

## Verification

The default `kwavers-analysis` all-target package check, formatting, and
warning-denied Clippy pass. Package Nextest passes 724/724 tests in 4.223 s;
the doctest gate passes 1 executable test with 21 ignored examples. The GPU
feature all-target check and feature Clippy pass. The focused feature
visualization lane passes 17/17 tests, and the feature doctest gate passes 1
executable test with 21 ignored examples. The complete feature Nextest lane
exceeds the 60-second native-test budget during execution; this is a suite
budget residual outside the typed visualization contract.
