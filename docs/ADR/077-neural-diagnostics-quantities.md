# ADR 077: Type Neural Diagnostics Quantities

- Status: Accepted
- Date: 2026-07-31
- Driver: `KWAVERS-AEQ-MET-39`

## Decision

Neural diagnostic lesion diameters and voxel edge lengths use Aequitas
`Length`. Beamforming performance targets and per-stage processing durations
use Aequitas `Time`. Clinical confidence, uncertainty thresholds, significance,
and GPU utilization are dimensionless quantities. Dense confidence and
uncertainty arrays remain at the Leto numerical-storage boundary.

Public unit-bearing names are removed: `size`, `voxel_size`, `performance_target`,
and stage `*_time` fields carry their type contract directly. The rolling
workflow history stores typed `Time` values; aggregate report maps preserve
their established millisecond presentation only at the explicit reporting
boundary. Memory accounting remains a
raw byte count because the current Aequitas provider has no information
dimension; assigning it a physical quantity would be dimensionally false.

GPU utilization is optional rather than represented by `NaN`: unavailable
hardware telemetry is absence, not a numeric metric.

## Eunomia compatibility

This neural diagnostics surface is real-valued. No public result represents a
phasor, complex impedance, or imaginary component, so no complex quantity or
imaginary unit is introduced. If a future coherent imaging boundary exposes a
complex physical result, it must use the existing Eunomia-backed Aequitas
complex scalar support at that formula/storage boundary.

## Rejected alternative

Keeping millimetre/millisecond suffixes and raw scalar fields would leave unit
conversion policy at every caller. Mapping memory bytes to `Frequency` or
another physical dimension would hide the absence of an information unit.
Using `NaN` for unavailable GPU telemetry would make invalid arithmetic appear
to be a measured result.

## Verification

The diagnostics test target compiles with `cargo check -p kwavers-diagnostics
--tests --offline`. The focused neural Nextest filter passes 40/40 tests with
151 unrelated diagnostics tests skipped. Warning-denied all-target Clippy,
doctests (1 executable and 5 ignored), Rustdoc, formatting, and diff checks
pass. The shared stack emits unused-provider-patch and linker warning
diagnostics; they are outside this contract.
