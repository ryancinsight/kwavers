# ADR 037: GPU PSTD output contract

## Status

Accepted

- Revision (2026-08-14): the `PstdOutputRequest` closed enum recorded below was
  replaced by the composable final-field and peak-pressure selections of
  [`ADR-040`](040-gpu-pstd-peak-pressure-output.md). The decisions this record
  still owns remain current: the GPU adapter exposes actual provider values
  rather than zero arrays, final-field readback is opt-in, and
  `SolverType::PstdGpu` returns `FeatureNotAvailable` instead of substituting
  CPU PSTD. ADR 040 makes no runner or substitution decision.
- Revision (2026-08-26): ADR 125 replaced `SolverType::PstdGpu` with
  `SolverType::PSTD` plus `FftBackend::Hephaestus`, and Hephaestus replaced the
  private bounded radix-2 FFT. The output and no-substitution decisions remain
  unchanged; the current selector and provider limits below reflect that
  migration.
- Revision (2026-11-11): no decision changed. The Verification section
  names the regressions added in ryancinsight/kwavers#1053, after three
  mutations of the code this record governs passed every test: the factory
  building the Leto solver for a Hephaestus request, the adapter keeping
  its zero arrays after a batch, and the provider returning final fields
  that were not requested.

## Context

`GpuPstdSimulationAdapter` implemented the generic `Solver` trait but returned
zero pressure and velocity arrays after a GPU batch. `SimulationRunner` also
mapped `SolverType::PstdGpu` to CPU PSTD in both GPU-enabled and GPU-disabled
builds. Both paths misrepresented the selected solver and could feed a
synthetic zero field to downstream focusing, statistics, or safety logic.

The WGPU PSTD state already owns the final pressure and staggered velocity
buffers, and its provider command contract supports staging-buffer readback.
Hephaestus prepares the rank-3 FFT over arbitrary positive dimensions, using
singleton axes for 1-D and 2-D grids. Its three lossless bind groups require 23
storage buffers per compute-shader stage; fractional-Laplacian absorption adds
a fourth eight-buffer group and requires 31.

## Decision

`GpuPstdSolver::run` accepts `PstdOutputRequest` and returns `PstdRunResult`.
`PstdOutputRequest::sensor_traces()` retains sensor-only transfer behavior.
`PstdOutputRequest::with_final_fields()` returns a `PstdFinalFields` value
containing final pressure and all three staggered velocity fields in row-major
grid order.

The generic GPU adapter requests final fields and exposes those actual values
through `Solver::{pressure_field,velocity_fields,statistics}`. The simulation
runner maps `SolverType::PSTD` with `FftBackend::Hephaestus` to the GPU path.
Unsupported request contracts and device failures are explicit errors. It
never selects Leto CPU execution as an implicit substitute.

The adapter and direct runner share one provider-owned medium snapshot and one
execution path. Medium absorption ownership is resolved while the snapshot is
prepared, before CPML construction or device acquisition; the adapter retains
that snapshot across batches instead of duplicating preparation logic.

## Consequences

- This was a [major] `kwavers-gpu` API change: callers add an explicit output
  request and consume `PstdRunResult`. ADR 040 supersedes the closed enum with
  composable final-field and peak-pressure selections.
- Final-field readback transfers four full volumes and is opt-in. Sensor-only
  runs retain their existing transfer budget.
- At this decision point a peak-over-time consumer remained unsupported; ADR
  040 now provides the provider-side envelope. CT-scale planning still needs a
  per-plan allocation-capacity check rather than a final-field substitution.
- Lossless PSTD remains available on a 23-buffer device. Absorption is an
  explicit 31-buffer capability requirement, not a reason to use Leto.

## Rejected alternatives

- Retain zero arrays with documentation: preserves a fabricated field value.
- Always read all fields: makes sensor-only acquisition pay a four-volume host
  transfer.
- Fall back to CPU PSTD for a GPU request: changes the selected backend and
  hides capability failures.

## Verification

The provider regressions run real WGPU batches. One requests full output and
verifies field cardinality and finite pressure. One sends four requests
through one solver and verifies that each result holds exactly the outputs
its request names. One compares the final staggered velocities with the
independent CPU reference.

The adapter regressions verify the row-major transfer element by element on
a grid of three distinct extents, with exact peak statistics, and on a device
that the final pressure at each sensor equals the last sample that sensor
recorded in the same batch. The factory regressions verify that a Hephaestus
PSTD configuration assembles the GPU adapter, and is rejected in a build
without the `gpu` feature, and that a Leto configuration assembles the Leto
solver. Feature-configured runner regressions execute a
non-power-of-two Hephaestus case and reject unsupported contracts instead of
selecting Leto. Two factory regressions (a medium with two absorption
exponents; a zero time step) and two runner regressions (such a medium;
zero time steps) verify the message of the adapter's or the provider's
refusal, which a fall-back to Leto on that error would replace with a
solver, a result or another error. No regression makes the provider return
a device error.

Of the regressions that need a device, the four-request one, the
CPU-reference one and those of the adapter and the runner are ignored by
default; the full-output one returns early when it finds no adapter. No
workflow runs the tests of the `gpu` build of `kwavers-simulation`, so the
evidence for its device regressions is a run on a host with an adapter.

### Theorem: output requests cannot fabricate fields

For the original sensor-trace request, the result does not expose an
unrequested field. For the final-field request, each returned volume is
produced by a sequential copy from the provider-owned pressure or
staggered-velocity buffer into the row-major staging buffer; the exact C-order
adapter regression checks every element and the real WGPU regression checks the
live execution path. ADR 040 extends this proof to the peak-pressure envelope.
Finally, the generic runner maps `FftBackend::Hephaestus` directly to the GPU
operation and returns typed errors for unsupported inputs or acquisition
failure, so no branch can substitute Leto for a GPU request. The theorem is
supported by type-level result selection and
value-semantic tests, not by a mock or a zero-filled fallback.
