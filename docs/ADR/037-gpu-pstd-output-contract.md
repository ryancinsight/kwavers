## Status

Superseded by [`ADR-040`](040-gpu-pstd-peak-pressure-output.md).

## Context

`GpuPstdSimulationAdapter` implemented the generic `Solver` trait but returned
zero pressure and velocity arrays after a GPU batch. `SimulationRunner` also
mapped `SolverType::PstdGpu` to CPU PSTD in both GPU-enabled and GPU-disabled
builds. Both paths misrepresented the selected solver and could feed a
synthetic zero field to downstream focusing, statistics, or safety logic.

The WGPU PSTD state already owns the final pressure and staggered velocity
buffers, and its provider command contract supports staging-buffer readback.
The current GPU algorithm accepts power-of-two axes through 1,024 cells and
does not yet produce a time-maximum pressure envelope. Its three
lossless bind groups require 24 storage buffers per compute-shader stage;
fractional-Laplacian absorption adds a fourth eight-buffer group and requires
32.

## Decision

`GpuPstdSolver::run` accepts `PstdOutputRequest` and returns `PstdRunResult`.
`PstdOutputRequest::sensor_traces()` retains sensor-only transfer behavior.
`PstdOutputRequest::with_final_fields()` returns a `PstdFinalFields` value
containing final pressure and all three staggered velocity fields in row-major
grid order.

The generic GPU adapter requests final fields and exposes those actual values
through `Solver::{pressure_field,velocity_fields,statistics}`. The simulation
runner returns `FeatureNotAvailable` for `SolverType::PstdGpu` until it can
map every request input and result field to the GPU adapter. It never selects
CPU PSTD as an implicit substitute.

## Consequences

- This was a [major] `kwavers-gpu` API change: callers add an explicit output
  request and consume `PstdRunResult`. ADR 040 supersedes the closed enum with
  composable final-field and peak-pressure selections.
- Final-field readback transfers four full volumes and is opt-in. Sensor-only
  runs retain their existing transfer budget.
- At this decision point a peak-over-time consumer remained unsupported; ADR
  040 now provides the provider-side envelope. CT-scale planning still needs a
  per-plan allocation-capacity check rather than a final-field substitution.
- Lossless PSTD remains available on a 24-buffer device. Absorption is an
  explicit 32-buffer capability requirement, not a reason to use CPU PSTD.

## Rejected alternatives

- Retain zero arrays with documentation: preserves a fabricated field value.
- Always read all fields: makes sensor-only acquisition pay a four-volume host
  transfer.
- Fall back to CPU PSTD for a GPU request: changes the selected backend and
  hides capability failures.

## Verification

The GPU regression requests full output from a real WGPU batch and verifies
field cardinality and finite pressure. The adapter regression proves exact
row-major transfer into the generic solver fields and exact peak statistics.
Feature-configured runner regressions assert `FeatureNotAvailable` instead of
CPU execution. Package compilation and Nextest provide compiler and
value-semantic evidence; a real GPU device is required for the WGPU run path.

### Theorem: output requests cannot fabricate fields

For the original sensor-trace request, the result does not expose an
unrequested field. For the final-field request, each returned volume is
produced by a sequential copy from the provider-owned pressure or
staggered-velocity buffer into the row-major staging buffer; the exact C-order
adapter regression checks every element and the real WGPU regression checks the
live execution path. ADR 040 extends this proof to the peak-pressure envelope.
Finally, the generic runner maps `SolverType::PstdGpu` to a typed
`FeatureNotAvailable` error, so no branch can substitute CPU PSTD for a GPU
request. The theorem is supported by type-level result selection and
value-semantic tests, not by a mock or a zero-filled fallback.

The current evidence is 144/144 GPU-feature tests (one skipped), 1036/1036
default scoped tests (four skipped), warning-denied Clippy, warning-clean
all-feature Rustdoc, and the 2/2 Hephaestus provider-limit regression merged
as `cf4df20`.
