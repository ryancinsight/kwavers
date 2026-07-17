## Status

Accepted for the next breaking `kwavers-gpu` release.

## Context

`GpuPstdSimulationAdapter` implemented the generic `Solver` trait but returned
zero pressure and velocity arrays after a GPU batch. `SimulationRunner` also
mapped `SolverType::PstdGpu` to CPU PSTD in both GPU-enabled and GPU-disabled
builds. Both paths misrepresented the selected solver and could feed a
synthetic zero field to downstream focusing, statistics, or safety logic.

The WGPU PSTD state already owns the final pressure and staggered velocity
buffers, and its provider command contract supports staging-buffer readback.
The current GPU algorithm remains limited to power-of-two axes no larger than
256 cells and does not produce a time-maximum pressure envelope. Its three
lossless bind groups require 24 storage buffers per compute-shader stage;
fractional-Laplacian absorption adds a fourth eight-buffer group and requires
32.

## Decision

`GpuPstdSolver::run` accepts `PstdOutputRequest` and returns `PstdRunResult`.
`SensorTraces` retains sensor-only transfer behavior. `SensorTracesAndFinalFields`
returns a `PstdFinalFields` value containing final pressure and all three
staggered velocity fields in row-major grid order.

The generic GPU adapter requests final fields and exposes those actual values
through `Solver::{pressure_field,velocity_fields,statistics}`. The simulation
runner returns `FeatureNotAvailable` for `SolverType::PstdGpu` until it can
map every request input and result field to the GPU adapter. It never selects
CPU PSTD as an implicit substitute.

## Consequences

- This is a [major] `kwavers-gpu` API change: callers add an explicit output
  request and consume `PstdRunResult`.
- Final-field readback transfers four full volumes and is opt-in. Sensor-only
  runs retain their existing transfer budget.
- A consumer requiring peak-over-time or CT-scale GPU planning remains
  unsupported; it receives an explicit constraint error rather than a final
  field mislabelled as a peak field.
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

For `SensorTraces`, the result type contains only sensor data, so a caller
cannot observe an unrequested field through the GPU run contract. For
`SensorTracesAndFinalFields`, each returned volume is produced by a sequential
copy from the provider-owned pressure or staggered-velocity buffer into the
row-major staging buffer; the exact C-order adapter regression checks every
element and the real WGPU regression checks the live execution path. Finally,
the generic runner maps `SolverType::PstdGpu` to a typed `FeatureNotAvailable`
error, so no branch can substitute CPU PSTD for a GPU request. The theorem is
supported by type-level result selection and value-semantic tests, not by a
mock or a zero-filled fallback.

The current evidence is 144/144 GPU-feature tests (one skipped), 1036/1036
default scoped tests (four skipped), warning-denied Clippy, warning-clean
all-feature Rustdoc, and the 2/2 Hephaestus provider-limit regression merged
as `cf4df20`.
