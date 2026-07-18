## Status

Accepted for the next `kwavers-gpu` major release.

## Context

`PstdOutputRequest` distinguishes sensor traces from final pressure and
staggered-velocity fields. A treatment planner consumes a pressure envelope,
defined pointwise by

`P_peak(x) = max_{0 <= n < N_t} |p(x, n)|`.

The final pressure frame is not that quantity: it may occur after the burst has
passed a focal voxel and can understate the therapeutic pressure. Downloading
all time steps to construct the envelope on the host is not viable for clinical
volumes and would defeat provider-resident time marching.

## Options

1. Reinterpret the final pressure frame as a peak envelope.
2. Download every pressure frame and reduce on the host.
3. Accumulate the absolute pressure maximum on the GPU and download it only on
   explicit request.

## Decision

Use option 3. `PstdOutputRequest` becomes a composable request type with
independent final-field and peak-pressure selections. Each pressure update
dispatches one GPU reduction pass when and only when the request includes the
peak field. The peak values reuse the run-local output storage behind the
sensor buffer; sensor-only runs retain their former storage size and dispatch
sequence.

The result exposes the optional peak-pressure vector separately from optional
final fields. Its grid order remains the provider's existing row-major
`(x, y, z)` flattening. A request for both returns both, so no consumer must
re-run the simulation or infer a peak from final state.

## Consequences

- This is a [major] API change: callers construct an explicit output request
  instead of selecting an enum variant.
- Peak collection adds one full-volume device-output region and one linear
  compute pass per time step only when requested.
- A live WGPU regression proves `P_peak[i] >= |p_final[i]|` for every voxel;
  this is the direct value invariant of the accumulator.
- This supplies the output contract needed by a full-wave planning consumer,
  but it does not prove that a treatment-scale three-dimensional grid fits a
  given device. Allocation remains a per-plan provider constraint.

## Rejected alternatives

- Final-frame substitution is physically false for a transient burst.
- Host time-series reduction turns an `O(N)` output into `O(N * N_t)` transfer
  and requires retaining every frame.
- A downstream-local accumulator would duplicate provider-owned GPU state and
  violate the stack ownership boundary.
