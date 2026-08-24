# 122. Per-view element-position rotation for the rotating opposed-linear-array acquisition

- Status: Accepted
- Date: 2026-08-20
- Item: `backlog.md#fwi-024-d` increment 2 (atlas `backlog.md#atlas-usct-fwi-024`)
- Depends on: ADR 115 (`TransmissionAcquisition` seam)

## Context

ADR 115 extracted the `TransmissionAcquisition` trait and proved the seam is
general. FWI-024-D's second increment is the rotating opposed-linear-array
acquisition itself: two linear transducer arrays facing each other across a
water tank, mounted on a rotation stage, swept through views at even angular
intervals.

The seam's `receivers(transmit)` method returns per-transmit receiver
coordinates, which is what makes a rotating acquisition expressible at all: at
view `v` both arrays are at angle `v * step`, and every element's position
changes. Two routes exist for implementing this.

### Route (a) — per-view element positions on one fixed reconstruction grid

The acquisition computes each element's physical `(x, y, z)` position at each
rotation angle and returns those coordinates through the `TransmissionAcquisition`
trait. The inversion loop does not change. The Born and CBS operators already
accept continuous coordinates and evaluate the scalar Green's function at each
receiver position, so they tolerate any coordinate, not only on-grid ones.

**The finite-window PSTD path is excluded.** `finite_window::receiver_indices_on_grid`
calls `exact_axis_index`, which rejects any coordinate not within 1e-9 of a
grid node. A linear array rotated by any angle that is not a multiple of 90°
will land most or all receivers off-grid. This path was written for a ring
array whose elements are placed on grid nodes by design; it is not suitable
for off-grid element positions without a BLI-receiver extension that is out of
scope here.

The interpolation, such as it is, lives in the **forward model** (Green's
function evaluated at a non-grid point), not in the gradient. The gradient
cross-correlation runs entirely in the volume representation, unaffected by
element coordinates.

### Route (b) — per-view model interpolation

Resample the slowness volume from the fixed reconstruction grid to each view's
axis-aligned simulation grid (rotated by the view angle), simulate there, rotate
the gradient back, and accumulate. Every operator works, including
finite-window PSTD, because each simulation sees an axis-aligned grid.

The cost is interpolation error **inside the gradient accumulation**. That error
is systematic in view angle: view 0° and view 180° introduce the same BLI error
in opposite halves of the volume, so the per-view errors do not average out in
the way thermal noise does. Deriving a rigorous bound on how much that
systematic bias shifts the recovered phantom requires more than a round-trip
test; it requires characterising the BLI kernel's interaction with the
phantom's spatial frequency content.

Route (b) is also more expensive: one forward resampling and one backward
resampling per view per iteration, each spanning the full volume.

## Decision

Adopt route (a). The implementation adds a `RotatingOpposedLinearArray` type to
`crates/kwavers-physics` that implements `TransmissionAcquisition` and accepts
any `f64` element position. `finite_window` is excluded from the rotating
acquisition — it is a diagnostic path for ring-array validation, and extending
it to BLI receivers is a separately scoped item once a concrete use case arises.

The acceptance oracle is:
- `RotatingOpposedLinearArray` driven through the Born operator recovers a
  known sound-speed anomaly from a simulated 360°/2° sweep within a derived
  tolerance.
- A single-view round-trip test: element positions rotated by `θ` then by `-θ`
  reproduce the original positions within floating-point tolerance.

## Consequences

A `RotatingOpposedLinearArray` implementation is added with the following API:
```rust
pub struct RotatingOpposedLinearArray { ... }
impl RotatingOpposedLinearArray {
    pub fn new(
        elements_per_array: usize,
        element_pitch_m: f64,
        standoff_m: f64,
        view_count: usize,
    ) -> KwaversResult<Self>;
}
impl TransmissionAcquisition for RotatingOpposedLinearArray { ... }
```

`transmission_count()` returns `elements_per_array * view_count` — each element
of the transmit array fires once per view. `receiver_count()` returns
`2 * elements_per_array` — both arrays receive on every transmit. `sources(t)`
and `receivers(t)` return pre-computed slices for the view corresponding to
transmit `t`.

The finite-window PSTD operator (`PstdFiniteWindowBornOperator`) documents that
it requires on-grid element positions and will return `KwaversError::InvalidInput`
if off-grid receivers are passed; this is existing behaviour, not a regression.

## Alternatives rejected

**Route (b): per-view model interpolation.** Interpolation error enters the
gradient in a view-correlated pattern that is harder to bound than the purely
physical approximation route (a) makes. Route (b) also requires two full-volume
resamples per view per iteration, which is more expensive for no accuracy gain
on the operators that already support off-grid positions.

**Extending `exact_axis_index` to BLI in finite_window.** The extension is
localised but non-trivial: `receiver_indices_on_grid` would need fractional
index storage, and `simulate_transmit` would need to replace direct array
indexing with trilinear interpolation. The scope is larger than the rotating
acquisition warrants; the Born and CBS operators already cover the clinical USCT
use case.

**Placing `RotatingOpposedLinearArray` in `kwavers-solver`.** The geometry is
pure physics, independent of the inversion: element positions, rotation geometry,
standoff distance. It belongs in `kwavers-physics` alongside `MultiRowRingArray`.
