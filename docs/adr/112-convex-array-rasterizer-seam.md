# 112. Wiring the convex array: orientation seam and angle units

- Status: Accepted
- Date: 2026-08-19
- Item: `backlog.md#cov-3`

## Context

`curvilinear::ConvexArrayGeometry` lays out N elements on a circular arc and is
covered by six behavioural tests. It is referenced nowhere outside its own
module, so the layout can be computed but cannot drive a simulation. Closing
COV-3 means feeding it to `KWaveArray`.

The obvious route does not work. `ElementShape::Arc` is the primitive the
backlog entry pointed at, and it disagrees with the geometry on three axes at
once:

| | plane | angle reference | units |
| --- | --- | --- | --- |
| `ConvexArrayGeometry` | x–z, `[r sinθ, 0, r(cosθ−1)]` | +z, apex at θ = 0 | radians |
| `ElementShape::Arc` (`rasterize_arc_points`) | x–y at constant z, `[cx + r cosθ, cy + r sinθ, cz]` | +x | degrees |

`rasterize_arc_points` writes `center.2` unchanged into every sample, so the Arc
primitive cannot express an arc outside the x–y plane at all. It carries no
orientation parameter. Mapping the geometry onto it would misplace every element
while still producing a plausible mask — a wrong answer rather than an error.

Two further facts settle the decision.

**The array already has an orientation-carrying primitive.**
`ElementShape::Rect` takes `euler_xyz_deg`, and `rasterize_rect_points` builds
its lattice at `(lx, ly, 0.0)` before rotating, so a rect's local normal is `+z`.
`euler_xyz_rotation_matrix` composes `Rz · Ry · Rx`, whose y-block is
`[[cos β, 0, sin β], [0, 1, 0], [−sin β, 0, cos β]]`. Therefore

```text
My(β) · [0, 0, 1]ᵀ = [sin β, 0, cos β]
```

and `ConvexArrayGeometry::element_normal(i) = [sin θ, 0, cos θ]`. These are the
same vector at `β = θ`. The convex array is exactly a set of rect elements each
rotated about y by its own element angle, with no sign flip and no axis swap.

**Angles already have a type in this crate, used inconsistently.**
`kwavers-transducer` depends on `aequitas` and `array_2d` stores its steering
angle as `aequitas::systems::si::quantities::Angle<f64>`, while `curvilinear`
and `kwave_array` pass bare `f64` and rely on parameter names (`euler_xyz_deg`,
`angular_pitch`) to carry the unit. The degrees/radians half of the conflict
above exists only because the unit is a naming convention rather than a type.

Aequitas models `Angle` as a `Dimension` with its own `AngleSemantics`, so an
angle is not interchangeable with a bare scalar. It defines exactly one unit for
that dimension, `Radian` (`SCALE = 1.0`). **There is no `Degree` unit anywhere in
aequitas**, and no degree↔radian conversion. Call sites work around it visibly:

```rust
Angle::from_unit::<Radian>(15.0_f64.to_radians())
```

## Decision

**1. Wire the convex array through the rotated-rect primitive, not the arc.**
Add a geometry-taking constructor following the `add_planar_aperture_element`
precedent, mapping each element to `add_rect_rot_element` at
`euler_xyz_deg = (0, θᵢ in degrees, 0)`. Derived above; assert it rather than
trust it — a test that the rasterised element normal equals
`element_normal(i)` for a non-trivial θ is the oracle, because the failure mode
is silent misplacement.

**2. Type the angle at this seam.** New transducer-geometry API takes
`aequitas::Angle<f64>` rather than `f64`, matching `array_2d`. This makes the
degrees/radians mismatch unrepresentable instead of documented. The conversion
to `euler_xyz_deg` happens once, inside the constructor, at the boundary with
the existing `f64`-degrees rasterizer surface.

`ElementShape` and the `add_*_element` family keep their current `f64` degree
signatures. They mirror k-Wave's `addArcElement` / `addRectElement` names and
argument order for parity, and rewriting that surface is a separate decision
from closing COV-3.

**3. Add `Degree` to aequitas upstream.** A `LinearUnit<dimensions::Angle>` with
`SCALE = π/180`, alongside `Radian`. It is the same shape as every other unit in
`units/base.rs`, it removes the `.to_radians()` dance at call sites, and it is
the provider's job under upstream ownership — kwavers should not carry a local
degree type for a dimension aequitas already owns.

## Alternatives rejected

**Give `ElementShape::Arc` an orientation parameter.** Changes a k-Wave-parity
primitive's signature for one caller's benefit, and the rotated rect already
provides orientation. If a genuinely arc-shaped *element* is later needed on an
oblique plane, revisit then with that requirement in hand.

**Express the convex layout in the rasterizer's x–y plane.** Would make
`element_position`/`element_normal` disagree with the physical convention the
module documents (apex at the origin facing +z), moving the conflict rather than
resolving it.

**Keep `f64` and document the units.** That is the status quo, and it is what
produced a geometry and a rasterizer whose angles differ in plane, reference
axis, and unit without any of the three being caught by a type.

## Consequences

COV-3 closes with a helper whose correctness is asserted against
`element_normal`, not argued from the geometry. Angle-typed transducer APIs
begin at this seam; the `f64`-degree `add_*_element` surface stays as the
k-Wave-parity layer beneath it, with one documented conversion between them.
Item 3 is an aequitas change and does not block items 1 and 2 — until it lands,
the constructor converts with `to_degrees()` at the single boundary point.
