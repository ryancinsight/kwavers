# ADR 136: One rectangle element in the k-Wave array

- Status: Accepted
- Date: 2026-10-20
- Item: `backlog/kw-rect-planar-consolidation.md`

## Context

`KWaveArray` carries two rectangles. `ElementShape::Rect` comes from
`add_rect_element`/`add_rect_rot_element` and the convex array (ADR 112).
`PlanarApertureShape::Rectangle` comes from `add_planar_aperture_element` and
shares the Rayleigh kernel's outline (ADR 035). Both describe a rectangle with
its width along `R·x̂` and its normal along `R·ẑ`, and both sample it on the
same lattice: `ceil(10·A/dx²)` points split by aspect ratio. Each has its own
point generator, mask dispatch and transform rule. The duplicate has already
diverged:

- **Edge sample.** With one sample along an axis, the `Rect` generator places
  it at the edge (`-w/2`) rather than the centre. k-wave-python's
  `make_cart_rect` and the planar generator both use the midpoint.
- **Count clamp.** The planar generator clamps the per-axis count to the target
  count; `Rect` does not.
- **Deposition.** `Rect` adds `scale` times the BLI stencil at each sample,
  with no normalisation. The planar path divides by the stencil sum, so each
  sample deposits exactly `scale`.
- **Array transform.** `set_array_position` moves a `Rect` centre and skips
  planar apertures entirely, so the transform is silently lost.
- **Length.** `Rect` takes a third `length` that no rasterizer reads.

The deposition difference is not a defect on either side. The `Rect` value is
an external reference. `test_rect_weighted_mask_matches_kwave_python_reference_mass`
pins 16.036 130 608 724 637 for an 8×2-cell rectangle, where the exact area is
16 cells. The test came from a k-wave-python run (commit `aeff4fb1a`).
k-wave-python's `get_off_grid_points` (`kwave/utils/kwave_array.py`, `rect`
branch and `off_grid_points`) adds `scale · sinc(...)` over the tolerance
stencil without renormalising. That source also applies the array affine to
the rectangle centre only (`self.affine(position)`), keeps the element
orientation, and takes `add_rect_element(position, Lx, Ly, theta)` with no
length. The planar path's exact area is ADR 035's contract.

## Decision

1. **One representation.** Delete `ElementShape::Rect`. `add_rect_element`
   and `add_rect_rot_element` build a `KWaveElement::PlanarAperture` with
   `PlanarApertureShape::Rectangle`. The normal is `R·ẑ` and the first axis is
   `R·x̂`, where `R` is the existing Euler XYZ matrix. One point generator
   serves every rectangle: the planar midpoint lattice with its clamp.
2. **The reference contract is a typed element property.** The planar element
   carries `SurfaceConvention`, a closed two-variant enum:
   - `KWave`: unnormalised BLI deposition, and `set_array_position` moves the
     centre only. This is k-wave-python's rectangle contract.
   - `AreaConserving`: normalised deposition, and `set_array_position` applies
     the full rigid transform to the centre, normal and first axis. This is
     ADR 035's contract.

   The k-Wave-named constructors, and through them the convex array, use
   `KWave`. `add_planar_aperture_element` uses `AreaConserving`. The binary
   mask is the same under both, because `mask_only` deposits no weights.
3. **Drop `length`.** The Rust signatures become `(position, width, height)`
   and `(position, width, height, euler_xyz_deg)`. The Python `dims` becomes
   a `(width, height)` pair, matching k-wave-python's `Lx, Ly`.
4. **Planar apertures honour the array transform.** This replaces the silent
   skip in mask, weighted-mask and geometry dispatch.

ADR 112's statement that the `add_*_element` signatures stay fixed is revised
for `add_rect_*` only. Arc, disc, bowl and annulus keep their signatures.

## Rejected alternatives

- **Area-conserving deposition for every rectangle.** This breaks the
  k-wave-python reference mass by construction (about 16.000 against 16.036).
  It also breaks the parity layer ADR 112 keeps.
- **BLI deposition for every planar aperture.** This drops ADR 035's
  exact-area verification and makes Rayleigh-coupled masks depend on stencil
  truncation.
- **Keep both shapes and fix the edge and clamp defects in each.** The two
  generators would still be duplicates, and the defects show they drift.
- **Separate deposition and transform enums.** k-wave-python fixes both
  together, so independent switches would admit two combinations no reference
  defines.

## Consequences

- The edge-sample and clamp defects disappear with the second generator.
- Planar apertures placed through `set_array_position` now move. Before this,
  they rasterized at their unplaced geometry.
- `pykwavers` 0.1.0 is unpublished, so the Python break reaches in-repo
  callers only. Those callers migrate in the same change, and the change
  carries a `BREAKING CHANGE:` footer.
- `ElementShape::Disc` and `PlanarApertureShape::Disk` are the same kind of
  duplicate pair. They are filed as `KW-DISC-PLANAR-CONSOLIDATION`, which
  needs this item.

## Verification

- The k-wave-python reference mass holds unchanged under `KWave`.
- The rectangle 90° swap and `set_array_position` IoU tests are unchanged.
- A one-sample-wide rectangle has its sample at the centre.
- A planar `AreaConserving` element under `set_array_position` matches the
  manually transformed geometry exactly.
- A `KWave` rectangle under a rotating transform keeps its orientation.
- The planar rectangle's area-conservation test is unchanged.
