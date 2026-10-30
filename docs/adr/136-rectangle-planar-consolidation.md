# ADR 136: One rectangle element in the k-Wave array

Status: Accepted

- Revision 2026-10-30 (rigid transform, salvaged from superseded PR
  #967): the installed array transform moves every planar centre by
  `R·p + t`, but the orientation now follows the surface convention —
  centre-only under `KWave` (k-wave-python parity), full rigid rotation
  under `AreaConserving` via `apply_transform_planar`. Ported its two
  contract tests (rigid frame, per-element paths).
- Revision 2026-10-30 (disc fold, item KW-DISC-PLANAR-CONSOLIDATION):
  `ElementShape::Disc` folds into `PlanarApertureShape::Disk` under
  `SurfaceConvention::KWave` the same way: the disc spiral runs on the
  stored frame through `rasterize_framed_disc_points`, so the k-Wave
  reference mass (28.339929259209097) and the centre sample hold unchanged.
  `ProfiledDisc` keeps its variant (no planar profile slot exists).
  `add_disc_element` keeps its signature — there was no dead parameter —
  and the installed array transform moves disc centres like every other
  planar element.

## Context

`kwave_array::ElementShape::Rect` and `PlanarApertureShape::Rectangle`
describe the same object — a rectangle on one oriented lattice (first axis
`R·e_x`, normal `R·e_z`) — through two point generators, two deposition
rules, and different `set_array_position` handling:

- **Surface mapping.** `Rect` deposits through the unnormalized k-Wave BLI
  (`map_surface_sample`), pinned by
  `test_rect_weighted_mask_matches_kwave_python_reference_mass` at
  `16.036130608724637`. The planar path renormalizes through
  `map_area_conserving_surface_sample`.
- **Lattice.** `Rect` samples the `[-1,1]^2` canonical lattice with a
  half-cell offset (`rasterize_rect_points`); the planar path samples a
  near-square midpoint lattice (`rasterize_planar_aperture_points`). Same
  mapping alone does not give identical masks.
- **Transform.** The planar path ignores `array_transform`. `Rect` moves the
  centre by the full rigid motion and keeps the per-element Euler
  (`apply_transform_rect`), pinned by
  `test_set_array_position_matches_manual_position_rotation` (IoU ≥ 0.90).
- **Parameters.** `Rect.length` ("retained for API compatibility") is matched
  as `length: _` at every use site; nothing consumes it.

ADR 112 (Accepted) keeps the `add_rect_*` surface as the k-Wave parity
layer. Folding `Rect` into `Rectangle` therefore changes a parity contract
and a public Python surface, and must keep the 16.0361 reference value.

## Decision

1. **New `SurfaceConvention { KWave, AreaConserving }` carried by
   `PlanarApertureGeometry`.** It selects lattice *and* deposition together:
   `KWave` runs the rect-canonical lattice through the raw BLI stencil
   (bit-for-bit the current `Rect` path); `AreaConserving` runs the planar
   midpoint lattice through the renormalized stencil (current planar
   behavior, unchanged). Existing planar constructors keep `AreaConserving`;
   nothing they emit changes value.
2. **Migrate every `Rect` caller to `Rectangle` under `KWave`.**
   `add_rect_element` / `add_rect_rot_element` build
   `PlanarApertureGeometry::oriented` with centre `position`, normal `R·e_z`,
   first axis `R·e_x` (`R` the element Euler matrix), and drop `length`,
   which no code path reads. The 16.0361 reference holds because the same
   lattice is evaluated; sample positions may differ by floating-point
   reassociation, inside the test's tolerance by orders of magnitude.
3. **One transform rule for all planar elements.** The installed array
   transform moves each element centre by `R·p + t` and leaves the stored
   orientation unchanged — the k-wave-python contract the IoU test pins.
   Planar apertures previously ignored the transform; no test pins that
   behavior, and two rules would preserve the duplication this ADR removes.
4. **Delete `ElementShape::Rect`, `rasterize_rect*`, and
   `apply_transform_rect`.** The dispatcher routes `PlanarAperture` by its
   convention. The Python `add_rect_*` names and 3-tuple dims stay (parity),
   minus `length`, with a `BREAKING CHANGE:` footer; stubs, examples, and
   `test_bindings_surface.py` migrate in the same change.

## Alternatives rejected

- **Keep both paths.** Preserves the duplicate generators and the
  transform-handling split that produced this item.
- **Map the full array rotation into each element orientation.** Contradicts
  the k-wave-python centre-only contract and the IoU oracle.
- **Keep `length`.** Dead parameter; keeping it re-introduces the
  compatibility debt the merge exists to remove.

## Consequences

One rectangle representation. The k-Wave reference mass test passes
unchanged in value; migrated rect masks reproduce the retired path at the
reference tolerance under the same mapping and transform. ADR 112's parity
layer survives as the `KWave` convention plus the `add_rect_*` names; its
"rewriting that surface is a separate decision" is superseded here,
recorded as a revision note on ADR 112.
