<a id="kw-rect-planar-consolidation"></a>

## KW-RECT-PLANAR-CONSOLIDATION — One rectangle element in the k-Wave array [arch] [major] — todo

priority: architecture; needs: none; scope: crates/kwavers-transducer/src/kwave_array, crates/kwavers-python (add_rect_element, add_rect_rot_element, .pyi, tests, examples), docs/adr/112-convex-array-rasterizer-seam.md

- **Finding.** `kwave_array::ElementShape::Rect` and `PlanarApertureShape::Rectangle` both describe a rectangle on the same oriented lattice: first axis `R·e_x`, normal `R·e_z`. They differ in three ways.
  - **Surface mapping.** `Rect` maps samples with unnormalized k-Wave BLI (`map_surface_sample`), pinned by `test_rect_weighted_mask_matches_kwave_python_reference_mass` at 16.0361. The planar path uses the area-conserving `map_area_conserving_surface_sample`.
  - **Transform.** The planar path ignores `array_transform`, while `apply_transform_rect` moves only the centre.
  - **Parameters.** `Rect`'s `length` parameter is "retained for API compatibility".
- **Why it is not a mechanical merge.** ADR 112 (Accepted) keeps the `add_rect_*` surface as the k-Wave parity layer. The Python `add_rect_element`/`add_rect_rot_element` take 3-tuple dimensions and appear in stubs, examples and `test_bindings_surface.py`. Folding `Rect` into `Rectangle` therefore changes a parity contract and a public Python surface.
- **Next step.** Draft ADR 136 revising ADR 112 in place: one `PlanarApertureShape::Rectangle` with an explicit surface-mapping choice (k-Wave band-limited versus area-conserving). Make the planar path honour `array_transform` (full rotation). Migrate every Rust and Python caller in one change, delete `ElementShape::Rect`, and keep the 16.0361 reference under the band-limited mapping.
- **Acceptance.** One rectangle representation. The k-Wave reference mass test passes unchanged in value. Planar and rect masks agree bit for bit under the same mapping and transform. The Python surface is migrated with a `BREAKING CHANGE:` footer.
