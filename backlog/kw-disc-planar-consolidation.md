<a id="kw-disc-planar-consolidation"></a>

## KW-DISC-PLANAR-CONSOLIDATION — One disc element in the k-Wave array [arch] — todo

priority: architecture; needs: KW-RECT-PLANAR-CONSOLIDATION; scope: crates/kwavers-transducer/src/kwave_array, crates/kwavers-python (add_disc_element), docs/adr

- **Finding.** `ElementShape::Disc` and `PlanarApertureShape::Disk` are the same duplicate pair that ADR 136 resolves for rectangles. They have two point generators, two deposition rules, and different `set_array_position` handling.
- **Next step.** Check k-wave-python's `disc` branch (`get_off_grid_points`, `make_cart_disc`) for its transform and deposition contract. Then fold `Disc` into a `PlanarAperture` `Disk` under `SurfaceConvention::KWave`, recording the outcome as a revision of ADR 136.
- **Acceptance.** One disc representation. Existing disc mask and k-Wave reference tests keep their values. A one-sample disc sits at its centre.
