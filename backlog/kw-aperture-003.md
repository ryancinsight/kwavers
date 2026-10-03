<a id="kw-aperture-003"></a>

## KW-APERTURE-003 — Planar sector BLI rasterization [minor] — review

priority: feature; needs: none; scope: `kwavers-transducer::kwave_array`, canonical planar aperture geometry

- **Driver:** a private downstream consumer's hybrid C/D sectors require full-wave PSTD sources without finite-disc substitution.
- **Acceptance:** validated oriented disk/annular-sector geometry rasterizes through the existing BLI per-element source path, conserves analytical aperture area, preserves independent element signals, and passes package gates. BLI rejects only sources beyond its finite window, preserving clipped apertures while preventing distant sinc-tail boundary injection.
- **Evidence:** warning-denied all-target/all-feature Clippy; Nextest 215/215 (one existing skip); exact per-quadrant analytical area and independent-signal regressions.
- **Next step:** re-validate at the default branch (`kwave_array/tests/elements.rs` holds the sector cases); close if present.
