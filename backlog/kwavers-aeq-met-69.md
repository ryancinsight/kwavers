<a id="kwavers-aeq-met-69"></a>

## KWAVERS-AEQ-MET-69 — Type B-mode scan-conversion geometry [major] [arch] — todo

priority: architecture; needs: none; basis: a3a72689be0fbfed3f64231c52186eac79dedc15; scope: `crates/kwavers-analysis/src/signal_processing/b_mode/`, direct tests, ADR 105; milestone: M-imaging-capability

- **Outcome:** beam angles use Aequitas `Angle` and apex/range/grid extents use `Length`. Conversion to radians/metres is confined to the trigonometric, interpolation-index and Cartesian-raster formula boundaries; no compatibility fields or forwarding constructors remain. Dense RF/image arrays stay scalar numerical storage.
- **Eunomia:** B-mode geometry is real-valued; a complex RF representation at an upstream signal boundary keeps the existing observable unit, with no imaginary angle or length unit.
- **Acceptance:** all direct constructors and tests compile against typed geometry; the bright-pixel and out-of-sector scan-conversion regressions keep their value semantics; package check, strict Clippy, Nextest, doctests, Rustdoc, formatting and raw-public-signature scans pass.
- **TGC migration, 2026-11-22.** The constructor validates the attenuation
  slope, both frequencies and velocity. Point queries return length and
  dimensionless gain. Signed echoes exercise the dB-domain oracle and both
  scalar precisions exercise the in-place path. The scope of ADR 105 is
  geometry and physical TGC calibration. Display compression still takes a
  scalar dB range. See the [gap audit](../gap_audit.md) for pending work.
