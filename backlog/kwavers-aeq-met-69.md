<a id="kwavers-aeq-met-69"></a>

## KWAVERS-AEQ-MET-69 — Type B-mode scan-conversion geometry [major] [arch] — in-progress 2026-08-06

priority: architecture; needs: none; scope: `crates/kwavers-analysis/src/signal_processing/b_mode/`, direct tests, ADR 105

- **Outcome:** beam angles use Aequitas `Angle` and apex/range/grid extents use `Length`. Conversion to radians/metres is confined to the trigonometric, interpolation-index and Cartesian-raster formula boundaries; no compatibility fields or forwarding constructors remain. Dense RF/image arrays stay scalar numerical storage.
- **Eunomia:** B-mode geometry is real-valued; a complex RF representation at an upstream signal boundary keeps the existing observable unit, with no imaginary angle or length unit.
- **Acceptance:** all direct constructors and tests compile against typed geometry; the bright-pixel and out-of-sector scan-conversion regressions keep their value semantics; package check, strict Clippy, Nextest, doctests, Rustdoc, formatting and raw-public-signature scans pass.
