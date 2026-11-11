<a id="kwavers-aeq-met-69"></a>

## KWAVERS-AEQ-MET-69 — Type B-mode scan-conversion geometry [major] [arch] — todo

priority: architecture; needs: none; basis: a3a72689be0fbfed3f64231c52186eac79dedc15; scope: `crates/kwavers-analysis/src/signal_processing/b_mode/`, direct tests, ADR 105

- **Outcome:** beam angles use Aequitas `Angle` and apex/range/grid extents use `Length`. Conversion to radians/metres is confined to the trigonometric, interpolation-index and Cartesian-raster formula boundaries; no compatibility fields or forwarding constructors remain. Dense RF/image arrays stay scalar numerical storage.
- **Eunomia:** B-mode geometry is real-valued; a complex RF representation at an upstream signal boundary keeps the existing observable unit, with no imaginary angle or length unit.
- **Acceptance:** all direct constructors and tests compile against typed geometry; the bright-pixel and out-of-sector scan-conversion regressions keep their value semantics; package check, strict Clippy, Nextest, doctests, Rustdoc, formatting and raw-public-signature scans pass.
- **Open clause.** "raw-public-signature scans pass": in the scoped directory, `tgc.rs` still exposes physical quantities as raw `f64`: the public fields `attenuation_db_cm_mhz` (line 32), `frequency_mhz` (line 34), `sound_speed` (line 36) and `sampling_rate` (line 38), and `pub fn depth_m(&self, i: usize) -> f64` (line 44). A closure note of 2026-10-28 recorded the scan-conversion types as typed and the package gates as green; it was not re-run here and did not cover `tgc.rs`.
- **Next step:** type the time-gain parameters and `depth_m` with Aequitas quantities and update their callers and tests; re-run the package gates and the scan; then close.
