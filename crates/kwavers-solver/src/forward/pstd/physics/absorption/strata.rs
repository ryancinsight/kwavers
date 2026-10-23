//! Stratified fractional-Laplacian operator for **spatially-varying** power-law
//! exponent y(x) — going beyond the single-global-exponent limitation of k-Wave
//! (and of the uniform path in [`super::init`]).
//!
//! ## The problem
//! The Treeby & Cox (2010) absorption term needs the fractional Laplacians
//! `∇^(y−2)` and `∇^(y−1)`, applied spectrally as `IFFT(|k|^(y−s)·FFT(·))`. The
//! symbol `|k|^(y−s)` depends on y, so a *single* spectral operator can only
//! represent one global exponent. k-Wave, and the uniform path here, therefore
//! force one y for the whole domain — wrong for a CT body model where soft tissue
//! (y ≈ 1.1) and bone (y ≈ 1.0) coexist.
//!
//! ## Stratification
//! We represent the operator at a small set of M representative exponents
//! `y_0 < … < y_{M−1}` (the distinct exponents present, or an `MAX_STRATA`-point
//! linspace when there are many), each with its own spectral symbol. Each voxel
//! is bracketed between two adjacent strata `[m, m+1]` with weight `t`, so the
//! effective operator is the partition-of-unity blend
//! `L_eff[f](x) = (1−t)·L(y_m)[f](x) + t·L(y_{m+1})[f](x)`.
//! This is **exact** wherever y(x) equals a stratum exponent (every distinct
//! tissue exponent when M = #distinct), and the linear blend bounds the error by
//! the convexity gap of `s ↦ |k|^s` over one bin elsewhere.
//!
//! ## Cost
//! The apply path shares the forward FFT structure and runs M inverse transforms
//! per Laplacian (vs 1 for the uniform path); strata are built only when y(x)
//! does not collapse to one exponent (see `collapses_to_uniform`), so lossless
//! and single-exponent media pay nothing.
//! Memory is M half-spectrum symbol pairs plus a compact per-voxel
//! `(u32 index, f64 weight)` — not M full per-stratum masks.
//!
//! ## References
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314, Eqs. 9–10, 19–21.

use super::kernel::ExponentSpectrum;
use crate::forward::pstd::config::PowerLawDispersion;
use kwavers_core::constants::ABSORPTION_SINGULARITY_THRESHOLD;
use leto::Array3;
use moirai_parallel::{map_collect_index_with, Adaptive};

/// Maximum number of exponent strata. Caps the per-step inverse-FFT count and
/// the symbol memory; a CT body model spans y ∈ [1.0, 1.1], so an 8-point
/// linspace bounds the per-bin exponent width to ≈ 0.014.
pub(crate) const MAX_STRATA: usize = 8;

/// Exponents within this absolute tolerance are treated as identical (so a
/// nominally homogeneous medium maps to the uniform path, not a 1-stratum set).
///
/// A field that collapses to the midpoint of its range moves no voxel's
/// exponent by more than `TOL/2`. The symbol `|k|^(y−s)` then
/// changes by the relative factor `|k|^(±TOL/2) − 1 ≈ (TOL/2)·|ln |k||`. The
/// nonzero bins of a grid with spacing above 1 nm and extent below 1000 km lie
/// in `2π/10⁶ < |k| ≤ √3·π/10⁻⁹` rad/m (the 3-D corner bin), so
/// `|ln |k|| < 22.5` and the change stays below `1.125e-5`; the thermal
/// coupling's `ω_c^y` moves by `(TOL/2)·ln ω_c`.
/// Causal dispersion adds `tan(πy/2)`, whose sensitivity is unbounded near
/// `y = 1`; [`collapses_to_uniform`] admits the midpoint only where it holds
/// that factor to the same bound.
pub(crate) const STRATA_UNIFORM_TOL: f64 = 1e-6;

/// Relative change every exponent-dependent factor must stay within when a
/// field collapses to its midpoint: the `|k|^(y−s)` bound above,
/// `(TOL/2)·22.5`.
const UNIFORM_RELATIVE_BOUND: f64 = 11.25 * STRATA_UNIFORM_TOL;

/// Whether an exponent field spanning `[y_min, y_max]` may take its midpoint
/// everywhere.
///
/// The range must be within [`STRATA_UNIFORM_TOL`]. Under causal dispersion
/// the dispersion factor `tan(πy/2)` must also move by no more than
/// [`UNIFORM_RELATIVE_BOUND`]: moving `y` by `d` changes it by the relative
/// factor `π·d/|sin(πy)|` to first order, which at `d = (y_max − y_min)/2`
/// grows without bound as the midpoint nears `y = 1`. A field straddling
/// `y = 1` therefore keeps each voxel's own exponent.
fn collapses_to_uniform(y_min: f64, y_max: f64, dispersion: PowerLawDispersion) -> bool {
    let range = y_max - y_min;
    if range > STRATA_UNIFORM_TOL {
        return false;
    }
    match dispersion {
        PowerLawDispersion::Omitted => true,
        PowerLawDispersion::Causal => {
            let midpoint = 0.5 * (y_min + y_max);
            0.5 * std::f64::consts::PI * range
                <= UNIFORM_RELATIVE_BOUND * (std::f64::consts::PI * midpoint).sin().abs()
        }
    }
}

#[inline]
fn dense_indices(index: usize, ny: usize, nz: usize) -> (usize, usize, usize) {
    let plane = ny * nz;
    let i = index / plane;
    let rem = index % plane;
    let j = rem / nz;
    let k = rem % nz;
    (i, j, k)
}

fn assign_stratum_brackets(
    bracket_lo: &mut Array3<u32>,
    weight_hi: &mut Array3<f64>,
    y_field: &Array3<f64>,
    exponents: &[f64],
) {
    assert_eq!(
        bracket_lo.shape(),
        weight_hi.shape(),
        "invariant: stratum bracket and weight shapes match"
    );
    assert_eq!(
        bracket_lo.shape(),
        y_field.shape(),
        "invariant: stratum bracket shape matches exponent field"
    );

    let m_count = exponents.len();
    let assign = |y: f64| -> (u32, f64) {
        // Largest m with exponents[m] <= y, clamped so m+1 is in range.
        let m = match exponents.binary_search_by(|e| e.partial_cmp(&y).expect("finite exponent")) {
            Ok(i) => i.min(m_count - 2),
            Err(i) => i.saturating_sub(1).min(m_count - 2),
        };
        let denom = exponents[m + 1] - exponents[m];
        let weight = if denom > 0.0 {
            ((y - exponents[m]) / denom).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (m as u32, weight)
    };

    if let (Some(lo_values), Some(weight_values), Some(y_values)) = (
        bracket_lo.as_slice_mut(),
        weight_hi.as_slice_mut(),
        y_field.as_slice(),
    ) {
        let assignments = map_collect_index_with::<Adaptive, _, _>(y_values.len(), |index| {
            assign(y_values[index])
        });
        for (index, (lo, weight)) in assignments.into_iter().enumerate() {
            lo_values[index] = lo;
            weight_values[index] = weight;
        }
        return;
    }

    let [nx, ny, nz] = bracket_lo.shape();
    for index in 0..(nx * ny * nz) {
        let (i, j, k) = dense_indices(index, ny, nz);
        let (lo, weight) = assign(y_field[[i, j, k]]);
        bracket_lo[[i, j, k]] = lo;
        weight_hi[[i, j, k]] = weight;
    }
}

/// Spectral operators for a spatially-varying power-law exponent, blended
/// per-voxel between adjacent exponent strata.
pub(crate) struct ExponentStrata {
    /// Sorted representative exponents, `len == M ≥ 2`.
    pub exponents: Vec<f64>,
    /// Per-stratum spectral symbol `|k|^(y_m − 2)`, half-spectrum `(nx, ny, nz_c)`.
    pub nabla1: Vec<Array3<f64>>,
    /// Per-stratum spectral symbol `|k|^(y_m − 1)`, half-spectrum.
    pub nabla2: Vec<Array3<f64>>,
    /// Per-voxel lower-bracket stratum index `m` with
    /// `exponents\[m\] ≤ y(voxel) ≤ exponents[m+1]`.
    pub bracket_lo: Array3<u32>,
    /// Per-voxel blend weight `t` toward the upper bracket, so
    /// `y(voxel) = (1−t)·exponents\[m\] + t·exponents[m+1]`.
    pub weight_hi: Array3<f64>,
}

impl ExponentStrata {
    /// Exponent the blend realizes at voxel `index`:
    /// `(1−t)·exponents[m] + t·exponents[m+1]`. It equals a stratum exponent
    /// exactly where `t ∈ {0, 1}` and the voxel's own exponent to rounding
    /// elsewhere.
    pub(crate) fn exponent_at(&self, index: [usize; 3]) -> f64 {
        let m = usize::try_from(self.bracket_lo[index])
            .expect("invariant: a stratum index below MAX_STRATA fits usize");
        let t = self.weight_hi[index];
        (1.0 - t) * self.exponents[m] + t * self.exponents[m + 1]
    }
}

/// Half-spectrum (`z ∈ [0, nz/2]`) symbol `|k|^power`, zero at `|k|` below
/// [`ABSORPTION_SINGULARITY_THRESHOLD`], from the full-spectrum wavenumber
/// magnitude `k_mag` in FFT order. The r2c apply path needs only this half:
/// `|k|` is even in `kz`.
fn half_spectrum_symbol(k_mag: &Array3<f64>, power: f64) -> Array3<f64> {
    let nz_c = k_mag.shape()[2] / 2 + 1;
    let k_half = k_mag
        .slice_with(&s![.., .., ..nz_c])
        .expect("invariant: nz_c <= nz half-spectrum length");
    k_half.as_array().mapv(|k| {
        if k > ABSORPTION_SINGULARITY_THRESHOLD {
            k.powf(power)
        } else {
            0.0
        }
    })
}

/// The single-exponent spectrum: `|k|^(y−2)` and `|k|^(y−1)` on the half
/// spectrum (Treeby & Cox 2010 Eq. 10).
pub(crate) fn uniform_spectrum(exponent: f64, k_mag: &Array3<f64>) -> ExponentSpectrum {
    ExponentSpectrum::Uniform {
        exponent,
        nabla1: half_spectrum_symbol(k_mag, exponent - 2.0),
        nabla2: half_spectrum_symbol(k_mag, exponent - 1.0),
    }
}

/// Resolve a per-voxel exponent field into its spectrum: the uniform path at
/// the midpoint of the field's range when [`collapses_to_uniform`] admits it,
/// otherwise stratified operators that reproduce every voxel's exponent.
///
/// `k_mag` is the full-spectrum wavenumber magnitude in FFT order; the symbols
/// are stored half-spectrum (`z ∈ [0, nz/2]`), matching the r2c apply path.
pub(crate) fn resolve_exponent_spectrum(
    y_field: &Array3<f64>,
    k_mag: &Array3<f64>,
    dispersion: PowerLawDispersion,
) -> ExponentSpectrum {
    // Sorted exponent samples → min/max and distinct values.
    let mut ys: Vec<f64> = y_field.iter().copied().collect();
    ys.sort_by(|a, b| a.partial_cmp(b).expect("exponent field must be finite"));
    let y_min = ys[0];
    let y_max = *ys.last().expect("non-empty field");
    if collapses_to_uniform(y_min, y_max, dispersion) {
        return uniform_spectrum(0.5 * (y_min + y_max), k_mag);
    }

    let mut distinct: Vec<f64> = Vec::new();
    for &y in &ys {
        if distinct.last().is_none_or(|&p| y - p > STRATA_UNIFORM_TOL) {
            distinct.push(y);
        }
    }
    // The top representative is the field's maximum, so voxels merged into
    // it within the tolerance stay inside [exponents[0], exponents[M−1]] and
    // their blend reproduces their own exponent; a range below the
    // tolerance that may not collapse gets its two endpoints.
    if distinct.len() == 1 {
        distinct.push(y_max);
    } else if let Some(top) = distinct.last_mut() {
        *top = y_max;
    }

    // Few distinct exponents → represent them exactly; many → MAX_STRATA linspace.
    let exponents: Vec<f64> = if (distinct.len()) <= MAX_STRATA {
        distinct
    } else {
        (0..MAX_STRATA)
            .map(|m| y_min + (y_max - y_min) * (m as f64) / ((MAX_STRATA - 1) as f64))
            .collect()
    };
    // Half-spectrum symbols |k|^(y_m − 2) and |k|^(y_m − 1); DC bin → 0.
    let nabla1: Vec<Array3<f64>> = exponents
        .iter()
        .map(|&y| half_spectrum_symbol(k_mag, y - 2.0))
        .collect();
    let nabla2: Vec<Array3<f64>> = exponents
        .iter()
        .map(|&y| half_spectrum_symbol(k_mag, y - 1.0))
        .collect();

    // Per-voxel bracket [m, m+1] and blend weight t, reconstructing y exactly.
    let mut bracket_lo = Array3::<u32>::zeros(y_field.shape());
    let mut weight_hi = Array3::<f64>::zeros(y_field.shape());
    assign_stratum_brackets(&mut bracket_lo, &mut weight_hi, y_field, &exponents);

    ExponentSpectrum::Stratified(ExponentStrata {
        exponents,
        nabla1,
        nabla2,
        bracket_lo,
        weight_hi,
    })
}

#[cfg(test)]
mod tests {
    use super::{resolve_exponent_spectrum, ExponentStrata, MAX_STRATA, STRATA_UNIFORM_TOL};
    use crate::forward::pstd::config::PowerLawDispersion;
    use crate::forward::pstd::physics::absorption::kernel::ExponentSpectrum;
    use eunomia::FloatElement;
    use leto::Array3;

    fn stratified(y: &Array3<f64>, k: &Array3<f64>) -> ExponentStrata {
        stratified_with(y, k, PowerLawDispersion::Causal)
    }

    fn stratified_with(
        y: &Array3<f64>,
        k: &Array3<f64>,
        dispersion: PowerLawDispersion,
    ) -> ExponentStrata {
        match resolve_exponent_spectrum(y, k, dispersion) {
            ExponentSpectrum::Stratified(strata) => strata,
            ExponentSpectrum::Uniform { exponent, .. } => {
                panic!("non-uniform field resolved to the uniform exponent {exponent}")
            }
        }
    }

    /// A field whose range is within the tolerance resolves to the uniform
    /// path at the midpoint of that range, and its symbols carry that
    /// exponent: `|k|^(y−2)` and `|k|^(y−1)` evaluated by the same `powf`
    /// are reproduced exactly.
    #[test]
    fn near_uniform_exponent_resolves_to_the_range_midpoint() {
        let (low, high) = (1.5_f64, 1.5 + 0.5 * STRATA_UNIFORM_TOL);
        let y = Array3::from_shape_fn((4, 4, 4), |[i, _, _]| if i == 0 { low } else { high });
        let k = Array3::from_shape_fn((4, 4, 4), |[i, j, l]| {
            1.0 + 0.5 * f64::from_count(i + j + l)
        });
        let ExponentSpectrum::Uniform {
            exponent,
            nabla1,
            nabla2,
        } = resolve_exponent_spectrum(&y, &k, PowerLawDispersion::Causal)
        else {
            panic!("a range within the tolerance must take the uniform path");
        };
        let midpoint = 0.5 * (low + high);
        assert_eq!(exponent, midpoint);
        // Half spectrum: nz = 4 keeps z ∈ [0, 2].
        assert_eq!(nabla1.shape(), [4, 4, 3]);
        for ((index, &n1), &n2) in nabla1.indexed_iter().zip(nabla2.iter()) {
            let kv = k[index];
            assert_eq!(n1, kv.powf(midpoint - 2.0), "nabla1 at {index:?}");
            assert_eq!(n2, kv.powf(midpoint - 1.0), "nabla2 at {index:?}");
        }
    }

    /// A field straddling `y = 1` within the tolerance collapses without
    /// dispersion, but under causal dispersion keeps each voxel's exponent:
    /// its midpoint would put `tan(πy/2)` at its pole while every voxel's own
    /// value is finite and the two halves differ in sign.
    #[test]
    fn exponent_straddling_unity_collapses_only_without_dispersion() {
        let (low, high) = (
            1.0 - 0.4 * STRATA_UNIFORM_TOL,
            1.0 + 0.4 * STRATA_UNIFORM_TOL,
        );
        let y = Array3::from_shape_fn((4, 4, 4), |[i, _, _]| if i < 2 { low } else { high });
        let k = Array3::from_elem((4, 4, 4), 2.0);
        let ExponentSpectrum::Uniform { exponent, .. } =
            resolve_exponent_spectrum(&y, &k, PowerLawDispersion::Omitted)
        else {
            panic!("without dispersion the range within the tolerance collapses");
        };
        assert_eq!(exponent, 0.5 * (low + high));
        let strata = stratified_with(&y, &k, PowerLawDispersion::Causal);
        assert_eq!(strata.exponents, vec![low, high]);
        for (index, &own) in y.indexed_iter() {
            assert_eq!(strata.exponent_at(index), own, "voxel {index:?}");
        }
    }

    /// Voxels merged into the top representative within the tolerance keep
    /// their own exponent: the top representative is the field's maximum.
    #[test]
    fn exponents_above_the_top_representative_are_reproduced() {
        let top = 1.2 + 0.5 * STRATA_UNIFORM_TOL;
        let y = Array3::from_shape_fn((3, 2, 2), |[i, _, _]| [1.0, 1.2, top][i]);
        let k = Array3::from_elem((3, 2, 2), 2.0);
        let strata = stratified(&y, &k);
        assert_eq!(strata.exponents, vec![1.0, top]);
        for (index, &own) in y.indexed_iter() {
            let realized = strata.exponent_at(index);
            // (1−t)·1 + t·top with t = (own − 1)/(top − 1): a quotient, a
            // product and a sum, each half an ulp of a value at most `top`.
            assert!(
                (realized - own).abs() <= 3.0 * 0.5 * f64::EPSILON * top,
                "voxel {index:?}: {realized} against {own}"
            );
        }
    }

    /// Two distinct exponents → exactly two strata at those values, and each
    /// voxel's (bracket, weight) reconstructs its exponent exactly.
    #[test]
    fn two_tissue_field_reconstructs_exponents_exactly() {
        let (ya, yb) = (1.1_f64, 1.5_f64);
        let mut y = Array3::from_elem([4, 4, 4], ya);
        for i in 2..4 {
            for j in 0..4 {
                for k in 0..4 {
                    y[[i, j, k]] = yb;
                }
            }
        }
        // k_mag with nonzero magnitudes so symbols are well-defined.
        let k = Array3::from_shape_fn((4, 4, 4), |[i, j, l]| (1 + i + j + l) as f64);
        let s = stratified(&y, &k);

        assert_eq!((s.exponents.len()), 2);
        assert!((s.exponents[0] - ya).abs() < 1e-12);
        assert!((s.exponents[1] - yb).abs() < 1e-12);

        // Per-voxel exact reconstruction: (1−t)·e[lo] + t·e[lo+1] == y.
        for ((idx, &lo), &t) in s.bracket_lo.indexed_iter().zip(s.weight_hi.iter()) {
            let lo = lo as usize;
            let recon = (1.0 - t) * s.exponents[lo] + t * s.exponents[lo + 1];
            assert!(
                (recon - y[idx]).abs() < 1e-12,
                "voxel {idx:?}: recon {recon} != y {}",
                y[idx]
            );
        }

        // Symbols carry the Treeby & Cox powers |k|^(y−2), |k|^(y−1).
        let kv = k[[1, 1, 1]];
        let nz_c = k.shape()[2] / 2 + 1;
        assert!(1 < nz_c);
        assert!((s.nabla1[0][[1, 1, 1]] - kv.powf(ya - 2.0)).abs() < 1e-12 * kv.powf(ya - 2.0));
        assert!((s.nabla2[1][[1, 1, 1]] - kv.powf(yb - 1.0)).abs() < 1e-12 * kv.powf(yb - 1.0));
    }

    /// A continuum of exponents is capped at MAX_STRATA linspace bins spanning
    /// the full range, and every voxel still reconstructs exactly.
    #[test]
    fn continuum_is_capped_and_spans_range() {
        let (n,) = (16usize,);
        let y = Array3::from_shape_fn((n, 1, 1), |[i, _, _]| {
            1.0 + 0.1 * (i as f64) / ((n - 1) as f64)
        });
        let k = Array3::from_shape_fn((n, 1, 1), |[i, _, _]| (i + 1) as f64);
        let s = stratified(&y, &k);
        assert!((s.exponents.len()) <= MAX_STRATA && (s.exponents.len()) >= 2);
        assert!((s.exponents[0] - 1.0).abs() < 1e-9);
        assert!((s.exponents.last().unwrap() - 1.1).abs() < 1e-9);
        for ((idx, &lo), &t) in s.bracket_lo.indexed_iter().zip(s.weight_hi.iter()) {
            let lo = lo as usize;
            let recon = (1.0 - t) * s.exponents[lo] + t * s.exponents[lo + 1];
            assert!((recon - y[idx]).abs() < 1e-12);
        }
    }
}
