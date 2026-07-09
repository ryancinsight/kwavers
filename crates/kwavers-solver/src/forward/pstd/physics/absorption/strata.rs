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
//! per Laplacian (vs 1 for the uniform path); strata are built only when y(x) is
//! genuinely non-uniform, so lossless and single-exponent media pay nothing.
//! Memory is M half-spectrum symbol pairs plus a compact per-voxel
//! `(u32 index, f64 weight)` — not M full per-stratum masks.
//!
//! ## References
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314, Eqs. 9–10, 19–21.

use kwavers_core::constants::ABSORPTION_SINGULARITY_THRESHOLD;
use moirai_parallel::{map_collect_index_with, Adaptive};
use leto::{
    /* s -- no leto equivalent */,
    Array3,
};

/// Maximum number of exponent strata. Caps the per-step inverse-FFT count and
/// the symbol memory; a CT body model spans y ∈ [1.0, 1.1], so an 8-point
/// linspace bounds the per-bin exponent width to ≈ 0.014.
pub(crate) const MAX_STRATA: usize = 8;

/// Exponents within this absolute tolerance are treated as identical (so a
/// nominally homogeneous medium maps to the uniform path, not a 1-stratum set).
pub(crate) const STRATA_UNIFORM_TOL: f64 = 1e-6;

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
        bracket_lo.as_slice_memory_order_mut(),
        weight_hi.as_slice_memory_order_mut(),
        y_field.as_slice_memory_order(),
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

    let (nx, ny, nz) = bracket_lo.dim();
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
    /// `exponents[m] ≤ y(voxel) ≤ exponents[m+1]`.
    pub bracket_lo: Array3<u32>,
    /// Per-voxel blend weight `t` toward the upper bracket, so
    /// `y(voxel) = (1−t)·exponents[m] + t·exponents[m+1]`.
    pub weight_hi: Array3<f64>,
}

/// Build stratified spectral operators from a per-voxel exponent field, or
/// `None` when the exponent is uniform (within [`STRATA_UNIFORM_TOL`]) — in
/// which case the caller keeps the cheaper uniform single-symbol path.
///
/// `k_mag` is the full-spectrum wavenumber magnitude in FFT order; the symbols
/// are stored half-spectrum (`z ∈ [0, nz/2]`), matching the r2c apply path.
pub(crate) fn build_exponent_strata(
    y_field: &Array3<f64>,
    k_mag: &Array3<f64>,
) -> Option<ExponentStrata> {
    // Sorted exponent samples → min/max and distinct values.
    let mut ys: Vec<f64> = y_field.iter().copied().collect();
    ys.sort_by(|a, b| a.partial_cmp(b).expect("exponent field must be finite"));
    let y_min = ys[0];
    let y_max = *ys.last().expect("non-empty field");
    if y_max - y_min <= STRATA_UNIFORM_TOL {
        return None; // uniform exponent → uniform path
    }

    let mut distinct: Vec<f64> = Vec::new();
    for &y in &ys {
        if distinct.last().is_none_or(|&p| y - p > STRATA_UNIFORM_TOL) {
            distinct.push(y);
        }
    }

    // Few distinct exponents → represent them exactly; many → MAX_STRATA linspace.
    let exponents: Vec<f64> = if distinct.len() <= MAX_STRATA {
        distinct
    } else {
        (0..MAX_STRATA)
            .map(|m| y_min + (y_max - y_min) * (m as f64) / ((MAX_STRATA - 1) as f64))
            .collect()
    };
    // Half-spectrum symbols |k|^(y_m − 2) and |k|^(y_m − 1); DC bin → 0.
    let nz_c = k_mag.dim().2 / 2 + 1;
    let k_half = k_mag.slice(s![.., .., ..nz_c]);
    let symbol = |power: f64| -> Array3<f64> {
        k_half.mapv(|k| {
            if k > ABSORPTION_SINGULARITY_THRESHOLD {
                k.powf(power)
            } else {
                0.0
            }
        })
    };
    let nabla1: Vec<Array3<f64>> = exponents.iter().map(|&y| symbol(y - 2.0)).collect();
    let nabla2: Vec<Array3<f64>> = exponents.iter().map(|&y| symbol(y - 1.0)).collect();

    // Per-voxel bracket [m, m+1] and blend weight t, reconstructing y exactly.
    let mut bracket_lo = Array3::<u32>::zeros(y_field.dim());
    let mut weight_hi = Array3::<f64>::zeros(y_field.dim());
    assign_stratum_brackets(&mut bracket_lo, &mut weight_hi, y_field, &exponents);

    Some(ExponentStrata {
        exponents,
        nabla1,
        nabla2,
        bracket_lo,
        weight_hi,
    })
}

#[cfg(test)]
mod tests {
    use super::{build_exponent_strata, MAX_STRATA};
    use leto::Array3;

    /// A uniform exponent field yields no strata (caller uses the uniform path).
    #[test]
    fn uniform_exponent_returns_none() {
        let y = Array3::from_elem((4, 4, 4), 1.5);
        let k = Array3::from_elem((4, 4, 4), 1.0);
        assert!(build_exponent_strata(&y, &k).is_none());
    }

    /// Two distinct exponents → exactly two strata at those values, and each
    /// voxel's (bracket, weight) reconstructs its exponent exactly.
    #[test]
    fn two_tissue_field_reconstructs_exponents_exactly() {
        let (ya, yb) = (1.1_f64, 1.5_f64);
        let mut y = Array3::from_elem((4, 4, 4), ya);
        for i in 2..4 {
            for j in 0..4 {
                for k in 0..4 {
                    y[[i, j, k]] = yb;
                }
            }
        }
        // k_mag with nonzero magnitudes so symbols are well-defined.
        let k = Array3::from_shape_fn((4, 4, 4), |(i, j, l)| (1 + i + j + l) as f64);
        let s = build_exponent_strata(&y, &k).expect("non-uniform → strata");

        assert_eq!(s.exponents.len(), 2);
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
        let nz_c = k.dim().2 / 2 + 1;
        assert!(1 < nz_c);
        assert!((s.nabla1[0][[1, 1, 1]] - kv.powf(ya - 2.0)).abs() < 1e-12 * kv.powf(ya - 2.0));
        assert!((s.nabla2[1][[1, 1, 1]] - kv.powf(yb - 1.0)).abs() < 1e-12 * kv.powf(yb - 1.0));
    }

    /// A continuum of exponents is capped at MAX_STRATA linspace bins spanning
    /// the full range, and every voxel still reconstructs exactly.
    #[test]
    fn continuum_is_capped_and_spans_range() {
        let (n,) = (16usize,);
        let y = Array3::from_shape_fn((n, 1, 1), |(i, _, _)| {
            1.0 + 0.1 * (i as f64) / ((n - 1) as f64)
        });
        let k = Array3::from_shape_fn((n, 1, 1), |(i, _, _)| (i + 1) as f64);
        let s = build_exponent_strata(&y, &k).expect("non-uniform");
        assert!(s.exponents.len() <= MAX_STRATA && s.exponents.len() >= 2);
        assert!((s.exponents[0] - 1.0).abs() < 1e-9);
        assert!((s.exponents.last().unwrap() - 1.1).abs() < 1e-9);
        for ((idx, &lo), &t) in s.bracket_lo.indexed_iter().zip(s.weight_hi.iter()) {
            let lo = lo as usize;
            let recon = (1.0 - t) * s.exponents[lo] + t * s.exponents[lo + 1];
            assert!((recon - y[idx]).abs() < 1e-12);
        }
    }
}
