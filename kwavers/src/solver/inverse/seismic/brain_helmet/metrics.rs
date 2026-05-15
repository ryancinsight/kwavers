//! Reconstruction quality metrics for brain-helmet FWI.
//!
//! Single authoritative implementations used by both the 2-D Born slice solver
//! and the 3-D volumetric reconstruction. Delegates to the canonical Pearson
//! implementation in `clinical::therapy::theranostic_guidance::metrics` would
//! create an upward dependency; these implementations remain here as the SSOT
//! for the brain-helmet solver boundary.

/// Pearson correlation coefficient between two slices of equal length.
///
/// Returns 0 when either slice has zero variance or fewer than 2 elements.
pub(super) fn pearson(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        return 0.0;
    }
    let ma = a.iter().sum::<f64>() / a.len() as f64;
    let mb = b.iter().sum::<f64>() / b.len() as f64;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for (&av, &bv) in a.iter().zip(b) {
        let xa = av - ma;
        let xb = bv - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if da > 0.0 && db > 0.0 {
        num / (da.sqrt() * db.sqrt())
    } else {
        0.0
    }
}

/// RMSE of `a` normalised by `‖a‖₂`.
///
/// Returns 0 when `a` is the zero vector.
pub(super) fn normalized_rmse(a: &[f64], b: &[f64]) -> f64 {
    let norm = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm == 0.0 {
        return 0.0;
    }
    let err = a
        .iter()
        .zip(b)
        .map(|(&av, &bv)| {
            let d = av - bv;
            d * d
        })
        .sum::<f64>()
        .sqrt();
    err / norm
}

/// Inter-percentile range P95 − P05 of a value distribution.
///
/// Returns 0 for fewer than 2 samples.
pub(super) fn percentile_range(mut values: Vec<f64>) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let last = values.len() - 1;
    let p05 = values[(0.05 * last as f64).round() as usize];
    let p95 = values[(0.95 * last as f64).round() as usize];
    p95 - p05
}
