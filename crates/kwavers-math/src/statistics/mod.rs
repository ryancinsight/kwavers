//! Statistical quality metrics and distribution summaries.
//!
//! Pure functions with no domain dependencies. Both solver and clinical layers
//! import from here; neither depends on the other.

mod special;

pub use special::erf;

/// Pearson product-moment correlation coefficient.
///
/// Returns 0 when either slice has zero variance, when the slices have
/// different lengths, or when fewer than 2 samples are provided.
#[must_use]
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
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

/// RMSE of `b` relative to `a`, normalised by `‖a‖₂`.
///
/// Measures error energy relative to the reference signal energy.
/// Returns 0 when `a` is the zero vector.
#[must_use]
pub fn normalized_rmse(a: &[f64], b: &[f64]) -> f64 {
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

/// RMSE of `b` relative to `a`, normalised by the dynamic range of `a`.
///
/// Measures error relative to the signal's peak-to-peak span.
/// Returns 0 when slices differ in length, are empty, or `a` has zero range.
#[must_use]
pub fn nrmse(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mse = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64;
    let span = a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - a.iter().copied().fold(f64::INFINITY, f64::min);
    mse.sqrt() / span.abs().max(1.0e-12)
}

/// Inter-percentile range P95 − P05 of a value distribution.
///
/// Returns 0 for fewer than 2 samples.
#[must_use]
pub fn percentile_range(mut values: Vec<f64>) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let last = values.len() - 1;
    let p05 = values[(0.05 * last as f64).round() as usize];
    let p95 = values[(0.95 * last as f64).round() as usize];
    p95 - p05
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_perfect_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson(&a, &b) - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn pearson_anticorrelated() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![3.0, 2.0, 1.0];
        assert!((pearson(&a, &b) + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn pearson_constant_returns_zero() {
        let a = vec![1.0, 1.0, 1.0];
        let b = vec![2.0, 3.0, 4.0];
        assert_eq!(pearson(&a, &b), 0.0);
    }

    #[test]
    fn pearson_length_mismatch_returns_zero() {
        assert_eq!(pearson(&[1.0, 2.0], &[1.0]), 0.0);
    }

    #[test]
    fn normalized_rmse_perfect_match() {
        let a = vec![1.0, 2.0, 3.0];
        assert_eq!(normalized_rmse(&a, &a), 0.0);
    }

    #[test]
    fn normalized_rmse_zero_reference() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(normalized_rmse(&a, &b), 0.0);
    }

    #[test]
    fn nrmse_perfect_match() {
        let a = vec![1.0, 2.0, 3.0];
        assert_eq!(nrmse(&a, &a), 0.0);
    }

    #[test]
    fn percentile_range_monotone() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let r = percentile_range(v);
        assert!(r > 0.0 && r <= 9.0);
    }

    #[test]
    fn percentile_range_single_element() {
        assert_eq!(percentile_range(vec![42.0]), 0.0);
    }
}
