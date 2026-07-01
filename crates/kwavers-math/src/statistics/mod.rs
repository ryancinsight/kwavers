//! Statistical quality metrics and distribution summaries.
//!
//! Pure functions with no domain dependencies. Both solver and clinical layers
//! import from here; neither depends on the other.

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

/// Pearson correlation of same-frequency sinusoids separated by phase.
///
/// For `A(t) = sin(omega t)` and `B(t) = sin(omega t + phi)`, the Pearson
/// coefficient is `cos(phi)`. Inputs are phase offsets in radians.
///
/// # Errors
///
/// Returns an error when any phase value is non-finite.
pub fn phase_shift_correlation_curve(phase_rad: &[f64]) -> Result<Vec<f64>, String> {
    if let Some((index, value)) = phase_rad
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!("phase_rad[{index}] must be finite, got {value}"));
    }

    Ok(phase_rad.iter().map(|phase| phase.cos()).collect())
}

/// Phase error in degrees for a target same-frequency sinusoid correlation.
///
/// This is the inverse of `r(phi) = cos(phi)` over `0 <= phi <= pi`.
///
/// # Errors
///
/// Returns an error when `correlation` is non-finite or outside `[-1, 1]`.
pub fn phase_error_degrees_for_correlation(correlation: f64) -> Result<f64, String> {
    if !correlation.is_finite() {
        return Err(format!("correlation must be finite, got {correlation}"));
    }
    if !(-1.0..=1.0).contains(&correlation) {
        return Err(format!("correlation must be in [-1, 1], got {correlation}"));
    }

    Ok(correlation.acos().to_degrees())
}

/// PSNR in dB from relative RMSE values.
///
/// For `relative_rmse = RMSE / MAX`, `PSNR = -20 * log10(relative_rmse)`.
///
/// # Errors
///
/// Returns an error when any relative RMSE is non-finite or non-positive.
pub fn validation_psnr_from_relative_rmse(relative_rmse: &[f64]) -> Result<Vec<f64>, String> {
    if let Some((index, value)) = relative_rmse
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value <= 0.0)
    {
        return Err(format!(
            "relative_rmse[{index}] must be finite and positive, got {value}"
        ));
    }

    Ok(relative_rmse
        .iter()
        .map(|value| -20.0 * value.log10())
        .collect())
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

/// Absolute root-mean-square error between `a` and `b`,
/// `RMSE = √(mean((aᵢ − bᵢ)²))`.
///
/// Returns 0 when the slices differ in length or are empty.
#[must_use]
pub fn rmse(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mse = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64;
    mse.sqrt()
}

/// Peak signal-to-noise ratio in dB between simulation `a` and reference `b`
/// (Chapter 19 §19.3):
///
/// `PSNR = 20·log₁₀(MAX_B / RMSE(a, b))`, where `MAX_B = max|bᵢ|` is the peak
/// magnitude of the reference `b` (Chapter 19 §19.3 — the absolute value handles
/// bipolar signals such as acoustic pressure).
///
/// Returns `f64::INFINITY` for an exact match (`RMSE = 0`, infinite fidelity) and
/// `0.0` for degenerate inputs (length mismatch, empty, or an all-zero reference
/// for which the dB ratio is undefined).
#[must_use]
pub fn psnr(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let max_b = b.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    if max_b <= 0.0 {
        return 0.0;
    }
    let err = rmse(a, b);
    if err == 0.0 {
        return f64::INFINITY;
    }
    20.0 * (max_b / err).log10()
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
    fn rmse_matches_definition() {
        // diffs [0, 0, 1] ⇒ MSE = 1/3 ⇒ RMSE = √(1/3).
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 4.0];
        assert!((rmse(&a, &b) - (1.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(rmse(&a, &a), 0.0);
        assert_eq!(rmse(&[1.0, 2.0], &[1.0]), 0.0); // length mismatch
        assert_eq!(rmse(&[], &[]), 0.0);
    }

    #[test]
    fn psnr_matches_definition_and_limits() {
        // MAX_B = 4, RMSE = √(1/3) ⇒ PSNR = 20·log₁₀(4/√(1/3)).
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 4.0];
        let expected = 20.0 * (4.0 / (1.0_f64 / 3.0).sqrt()).log10();
        assert!(
            (psnr(&a, &b) - expected).abs() < 1e-12,
            "psnr = {}",
            psnr(&a, &b)
        );
        // Exact match ⇒ infinite fidelity.
        assert!(psnr(&a, &a).is_infinite());
        // A 40 dB target corresponds to RMSE = MAX_B / 100: peak 1.0, err 0.01.
        let r = vec![1.0; 100];
        let mut s = r.clone();
        s[0] = 1.0 - 0.01 * (100.0_f64).sqrt(); // single-sample error giving RMSE = 0.01
        assert!(
            (psnr(&s, &r) - 40.0).abs() < 1e-9,
            "psnr = {}",
            psnr(&s, &r)
        );
        // MAX_B uses peak magnitude, so a bipolar reference works: peak |−4| = 4.
        let bi_ref = [1.0, -4.0, 2.0];
        let bi_sim = [1.0, -4.0, 2.0];
        assert!(psnr(&bi_sim, &bi_ref).is_infinite());
        let bi_err = [1.0, -3.0, 2.0]; // err only at the |−4| sample
        assert!(psnr(&bi_err, &bi_ref) > 0.0 && psnr(&bi_err, &bi_ref).is_finite());
        // Degenerate guards.
        assert_eq!(psnr(&[1.0, 2.0], &[1.0]), 0.0); // length mismatch
        assert_eq!(psnr(&[0.0, 0.0], &[0.0, 0.0]), 0.0); // all-zero reference
    }

    /// §19.2 phase-sensitivity theorem: for A = sin(kx), B = sin(kx + φ) the
    /// Pearson correlation equals cos(φ).
    #[test]
    fn pearson_equals_cosine_of_phase_shift() {
        use std::f64::consts::PI;
        let n = 2000;
        let k = 2.0 * PI / n as f64; // one period over the window
        for &phi in &[0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI] {
            let a: Vec<f64> = (0..n).map(|i| (k * i as f64).sin()).collect();
            let b: Vec<f64> = (0..n).map(|i| (k * i as f64 + phi).sin()).collect();
            assert!(
                (pearson(&a, &b) - phi.cos()).abs() < 1e-3,
                "phase {phi}: r={} vs cos={}",
                pearson(&a, &b),
                phi.cos()
            );
        }
    }

    #[test]
    fn phase_shift_correlation_curve_matches_theorem_samples() {
        use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4};
        let phase = [0.0, FRAC_PI_4, FRAC_PI_2];
        let observed = phase_shift_correlation_curve(&phase).unwrap();
        let expected = [1.0, FRAC_1_SQRT_2, 0.0];

        for (actual, expected) in observed.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 2.0 * f64::EPSILON,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn phase_shift_correlation_curve_rejects_nonfinite_phase() {
        let err = phase_shift_correlation_curve(&[0.0, f64::NAN]).unwrap_err();
        assert!(err.contains("phase_rad[1] must be finite"));
    }

    #[test]
    fn phase_error_degrees_for_correlation_matches_inverse_theorem() {
        let observed = phase_error_degrees_for_correlation(0.99).unwrap();
        let expected = 0.99_f64.acos().to_degrees();

        assert!((observed - expected).abs() <= f64::EPSILON);
        assert_eq!(phase_error_degrees_for_correlation(1.0).unwrap(), 0.0);
        assert_eq!(phase_error_degrees_for_correlation(-1.0).unwrap(), 180.0);
    }

    #[test]
    fn phase_error_degrees_for_correlation_rejects_invalid_correlation() {
        let high = phase_error_degrees_for_correlation(1.01).unwrap_err();
        assert!(high.contains("correlation must be in [-1, 1]"));

        let nonfinite = phase_error_degrees_for_correlation(f64::INFINITY).unwrap_err();
        assert!(nonfinite.contains("correlation must be finite"));
    }

    #[test]
    fn validation_psnr_from_relative_rmse_matches_definition() {
        let observed = validation_psnr_from_relative_rmse(&[1.0, 0.1, 0.01, 0.001]).unwrap();
        let expected = [0.0, 20.0, 40.0, 60.0];

        for (actual, expected) in observed.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 8.0 * f64::EPSILON,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn validation_psnr_from_relative_rmse_rejects_invalid_error() {
        let zero = validation_psnr_from_relative_rmse(&[0.0]).unwrap_err();
        assert!(zero.contains("relative_rmse[0] must be finite and positive"));

        let nonfinite = validation_psnr_from_relative_rmse(&[f64::NAN]).unwrap_err();
        assert!(nonfinite.contains("relative_rmse[0] must be finite and positive"));
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
