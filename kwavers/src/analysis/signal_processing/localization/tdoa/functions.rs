//! Free-standing time-delay estimation helpers.
//!
//! References:
//! - Knapp, C. H., & Carter, G. C. (1976). "The generalized correlation method for estimation of time delay"

/// Estimate time delay between two signals using direct cross-correlation.
///
/// Computes `R(τ) = Σ_t x[t]·y[t+τ]` over all valid lags and returns the
/// lag (in seconds) that maximises `R`.
pub(super) fn cross_correlation_delay(x: &[f64], y: &[f64], dt: f64) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }

    let n = x.len().min(y.len());
    let max_lag = n; // search full range

    let mut best_lag: isize = 0;
    let mut best_corr = f64::NEG_INFINITY;

    // Negative lags: y is shifted left relative to x
    // Positive lags: y is shifted right relative to x
    for lag in -(max_lag as isize)..=(max_lag as isize) {
        let mut sum = 0.0;
        let mut count = 0usize;
        for (t, &xv) in x.iter().enumerate().take(n) {
            let t2 = t as isize + lag;
            if t2 >= 0 && (t2 as usize) < y.len() {
                sum += xv * y[t2 as usize];
                count += 1;
            }
        }
        // Normalise by overlap length to avoid bias towards zero lag
        if count > 0 {
            let corr = sum / count as f64;
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }
    }

    // Quadratic interpolation for sub-sample accuracy
    let lag_f = subsample_refine(x, y, best_lag, n);
    lag_f * dt
}

/// GCC-PHAT style time-delay estimation (time-domain approximation).
///
/// Normalises the cross-correlation by the product of running RMS amplitudes
/// to approximate the Phase Transform (PHAT) weighting which whitens the
/// cross-spectral density and sharpens the correlation peak.
pub(super) fn gcc_phat_delay(x: &[f64], y: &[f64], dt: f64) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }

    let n = x.len().min(y.len());
    let max_lag = n;

    let x_energy: f64 = x.iter().take(n).map(|v| v * v).sum();
    let y_energy: f64 = y.iter().take(n).map(|v| v * v).sum();
    let norm = (x_energy * y_energy).sqrt().max(1e-30);

    let mut best_lag: isize = 0;
    let mut best_corr = f64::NEG_INFINITY;

    for lag in -(max_lag as isize)..=(max_lag as isize) {
        let mut sum = 0.0;
        for (t, &xv) in x.iter().enumerate().take(n) {
            let t2 = t as isize + lag;
            if t2 >= 0 && (t2 as usize) < y.len() {
                sum += xv * y[t2 as usize];
            }
        }
        let corr = sum / norm;
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    let lag_f = subsample_refine(x, y, best_lag, n);
    lag_f * dt
}

/// Quadratic (parabolic) interpolation around the peak lag for sub-sample
/// accuracy.  Returns the refined lag as a floating-point sample offset.
pub(super) fn subsample_refine(x: &[f64], y: &[f64], peak_lag: isize, n: usize) -> f64 {
    let corr_at = |lag: isize| -> f64 {
        let mut sum = 0.0;
        for (t, &xv) in x.iter().enumerate().take(n) {
            let t2 = t as isize + lag;
            if t2 >= 0 && (t2 as usize) < y.len() {
                sum += xv * y[t2 as usize];
            }
        }
        sum
    };

    let r0 = corr_at(peak_lag);
    let rm = corr_at(peak_lag - 1);
    let rp = corr_at(peak_lag + 1);

    let denom = 2.0 * (2.0 * r0 - rm - rp);
    if denom.abs() > 1e-30 {
        peak_lag as f64 + (rp - rm) / denom
    } else {
        peak_lag as f64
    }
}
