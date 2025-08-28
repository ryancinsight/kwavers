//! Comparison metrics for validation

use ndarray::{Array3, Zip};

/// Comparison metrics for field validation
pub struct ComparisonMetrics;

impl ComparisonMetrics {
    /// L2 norm error
    pub fn l2_error(computed: &Array3<f64>, reference: &Array3<f64>) -> f64 {
        let mut sum_squared_diff = 0.0;
        let mut sum_squared_ref = 0.0;

        Zip::from(computed).and(reference).for_each(|&c, &r| {
            let diff = c - r;
            sum_squared_diff += diff * diff;
            sum_squared_ref += r * r;
        });

        if sum_squared_ref > 0.0 {
            (sum_squared_diff / sum_squared_ref).sqrt()
        } else {
            sum_squared_diff.sqrt()
        }
    }

    /// L-infinity norm error (maximum absolute error)
    pub fn linf_error(computed: &Array3<f64>, reference: &Array3<f64>) -> f64 {
        let mut max_error = 0.0;

        Zip::from(computed).and(reference).for_each(|&c, &r| {
            max_error = f64::max(max_error, (c - r).abs());
        });

        max_error
    }

    /// Root mean square error
    pub fn rmse(computed: &Array3<f64>, reference: &Array3<f64>) -> f64 {
        let mut sum_squared = 0.0;
        let mut count = 0;

        Zip::from(computed).and(reference).for_each(|&c, &r| {
            let diff = c - r;
            sum_squared += diff * diff;
            count += 1;
        });

        (sum_squared / count as f64).sqrt()
    }

    /// Peak signal-to-noise ratio in dB
    pub fn psnr(computed: &Array3<f64>, reference: &Array3<f64>) -> f64 {
        let mse = Self::rmse(computed, reference).powi(2);
        let max_val = reference.iter().cloned().fold(f64::NEG_INFINITY, |a, b| a.max(b));

        if mse > 0.0 {
            20.0 * (max_val / mse.sqrt()).log10()
        } else {
            f64::INFINITY
        }
    }

    /// Structural similarity index (simplified version)
    pub fn ssim(computed: &Array3<f64>, reference: &Array3<f64>) -> f64 {
        const C1: f64 = 0.01;
        const C2: f64 = 0.03;

        let mean_c = computed.mean().unwrap_or(0.0);
        let mean_r = reference.mean().unwrap_or(0.0);

        let var_c =
            computed.iter().map(|&x| (x - mean_c).powi(2)).sum::<f64>() / computed.len() as f64;
        let var_r =
            reference.iter().map(|&x| (x - mean_r).powi(2)).sum::<f64>() / reference.len() as f64;

        let mut covar = 0.0;
        Zip::from(computed).and(reference).for_each(|&c, &r| {
            covar += (c - mean_c) * (r - mean_r);
        });
        covar /= computed.len() as f64;

        let numerator = (2.0 * mean_c * mean_r + C1) * (2.0 * covar + C2);
        let denominator = (mean_c.powi(2) + mean_r.powi(2) + C1) * (var_c + var_r + C2);

        numerator / denominator
    }
}
