//! Validation Framework for PINN Predictions
//!
//! Provides comprehensive validation metrics comparing PINN predictions
//! against FDTD reference solutions.
//!
//! ## Metrics
//!
//! - Mean Absolute Error (MAE)
//! - Root Mean Squared Error (RMSE)
//! - Relative L2 Error
//! - Maximum Pointwise Error
//! - Correlation Coefficient

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::math::ml::pinn::fdtd_reference::{FDTD1DWaveSolver, FDTDConfig};
use crate::domain::math::ml::pinn::wave_equation_1d::{PINN1DWave, ValidationMetrics};
use ndarray::{Array1, Array2};

/// Comprehensive validation results comparing PINN vs FDTD
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Standard validation metrics
    pub metrics: ValidationMetrics,
    /// Pearson correlation coefficient
    pub correlation: f64,
    /// Mean relative error (%)
    pub mean_relative_error_percent: f64,
    /// Number of points compared
    pub num_points: usize,
    /// FDTD solve time (seconds)
    pub fdtd_time_secs: f64,
    /// PINN inference time (seconds)
    pub pinn_time_secs: f64,
    /// Speedup factor (FDTD time / PINN time)
    pub speedup_factor: f64,
}

impl ValidationReport {
    /// Check if validation passes target thresholds
    ///
    /// # Arguments
    ///
    /// * `max_relative_error` - Maximum acceptable relative L2 error (e.g., 0.05 for 5%)
    ///
    /// # Returns
    ///
    /// true if validation passes, false otherwise
    pub fn passes(&self, max_relative_error: f64) -> bool {
        self.metrics.relative_l2_error < max_relative_error
    }

    /// Generate human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "PINN Validation Report\n\
             =====================\n\
             MAE: {:.6}\n\
             RMSE: {:.6}\n\
             Relative L2 Error: {:.2}%\n\
             Max Error: {:.6}\n\
             Correlation: {:.4}\n\
             Mean Relative Error: {:.2}%\n\
             Points Compared: {}\n\
             FDTD Time: {:.4}s\n\
             PINN Time: {:.6}s\n\
             Speedup: {:.1}×\n\
             Status: {}",
            self.metrics.mean_absolute_error,
            self.metrics.rmse,
            self.metrics.relative_l2_error * 100.0,
            self.metrics.max_error,
            self.correlation,
            self.mean_relative_error_percent,
            self.num_points,
            self.fdtd_time_secs,
            self.pinn_time_secs,
            self.speedup_factor,
            if self.passes(0.10) {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        )
    }
}

/// Validate PINN predictions against FDTD reference solution
///
/// # Arguments
///
/// * `pinn` - Trained PINN model
/// * `fdtd_config` - FDTD configuration for reference solution
///
/// # Returns
///
/// Comprehensive validation report
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "pinn")]
/// # {
/// use kwavers::ml::pinn::{PINN1DWave, PINNConfig};
/// use kwavers::ml::pinn::fdtd_reference::FDTDConfig;
/// use kwavers::ml::pinn::validation::validate_pinn_vs_fdtd;
///
/// let mut pinn = PINN1DWave::new(1500.0, PINNConfig::default())?;
/// // Train PINN...
///
/// let fdtd_config = FDTDConfig::default();
/// let report = validate_pinn_vs_fdtd(&pinn, fdtd_config)?;
/// println!("{}", report.summary());
/// # Ok::<(), kwavers::error::KwaversError>(())
/// # }
/// ```
pub fn validate_pinn_vs_fdtd(
    pinn: &PINN1DWave,
    fdtd_config: FDTDConfig,
) -> KwaversResult<ValidationReport> {
    use std::time::Instant;

    // Generate FDTD reference solution
    let fdtd_start = Instant::now();
    let mut fdtd_solver = FDTD1DWaveSolver::new(fdtd_config.clone())?;
    let fdtd_solution = fdtd_solver.solve()?;
    let fdtd_time = fdtd_start.elapsed().as_secs_f64();

    // Generate PINN prediction
    let pinn_start = Instant::now();
    let x = Array1::linspace(
        0.0,
        (fdtd_config.nx - 1) as f64 * fdtd_config.dx,
        fdtd_config.nx,
    );
    let t = Array1::linspace(
        0.0,
        (fdtd_config.nt - 1) as f64 * fdtd_config.dt,
        fdtd_config.nt,
    );
    let pinn_prediction = pinn.predict(&x, &t);
    let pinn_time = pinn_start.elapsed().as_secs_f64();

    // Compute validation metrics
    let metrics = compute_validation_metrics(&fdtd_solution, &pinn_prediction)?;

    // Compute correlation
    let correlation = compute_correlation(&fdtd_solution, &pinn_prediction)?;

    // Compute mean relative error
    let mean_relative_error = compute_mean_relative_error(&fdtd_solution, &pinn_prediction)?;

    let num_points = fdtd_solution.len();
    let speedup_factor = if pinn_time > 0.0 {
        fdtd_time / pinn_time
    } else {
        f64::INFINITY
    };

    Ok(ValidationReport {
        metrics,
        correlation,
        mean_relative_error_percent: mean_relative_error * 100.0,
        num_points,
        fdtd_time_secs: fdtd_time,
        pinn_time_secs: pinn_time,
        speedup_factor,
    })
}

/// Compute validation metrics between two solutions
fn compute_validation_metrics(
    reference: &Array2<f64>,
    prediction: &Array2<f64>,
) -> KwaversResult<ValidationMetrics> {
    if reference.dim() != prediction.dim() {
        return Err(KwaversError::InvalidInput(
            "Reference and prediction must have same dimensions".to_string(),
        ));
    }

    let (nx, nt) = reference.dim();
    let mut sum_abs_error = 0.0;
    let mut sum_squared_error = 0.0;
    let mut max_error: f64 = 0.0;
    let mut sum_squared_ref = 0.0;

    for i in 0..nx {
        for j in 0..nt {
            let ref_val = reference[[i, j]];
            let pred_val = prediction[[i, j]];
            let error = (pred_val - ref_val).abs();

            sum_abs_error += error;
            sum_squared_error += error * error;
            sum_squared_ref += ref_val * ref_val;
            max_error = max_error.max(error);
        }
    }

    let n = (nx * nt) as f64;
    let mean_absolute_error = sum_abs_error / n;
    let rmse = (sum_squared_error / n).sqrt();
    let relative_l2_error = if sum_squared_ref > 0.0 {
        (sum_squared_error / sum_squared_ref).sqrt()
    } else {
        0.0
    };

    Ok(ValidationMetrics {
        mean_absolute_error,
        rmse,
        relative_l2_error,
        max_error,
    })
}

/// Compute Pearson correlation coefficient
fn compute_correlation(reference: &Array2<f64>, prediction: &Array2<f64>) -> KwaversResult<f64> {
    let n = reference.len() as f64;

    let mean_ref: f64 = reference.iter().sum::<f64>() / n;
    let mean_pred: f64 = prediction.iter().sum::<f64>() / n;

    let mut sum_prod = 0.0;
    let mut sum_sq_ref = 0.0;
    let mut sum_sq_pred = 0.0;

    for (r, p) in reference.iter().zip(prediction.iter()) {
        let diff_ref = r - mean_ref;
        let diff_pred = p - mean_pred;
        sum_prod += diff_ref * diff_pred;
        sum_sq_ref += diff_ref * diff_ref;
        sum_sq_pred += diff_pred * diff_pred;
    }

    let correlation = if sum_sq_ref > 0.0 && sum_sq_pred > 0.0 {
        sum_prod / (sum_sq_ref.sqrt() * sum_sq_pred.sqrt())
    } else {
        0.0
    };

    Ok(correlation)
}

/// Compute mean relative error (avoiding division by small numbers)
fn compute_mean_relative_error(
    reference: &Array2<f64>,
    prediction: &Array2<f64>,
) -> KwaversResult<f64> {
    let epsilon = 1e-10;
    let mut sum_relative_error = 0.0;
    let mut count = 0;

    for (r, p) in reference.iter().zip(prediction.iter()) {
        if r.abs() > epsilon {
            sum_relative_error += ((p - r) / r).abs();
            count += 1;
        }
    }

    Ok(if count > 0 {
        sum_relative_error / count as f64
    } else {
        0.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::math::ml::pinn::wave_equation_1d::PINNConfig;

    #[test]
    fn test_compute_validation_metrics() {
        let reference = Array2::from_shape_fn((10, 10), |(i, j)| (i + j) as f64);
        let prediction = Array2::from_shape_fn((10, 10), |(i, j)| (i + j) as f64 + 0.1);

        let metrics = compute_validation_metrics(&reference, &prediction).unwrap();

        assert!(metrics.mean_absolute_error > 0.0);
        assert!(metrics.rmse > 0.0);
        assert!(metrics.max_error >= metrics.mean_absolute_error);
    }

    #[test]
    fn test_compute_correlation() {
        let reference = Array2::from_shape_fn((10, 10), |(i, j)| (i + j) as f64);
        let prediction = Array2::from_shape_fn((10, 10), |(i, j)| (i + j) as f64 * 1.1);

        let corr = compute_correlation(&reference, &prediction).unwrap();

        // Should be highly correlated (linear relationship)
        assert!(corr > 0.9);
    }

    #[test]
    fn test_validation_report_passes() {
        let metrics = ValidationMetrics {
            mean_absolute_error: 0.01,
            rmse: 0.02,
            relative_l2_error: 0.03,
            max_error: 0.05,
        };

        let report = ValidationReport {
            metrics,
            correlation: 0.99,
            mean_relative_error_percent: 3.0,
            num_points: 1000,
            fdtd_time_secs: 1.0,
            pinn_time_secs: 0.001,
            speedup_factor: 1000.0,
        };

        assert!(report.passes(0.10)); // 10% threshold
        assert!(!report.passes(0.02)); // 2% threshold
    }

    #[test]
    fn test_validation_report_summary() {
        let metrics = ValidationMetrics {
            mean_absolute_error: 0.01,
            rmse: 0.02,
            relative_l2_error: 0.03,
            max_error: 0.05,
        };

        let report = ValidationReport {
            metrics,
            correlation: 0.99,
            mean_relative_error_percent: 3.0,
            num_points: 1000,
            fdtd_time_secs: 1.0,
            pinn_time_secs: 0.001,
            speedup_factor: 1000.0,
        };

        let summary = report.summary();
        assert!(summary.contains("PINN Validation Report"));
        assert!(summary.contains("Speedup: 1000.0×"));
        assert!(summary.contains("✅ PASS"));
    }

    #[test]
    fn test_validate_pinn_vs_fdtd() {
        let config = PINNConfig::default();
        let mut pinn = PINN1DWave::new(1500.0, config).unwrap();

        // Train on dummy data
        let reference_data = Array2::from_elem((50, 50), 0.5);
        pinn.train(&reference_data, 10).unwrap();

        let fdtd_config = FDTDConfig {
            wave_speed: 1500.0,
            nx: 50,
            nt: 50,
            dx: 0.01,
            dt: 0.000005,
            ..Default::default()
        };

        let report = validate_pinn_vs_fdtd(&pinn, fdtd_config);
        assert!(report.is_ok());

        let report = report.unwrap();
        assert!(report.num_points > 0);
        assert!(report.speedup_factor > 0.0);
        assert!(report.correlation.abs() <= 1.0);
    }
}
