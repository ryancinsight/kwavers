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

mod metrics;
#[cfg(test)]
mod tests;

pub use metrics::{compute_correlation, compute_mean_relative_error, compute_validation_metrics};

use crate::inverse::pinn::ml::burn_wave_equation_1d::BurnPINN1DWave;
use crate::inverse::pinn::ml::fdtd_reference::{FDTD1DWaveSolver, FDTDConfig};
use burn::tensor::backend::AutodiffBackend;
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array1;

/// Standard validation metrics.
#[derive(Debug, Clone)]
pub struct PinnValidationMetrics {
    pub mean_absolute_error: f64,
    pub rmse: f64,
    pub relative_l2_error: f64,
    pub max_error: f64,
}

/// Comprehensive validation results comparing PINN vs FDTD.
#[derive(Debug, Clone)]
pub struct PinnValidationReport {
    /// Standard validation metrics
    pub metrics: PinnValidationMetrics,
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

impl PinnValidationReport {
    /// Check if validation passes target thresholds.
    ///
    /// * `max_relative_error` — Maximum acceptable relative L2 error (e.g., 0.05 for 5%).
    pub fn passes(&self, max_relative_error: f64) -> bool {
        self.metrics.relative_l2_error < max_relative_error
    }

    /// Generate human-readable summary.
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

/// Validate PINN predictions against FDTD reference solution.
///
/// Returns a comprehensive report including timing, metrics, and correlation.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn validate_pinn_vs_fdtd<B: AutodiffBackend>(
    pinn: &BurnPINN1DWave<B>,
    device: &B::Device,
    fdtd_config: FDTDConfig,
) -> KwaversResult<PinnValidationReport> {
    use std::time::Instant;

    let fdtd_start = Instant::now();
    let mut fdtd_solver = FDTD1DWaveSolver::new(fdtd_config.clone())?;
    let fdtd_solution = fdtd_solver.solve()?;
    let fdtd_time = fdtd_start.elapsed().as_secs_f64();

    let pinn_start = Instant::now();
    let x_coords = Array1::linspace(
        0.0,
        (fdtd_config.nx - 1) as f64 * fdtd_config.dx,
        fdtd_config.nx,
    );
    let t_coords = Array1::linspace(
        0.0,
        (fdtd_config.nt - 1) as f64 * fdtd_config.dt,
        fdtd_config.nt,
    );
    let mut x = Vec::with_capacity(fdtd_config.nx * fdtd_config.nt);
    let mut t = Vec::with_capacity(fdtd_config.nx * fdtd_config.nt);
    for x_val in x_coords.iter() {
        for t_val in t_coords.iter() {
            x.push(*x_val);
            t.push(*t_val);
        }
    }
    let x = Array1::from_vec(x);
    let t = Array1::from_vec(t);
    let pinn_prediction_flat = pinn.predict(&x, &t, device)?;
    let pinn_prediction = ndarray::Array2::from_shape_vec(
        (fdtd_config.nx, fdtd_config.nt),
        pinn_prediction_flat.iter().copied().collect(),
    )
    .map_err(|err| KwaversError::InternalError(err.to_string()))?;
    let pinn_time = pinn_start.elapsed().as_secs_f64();

    let metrics = compute_validation_metrics(&fdtd_solution, &pinn_prediction)?;
    let correlation = compute_correlation(&fdtd_solution, &pinn_prediction)?;
    let mean_relative_error = compute_mean_relative_error(&fdtd_solution, &pinn_prediction)?;
    let num_points = fdtd_solution.len();
    let speedup_factor = if pinn_time > 0.0 {
        fdtd_time / pinn_time
    } else {
        f64::INFINITY
    };

    Ok(PinnValidationReport {
        metrics,
        correlation,
        mean_relative_error_percent: mean_relative_error * 100.0,
        num_points,
        fdtd_time_secs: fdtd_time,
        pinn_time_secs: pinn_time,
        speedup_factor,
    })
}
