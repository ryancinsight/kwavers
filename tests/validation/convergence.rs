//! Convergence Validation for Numerical Solvers
//!
//! This module provides convergence analysis tools for validating that
//! numerical solvers converge to analytical solutions at the expected rate.
//!
//! # Convergence Rate Analysis
//!
//! For a numerical method with spatial discretization Δx and temporal
//! discretization Δt, the global error should scale as:
//!
//! ```text
//! E(Δx, Δt) = C₁(Δx)ᵖ + C₂(Δt)ᵍ
//! ```
//!
//! where:
//! - p is the spatial convergence order
//! - q is the temporal convergence order
//! - C₁, C₂ are method-dependent constants
//!
//! # Expected Convergence Orders
//!
//! - **FDTD**: O(Δx²) spatial, O(Δt²) temporal (centered differences)
//! - **PSTD**: O(Δxᴺ) spatial (spectral), O(Δt²) temporal
//! - **PINN**: Architecture-dependent, typically tested for monotonic convergence
//! - **FEM**: O(hᵖ⁺¹) where p is polynomial order
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use kwavers::tests::validation::convergence::ConvergenceStudy;
//! use kwavers::tests::validation::analytical_solutions::PlaneWave2D;
//!
//! let analytical = PlaneWave2D::p_wave(...);
//! let mut study = ConvergenceStudy::new(analytical);
//!
//! // Run solver at different resolutions
//! for resolution in [32, 64, 128, 256] {
//!     let error = run_solver_at_resolution(resolution);
//!     study.add_measurement(resolution as f64, error);
//! }
//!
//! // Analyze convergence rate
//! let rate = study.compute_convergence_rate();
//! assert!(rate > 1.8, "Expected at least second-order convergence");
//! ```

use super::error_metrics::ErrorMetrics;

/// Convergence study for numerical solver validation
///
/// # Mathematical Specification
///
/// Records error measurements E(h) at different discretization levels h,
/// then fits a power law:
///
/// ```text
/// E(h) = C hᵖ  =>  log(E) = log(C) + p log(h)
/// ```
///
/// The convergence rate p is estimated via least-squares linear regression
/// on log-log data.
#[derive(Debug, Clone)]
pub struct ConvergenceStudy {
    /// Discretization parameters (h values)
    pub discretizations: Vec<f64>,
    /// Corresponding error measurements
    pub errors: Vec<f64>,
    /// Name/description of the study
    pub name: String,
}

impl ConvergenceStudy {
    /// Create new convergence study
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            discretizations: Vec::new(),
            errors: Vec::new(),
            name: name.into(),
        }
    }

    /// Add measurement at a given discretization level
    ///
    /// # Arguments
    ///
    /// * `h` - Discretization parameter (grid spacing, Δx, etc.)
    /// * `error` - Measured error (L² or L∞ norm)
    pub fn add_measurement(&mut self, h: f64, error: f64) {
        self.discretizations.push(h);
        self.errors.push(error);
    }

    /// Add measurement from error metrics
    pub fn add_from_metrics(&mut self, h: f64, metrics: &ErrorMetrics) {
        self.add_measurement(h, metrics.l2_error);
    }

    /// Compute convergence rate via least-squares fit
    ///
    /// # Returns
    ///
    /// Convergence rate p where E(h) ∝ hᵖ, or None if insufficient data
    ///
    /// # Mathematical Specification
    ///
    /// Fit: log(E) = a + p·log(h)
    ///
    /// Least squares solution:
    /// ```text
    /// p = (n·Σ(xᵢyᵢ) - Σxᵢ·Σyᵢ) / (n·Σ(xᵢ²) - (Σxᵢ)²)
    /// ```
    /// where xᵢ = log(hᵢ), yᵢ = log(Eᵢ)
    pub fn compute_convergence_rate(&self) -> Option<f64> {
        let n = self.discretizations.len();
        if n < 2 {
            return None;
        }

        // Filter out zero errors (would cause log(0) = -∞)
        let mut valid_points = Vec::new();
        for i in 0..n {
            if self.errors[i] > 0.0 && self.discretizations[i] > 0.0 {
                let x = self.discretizations[i].ln();
                let y = self.errors[i].ln();
                if x.is_finite() && y.is_finite() {
                    valid_points.push((x, y));
                }
            }
        }

        if valid_points.len() < 2 {
            return None;
        }

        let n = valid_points.len() as f64;
        let sum_x: f64 = valid_points.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = valid_points.iter().map(|(_, y)| y).sum();
        let sum_xx: f64 = valid_points.iter().map(|(x, _)| x * x).sum();
        let sum_xy: f64 = valid_points.iter().map(|(x, y)| x * y).sum();

        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < 1e-15 {
            return None;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        Some(slope)
    }

    /// Compute R² coefficient of determination for convergence fit
    ///
    /// # Returns
    ///
    /// R² ∈ [0, 1] where 1 indicates perfect fit, or None if insufficient data
    pub fn compute_r_squared(&self) -> Option<f64> {
        let rate = self.compute_convergence_rate()?;

        // Compute intercept: a = mean(log(E)) - p·mean(log(h))
        let mut log_h_sum = 0.0;
        let mut log_e_sum = 0.0;
        let mut count = 0;

        for i in 0..self.discretizations.len() {
            if self.errors[i] > 0.0 && self.discretizations[i] > 0.0 {
                log_h_sum += self.discretizations[i].ln();
                log_e_sum += self.errors[i].ln();
                count += 1;
            }
        }

        if count < 2 {
            return None;
        }

        let mean_log_h = log_h_sum / count as f64;
        let mean_log_e = log_e_sum / count as f64;
        let intercept = mean_log_e - rate * mean_log_h;

        // Compute R²
        let mut ss_res = 0.0; // Residual sum of squares
        let mut ss_tot = 0.0; // Total sum of squares

        for i in 0..self.discretizations.len() {
            if self.errors[i] > 0.0 && self.discretizations[i] > 0.0 {
                let log_h = self.discretizations[i].ln();
                let log_e = self.errors[i].ln();

                let predicted = intercept + rate * log_h;
                let residual = log_e - predicted;
                ss_res += residual * residual;

                let deviation = log_e - mean_log_e;
                ss_tot += deviation * deviation;
            }
        }

        if ss_tot < 1e-15 {
            return None;
        }

        Some(1.0 - ss_res / ss_tot)
    }

    /// Check if convergence is monotonic (errors decrease with refinement)
    ///
    /// # Returns
    ///
    /// true if E(hᵢ₊₁) < E(hᵢ) for sorted discretizations
    pub fn is_monotonic(&self) -> bool {
        if self.errors.len() < 2 {
            return true;
        }

        // Sort by discretization (largest to smallest)
        let mut paired: Vec<_> = self
            .discretizations
            .iter()
            .zip(self.errors.iter())
            .collect();
        paired.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());

        // Check if errors are decreasing
        for i in 0..paired.len() - 1 {
            if paired[i + 1].1 >= paired[i].1 {
                return false;
            }
        }

        true
    }

    /// Estimate error at a target discretization level
    ///
    /// # Arguments
    ///
    /// * `h_target` - Target discretization parameter
    ///
    /// # Returns
    ///
    /// Extrapolated error E(h_target) = C·h_target^p, or None if fit failed
    pub fn extrapolate(&self, h_target: f64) -> Option<f64> {
        let rate = self.compute_convergence_rate()?;

        // Compute intercept from mean values
        let mut log_h_sum = 0.0;
        let mut log_e_sum = 0.0;
        let mut count = 0;

        for i in 0..self.discretizations.len() {
            if self.errors[i] > 0.0 && self.discretizations[i] > 0.0 {
                log_h_sum += self.discretizations[i].ln();
                log_e_sum += self.errors[i].ln();
                count += 1;
            }
        }

        if count == 0 {
            return None;
        }

        let mean_log_h = log_h_sum / count as f64;
        let mean_log_e = log_e_sum / count as f64;
        let intercept = mean_log_e - rate * mean_log_h;

        // Extrapolate: log(E) = intercept + rate·log(h)
        let log_e_target = intercept + rate * h_target.ln();
        Some(log_e_target.exp())
    }
}

/// Convergence validation result
#[derive(Debug, Clone)]
pub struct ConvergenceResult {
    /// Convergence rate (order of accuracy)
    pub rate: f64,
    /// Coefficient of determination (fit quality)
    pub r_squared: f64,
    /// Is convergence monotonic?
    pub is_monotonic: bool,
    /// Expected convergence rate
    pub expected_rate: f64,
    /// Test passed (rate meets expectations)?
    pub passed: bool,
}

impl ConvergenceResult {
    /// Create convergence result with validation
    ///
    /// # Arguments
    ///
    /// * `study` - Convergence study with measurements
    /// * `expected_rate` - Expected convergence order (e.g., 2.0 for second-order)
    /// * `tolerance` - Acceptable deviation from expected rate
    pub fn from_study(
        study: &ConvergenceStudy,
        expected_rate: f64,
        tolerance: f64,
    ) -> Option<Self> {
        let rate = study.compute_convergence_rate()?;
        let r_squared = study.compute_r_squared().unwrap_or(0.0);
        let is_monotonic = study.is_monotonic();

        let rate_deviation = (rate - expected_rate).abs();
        let passed = rate_deviation <= tolerance && is_monotonic && r_squared > 0.9;

        Some(Self {
            rate,
            r_squared,
            is_monotonic,
            expected_rate,
            passed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_study_perfect_quadratic() {
        let mut study = ConvergenceStudy::new("test");

        // Perfect quadratic convergence: E = h²
        for &h in &[1.0, 0.5, 0.25, 0.125] {
            study.add_measurement(h, h * h);
        }

        let rate = study.compute_convergence_rate().unwrap();
        assert!((rate - 2.0).abs() < 0.01, "Expected rate=2, got {}", rate);

        let r_squared = study.compute_r_squared().unwrap();
        assert!(r_squared > 0.999, "Expected R²≈1, got {}", r_squared);
    }

    #[test]
    fn test_convergence_study_first_order() {
        let mut study = ConvergenceStudy::new("test");

        // First-order convergence: E = h
        for &h in &[1.0, 0.5, 0.25, 0.125] {
            study.add_measurement(h, h);
        }

        let rate = study.compute_convergence_rate().unwrap();
        assert!((rate - 1.0).abs() < 0.01, "Expected rate=1, got {}", rate);
    }

    #[test]
    fn test_convergence_monotonic() {
        let mut study = ConvergenceStudy::new("test");

        study.add_measurement(1.0, 1.0);
        study.add_measurement(0.5, 0.25);
        study.add_measurement(0.25, 0.0625);

        assert!(study.is_monotonic());
    }

    #[test]
    fn test_convergence_non_monotonic() {
        let mut study = ConvergenceStudy::new("test");

        study.add_measurement(1.0, 1.0);
        study.add_measurement(0.5, 0.3); // Decreases
        study.add_measurement(0.25, 0.35); // Increases - non-monotonic!

        assert!(!study.is_monotonic());
    }

    #[test]
    fn test_convergence_extrapolation() {
        let mut study = ConvergenceStudy::new("test");

        // E = 0.1 * h²
        for &h in &[1.0, 0.5, 0.25] {
            study.add_measurement(h, 0.1 * h * h);
        }

        let extrapolated = study.extrapolate(0.125).unwrap();
        let expected = 0.1 * 0.125 * 0.125;

        assert!((extrapolated - expected).abs() / expected < 0.01);
    }

    #[test]
    fn test_convergence_result_validation() {
        let mut study = ConvergenceStudy::new("test");

        // Perfect second-order convergence
        for &h in &[1.0, 0.5, 0.25, 0.125] {
            study.add_measurement(h, h * h);
        }

        let result = ConvergenceResult::from_study(&study, 2.0, 0.1).unwrap();

        assert!(result.passed);
        assert!((result.rate - 2.0).abs() < 0.01);
        assert!(result.is_monotonic);
    }

    #[test]
    fn test_convergence_insufficient_data() {
        let study = ConvergenceStudy::new("test");
        assert!(study.compute_convergence_rate().is_none());

        let mut study_single = ConvergenceStudy::new("test");
        study_single.add_measurement(1.0, 1.0);
        assert!(study_single.compute_convergence_rate().is_none());
    }

    #[test]
    fn test_convergence_zero_error_handling() {
        let mut study = ConvergenceStudy::new("test");

        study.add_measurement(1.0, 1.0);
        study.add_measurement(0.5, 0.0); // Zero error would cause log(0)
        study.add_measurement(0.25, 0.25);

        // Should still compute rate from valid points
        let rate = study.compute_convergence_rate();
        assert!(rate.is_some());
    }
}
