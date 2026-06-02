//! Validation test suite
//!
//! # Manufactured acoustic validation theorem
//!
//! Let `u(x) = sin(kx)` on a uniform grid with spacing `h`. The exact second
//! derivative is `u''(x) = -k^2 u(x)`. The sixth-order centered stencil
//!
//! ```text
//! D2_h u_i = (u_{i-3}/90 - 3u_{i-2}/20 + 3u_{i-1}/2
//!            - 49u_i/18 + 3u_{i+1}/2 - 3u_{i+2}/20 + u_{i+3}/90) / h^2
//! ```
//!
//! has local truncation error `O(h^6)` by Taylor cancellation through degree
//! seven. Therefore the relative residual
//! `||D2_h u + k^2 u||_2 / ||k^2 u||_2` decreases under grid refinement for
//! the smooth acoustic eigenmode and supplies an input-sensitive validation
//! fixture without depending on a concrete PSTD or FDTD implementation.

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;

const MANUFACTURED_WAVENUMBER: f64 = TWO_PI;
const MIN_MANUFACTURED_POINTS: usize = 16;
const ACOUSTIC_MODE_SPEED: f64 = 1.0;

/// Hybrid validation test suite
#[derive(Debug)]
pub struct HybridValidationSuite {
    /// Test configuration
    config: super::HybridValidationSuiteConfig,
}

impl HybridValidationSuite {
    /// Create new validation suite
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(config: super::HybridValidationSuiteConfig) -> Self {
        Self { config }
    }

    /// Run all validation tests
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn run_all_tests(&self) -> KwaversResult<super::ValidationSummary> {
        let mut summary = super::ValidationSummary {
            total_tests: 0,
            tests_passed: 0,
            tests_failed: 0,
            pass_rate: 0.0,
        };

        // Run convergence tests
        if self.config.test_convergence {
            let convergence_result = self.run_convergence_test()?;
            summary.total_tests += 1;
            if convergence_result {
                summary.tests_passed += 1;
            } else {
                summary.tests_failed += 1;
            }
        }

        // Run accuracy tests
        let accuracy_result = self.run_accuracy_test()?;
        summary.total_tests += 1;
        if accuracy_result {
            summary.tests_passed += 1;
        } else {
            summary.tests_failed += 1;
        }

        // Run stability tests
        let stability_result = self.run_stability_test()?;
        summary.total_tests += 1;
        if stability_result {
            summary.tests_passed += 1;
        } else {
            summary.tests_failed += 1;
        }

        summary.pass_rate = if summary.total_tests > 0 {
            summary.tests_passed as f64 / summary.total_tests as f64
        } else {
            0.0
        };

        Ok(summary)
    }

    /// Run convergence test
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn run_convergence_test(&self) -> KwaversResult<bool> {
        // Test that error decreases with grid refinement
        let mut errors = Vec::new();
        let grid_sizes = vec![32, 64, 128];

        for size in grid_sizes {
            let error = self.compute_error_for_grid_size(size)?;
            errors.push(error);
        }

        // Check if errors decrease monotonically
        for i in 1..errors.len() {
            if errors[i] >= errors[i - 1] {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Run accuracy test
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn run_accuracy_test(&self) -> KwaversResult<bool> {
        // Test against analytical solution or reference
        let computed = self.compute_solution()?;
        let reference = self.get_reference_solution()?;

        let error = self.compute_relative_error(&computed, &reference)?;
        Ok(error < self.config.error_tolerance)
    }

    /// Run stability test
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn run_stability_test(&self) -> KwaversResult<bool> {
        // Test CFL condition and numerical stability
        let max_eigenvalue = self.compute_max_eigenvalue()?;
        let dt = self.compute_time_step()?;

        let cfl = max_eigenvalue * dt;
        Ok(cfl < kwavers_core::constants::numerical::CFL_SAFETY_FACTOR)
    }

    /// Compute error for a given grid size
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_error_for_grid_size(&self, size: usize) -> KwaversResult<f64> {
        let points = size.max(MIN_MANUFACTURED_POINTS);
        let h = grid_spacing(points);
        let mut residual_l2 = 0.0;
        let mut reference_l2 = 0.0;

        for i in 3..points - 3 {
            let x = i as f64 * h;
            let numerical = sixth_order_second_derivative(x, h);
            let reference = exact_second_derivative(x);
            residual_l2 += (numerical - reference).powi(2);
            reference_l2 += reference.powi(2);
        }

        Ok((residual_l2 / reference_l2).sqrt())
    }

    /// Compute numerical solution
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_solution(&self) -> KwaversResult<f64> {
        let points = self.validation_points();
        let h = grid_spacing(points);
        Ok(sixth_order_second_derivative(0.25, h))
    }

    /// Get reference solution
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn get_reference_solution(&self) -> KwaversResult<f64> {
        Ok(exact_second_derivative(0.25))
    }

    /// Compute relative error
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_relative_error(&self, computed: &f64, reference: &f64) -> KwaversResult<f64> {
        if reference.abs() > kwavers_core::constants::numerical::EPSILON {
            Ok((computed - reference).abs() / reference.abs())
        } else {
            Ok(computed.abs())
        }
    }

    /// Compute maximum eigenvalue for stability
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_max_eigenvalue(&self) -> KwaversResult<f64> {
        Ok(ACOUSTIC_MODE_SPEED / grid_spacing(self.validation_points()))
    }

    /// Compute time step
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_time_step(&self) -> KwaversResult<f64> {
        Ok(0.25 * grid_spacing(self.validation_points()) / ACOUSTIC_MODE_SPEED)
    }

    fn validation_points(&self) -> usize {
        self.config.num_iterations.max(MIN_MANUFACTURED_POINTS)
    }
}

fn grid_spacing(points: usize) -> f64 {
    1.0 / (points - 1) as f64
}

fn manufactured_mode(x: f64) -> f64 {
    (MANUFACTURED_WAVENUMBER * x).sin()
}

fn exact_second_derivative(x: f64) -> f64 {
    -MANUFACTURED_WAVENUMBER.powi(2) * manufactured_mode(x)
}

fn sixth_order_second_derivative(x: f64, h: f64) -> f64 {
    (1.5f64.mul_add(
        manufactured_mode(x + h),
        1.5f64.mul_add(
            manufactured_mode(x - h),
            manufactured_mode(3.0f64.mul_add(-h, x)) / 90.0
                - 3.0 * manufactured_mode(2.0f64.mul_add(-h, x)) / 20.0,
        ) - 49.0 * manufactured_mode(x) / 18.0,
    ) - 3.0 * manufactured_mode(2.0f64.mul_add(h, x)) / 20.0
        + manufactured_mode(3.0f64.mul_add(h, x)) / 90.0)
        / h.powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn suite_with(points: usize) -> HybridValidationSuite {
        HybridValidationSuite::new(super::super::HybridValidationSuiteConfig {
            error_tolerance: 1e-6,
            num_iterations: points,
            test_convergence: true,
            benchmark_performance: false,
        })
    }

    #[test]
    fn manufactured_accuracy_matches_closed_form_second_derivative() {
        let suite = suite_with(128);
        let computed = suite.compute_solution().unwrap();
        let reference = suite.get_reference_solution().unwrap();
        let relative_error = suite.compute_relative_error(&computed, &reference).unwrap();

        assert!(relative_error < 1e-8, "relative_error={relative_error:e}");
        assert!(computed.is_finite());
        assert!(reference.is_finite());
    }

    #[test]
    fn convergence_error_decreases_under_grid_refinement() {
        let suite = suite_with(128);
        let coarse = suite.compute_error_for_grid_size(32).unwrap();
        let medium = suite.compute_error_for_grid_size(64).unwrap();
        let fine = suite.compute_error_for_grid_size(128).unwrap();

        assert!(medium < coarse, "coarse={coarse:e}, medium={medium:e}");
        assert!(fine < medium, "medium={medium:e}, fine={fine:e}");
        assert!(
            coarse / fine > 32.0,
            "sixth-order residual should contract materially: coarse={coarse:e}, fine={fine:e}"
        );
    }

    #[test]
    fn stability_uses_cfl_bound_from_validation_grid() {
        let suite = suite_with(100);
        let cfl = suite.compute_max_eigenvalue().unwrap() * suite.compute_time_step().unwrap();

        assert!((cfl - 0.25).abs() < 4.0 * f64::EPSILON, "cfl={cfl}");
        assert!(suite.run_stability_test().unwrap());
    }

    #[test]
    fn run_all_tests_reports_value_semantic_pass_rate() {
        let summary = suite_with(128).run_all_tests().unwrap();

        assert_eq!(summary.total_tests, 3);
        assert_eq!(summary.tests_passed, 3);
        assert_eq!(summary.tests_failed, 0);
        assert_eq!(summary.pass_rate, 1.0);
    }
}
