use super::{
    EquivalenceReport, DEFAULT_ABSOLUTE_TOLERANCE, DEFAULT_RELATIVE_TOLERANCE, MEASUREMENT_STEPS,
    WARMUP_STEPS,
};
use crate::core::error::ValidationError;
use ndarray::{Array3, Zip};

/// Configuration for equivalence validation
///
/// Controls tolerance thresholds and validation behavior for
/// GPU/CPU numerical equivalence testing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EquivalenceValidator {
    /// Absolute error tolerance for comparisons
    pub tolerance_absolute: f64,

    /// Relative error tolerance for comparisons
    pub tolerance_relative: f64,

    /// Require exact bitwise equality (for deterministic operations)
    pub require_bitwise: bool,

    /// Number of warmup steps before measurement
    pub warmup_steps: usize,

    /// Number of measurement steps
    pub measurement_steps: usize,

    /// Validate velocity fields in addition to pressure
    pub validate_velocity: bool,
}

impl Default for EquivalenceValidator {
    fn default() -> Self {
        Self {
            tolerance_absolute: DEFAULT_ABSOLUTE_TOLERANCE,
            tolerance_relative: DEFAULT_RELATIVE_TOLERANCE,
            require_bitwise: false, // Allow for parallel reduction differences
            warmup_steps: WARMUP_STEPS,
            measurement_steps: MEASUREMENT_STEPS,
            validate_velocity: false,
        }
    }
}

impl EquivalenceValidator {
    /// Create a validator with strict bitwise equality requirements
    ///
    /// Use this for deterministic operations where operation order
    /// is guaranteed to be identical between GPU and CPU.
    pub fn strict() -> Self {
        Self {
            tolerance_absolute: 0.0,
            tolerance_relative: 0.0,
            require_bitwise: true,
            ..Default::default()
        }
    }

    /// Create a validator with relaxed tolerances for parallel reductions
    ///
    /// Use this when operation ordering differs (e.g., parallel sums).
    pub fn relaxed(allowed_relative_error: f64) -> Self {
        Self {
            tolerance_relative: allowed_relative_error.max(DEFAULT_RELATIVE_TOLERANCE),
            require_bitwise: false,
            ..Default::default()
        }
    }

    /// Validate two arrays for equivalence
    ///
    /// Computes error metrics and returns a report with pass/fail status.
    pub fn validate_arrays(
        &self,
        cpu_result: &Array3<f64>,
        gpu_result: &Array3<f64>,
        cpu_time_ms: f64,
        gpu_time_ms: f64,
    ) -> Result<EquivalenceReport, ValidationError> {
        // Check dimensions match
        if cpu_result.shape() != gpu_result.shape() {
            return Err(ValidationError::DimensionMismatch {
                expected: format!("{:?}", cpu_result.shape()),
                actual: format!("{:?}", gpu_result.shape()),
            });
        }

        let total_points = cpu_result.len();
        let mut report = EquivalenceReport::new(self.tolerance_relative, total_points);

        // Compute peak pressures
        report.cpu_peak_pressure = cpu_result.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        report.gpu_peak_pressure = gpu_result.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);

        // Compute error metrics
        let mut max_abs_error: f64 = 0.0;
        let mut max_rel_error: f64 = 0.0;
        let mut divergent_count: usize = 0;
        let min_threshold = self.tolerance_absolute.max(1e-300);

        Zip::from(cpu_result)
            .and(gpu_result)
            .for_each(|&cpu_val, &gpu_val| {
                let abs_error = (gpu_val - cpu_val).abs();
                max_abs_error = max_abs_error.max(abs_error);

                // Relative error (only when |cpu_val| > threshold)
                let rel_error = if cpu_val.abs() > min_threshold {
                    abs_error / cpu_val.abs()
                } else {
                    abs_error / min_threshold
                };

                max_rel_error = max_rel_error.max(rel_error);

                if rel_error > self.tolerance_relative {
                    divergent_count += 1;
                }
            });

        report.max_absolute_error = max_abs_error;
        report.max_relative_error = max_rel_error;
        report.divergent_points = divergent_count;
        report.cpu_time_ms = cpu_time_ms;
        report.gpu_time_ms = gpu_time_ms;
        report.speedup = if gpu_time_ms > 0.0 {
            cpu_time_ms / gpu_time_ms
        } else {
            0.0
        };

        // Determine pass/fail
        report.passed = max_rel_error <= self.tolerance_relative && divergent_count == 0;

        if !report.passed {
            report.failure_reason = Some(format!(
                "Max relative error {:.6e} exceeds threshold {:.6e}, or {} divergent points detected",
                max_rel_error, self.tolerance_relative, divergent_count
            ));
        }

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test validator strict mode
    #[test]
    fn test_validator_strict() {
        let strict = EquivalenceValidator::strict();
        assert!(strict.require_bitwise, "Strict mode requires bitwise");
        assert_eq!(strict.tolerance_absolute, 0.0, "Strict mode: zero absolute");
        assert_eq!(strict.tolerance_relative, 0.0, "Strict mode: zero relative");
    }

    /// Test validator relaxed mode
    #[test]
    fn test_validator_relaxed() {
        let relaxed = EquivalenceValidator::relaxed(1e-10);
        assert!(!relaxed.require_bitwise, "Relaxed mode allows non-bitwise");
        assert_eq!(relaxed.tolerance_relative, 1e-10, "Custom tolerance set");
    }

    /// Test validator default values
    #[test]
    fn test_validator_default() {
        let default = EquivalenceValidator::default();
        assert_eq!(default.tolerance_relative, DEFAULT_RELATIVE_TOLERANCE);
        assert_eq!(default.tolerance_absolute, DEFAULT_ABSOLUTE_TOLERANCE);
        assert_eq!(default.warmup_steps, WARMUP_STEPS);
        assert_eq!(default.measurement_steps, MEASUREMENT_STEPS);
    }

    /// Test array validation with identical arrays
    #[test]
    fn test_validate_arrays_identical() {
        let shape = (10, 10, 10);
        let cpu = Array3::from_elem(shape, 1.0);
        let gpu = Array3::from_elem(shape, 1.0);

        let validator = EquivalenceValidator::strict();
        let report = validator
            .validate_arrays(&cpu, &gpu, 1.0, 0.5)
            .expect("Validation should succeed");

        assert!(
            report.passed(),
            "Identical arrays should pass strict validation"
        );
        assert_eq!(report.max_absolute_error, 0.0, "Zero error expected");
        assert_eq!(
            report.max_relative_error, 0.0,
            "Zero relative error expected"
        );
    }

    /// Test array validation with small error
    #[test]
    fn test_validate_arrays_small_error() {
        let shape = (10, 10, 10);
        let cpu = Array3::from_elem(shape, 1.0);
        let mut gpu = Array3::from_elem(shape, 1.0);

        // Introduce 1 ULP error at one point
        gpu[[5, 5, 5]] = 1.0 + f64::EPSILON;

        let validator = EquivalenceValidator::strict();
        let report = validator
            .validate_arrays(&cpu, &gpu, 1.0, 1.0)
            .expect("Validation should complete");

        // Should fail strict validation
        assert!(
            !report.passed(),
            "Should detect 1 ULP difference in strict mode"
        );
        assert!(
            report.max_absolute_error > 0.0,
            "Should detect non-zero error"
        );
    }

    /// Test array validation with dimension mismatch
    #[test]
    fn test_validate_arrays_dimension_mismatch() {
        let cpu = Array3::zeros((10, 10, 10));
        let gpu = Array3::zeros((10, 10, 11));

        let validator = EquivalenceValidator::default();
        let result = validator.validate_arrays(&cpu, &gpu, 1.0, 1.0);

        assert!(result.is_err(), "Should error on dimension mismatch");
    }

    /// Test validation with zero pressure arrays
    #[test]
    fn test_validate_arrays_zero_pressure() {
        let shape = (10, 10, 10);
        let cpu = Array3::zeros(shape);
        let gpu = Array3::zeros(shape);

        let validator = EquivalenceValidator::default();
        let report = validator
            .validate_arrays(&cpu, &gpu, 0.0, 0.0)
            .expect("Validation should handle zeros");

        assert!(report.passed(), "Zero arrays should be equivalent");
        assert_eq!(report.max_absolute_error, 0.0);
    }

    /// Test error symmetry for deterministic operations
    /// |GPU - CPU| = |CPU - GPU|
    #[test]
    fn test_error_symmetry() {
        let cpu = Array3::from_elem((10, 10, 10), 1.5);
        let gpu = Array3::from_elem((10, 10, 10), 1.5);

        let validator1 = EquivalenceValidator::default();
        let report1 = validator1
            .validate_arrays(&cpu, &gpu, 1.0, 0.5)
            .expect("Validation should succeed");

        // Swap order
        let validator2 = EquivalenceValidator::default();
        let report2 = validator2
            .validate_arrays(&gpu, &cpu, 0.5, 1.0)
            .expect("Validation should succeed");

        // Symmetric error: max_absolute_error should be same
        assert_eq!(
            report1.max_absolute_error, report2.max_absolute_error,
            "Error should be symmetric"
        );
    }

    /// Test zero error for identical values property
    #[test]
    fn test_zero_error_for_identical() {
        let array = Array3::from_elem((5, 5, 5), std::f64::consts::E);
        let validator = EquivalenceValidator::strict();

        let report = validator
            .validate_arrays(&array, &array, 1.0, 1.0)
            .expect("Validation should succeed");

        assert_eq!(
            report.max_absolute_error, 0.0,
            "Identical arrays: zero error"
        );
        assert_eq!(
            report.max_relative_error, 0.0,
            "Identical arrays: zero relative error"
        );
    }
}
