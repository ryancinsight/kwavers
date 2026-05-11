use std::fmt;

/// Comprehensive report from GPU/CPU equivalence validation
///
/// Contains statistical measures of numerical divergence between
/// GPU and CPU implementations.
#[derive(Debug, Clone, PartialEq)]
pub struct EquivalenceReport {
    /// Maximum absolute error observed across all points
    /// |GPU_i - CPU_i| for any grid point i
    pub max_absolute_error: f64,

    /// Maximum relative error observed across all points
    /// max_i |GPU_i - CPU_i| / |CPU_i| for CPU_i ≠ 0
    pub max_relative_error: f64,

    /// Number of points exceeding tolerance thresholds
    pub divergent_points: usize,

    /// Total number of grid points compared
    pub total_points: usize,

    /// Applied tolerance threshold for pass/fail determination
    pub pass_threshold: f64,

    /// Peak absolute pressure value from CPU reference
    pub cpu_peak_pressure: f64,

    /// Peak absolute pressure value from GPU result
    pub gpu_peak_pressure: f64,

    /// CPU execution time (ms)
    pub cpu_time_ms: f64,

    /// GPU execution time (ms)
    pub gpu_time_ms: f64,

    /// Speedup factor (cpu_time / gpu_time)
    pub speedup: f64,

    /// Whether the validation passed
    pub passed: bool,

    /// Detailed error message if validation failed
    pub failure_reason: Option<String>,
}

impl EquivalenceReport {
    /// Create a new equivalence report
    pub fn new(pass_threshold: f64, total_points: usize) -> Self {
        Self {
            max_absolute_error: 0.0,
            max_relative_error: 0.0,
            divergent_points: 0,
            total_points,
            pass_threshold,
            cpu_peak_pressure: 0.0,
            gpu_peak_pressure: 0.0,
            cpu_time_ms: 0.0,
            gpu_time_ms: 0.0,
            speedup: 0.0,
            passed: false,
            failure_reason: None,
        }
    }

    /// Check if validation passed
    ///
    /// Validation passes if:
    /// - max_relative_error ≤ pass_threshold
    /// - divergent_points = 0 (strict)
    pub fn passed(&self) -> bool {
        self.passed && self.max_relative_error <= self.pass_threshold && self.divergent_points == 0
    }

    /// Get divergent fraction
    pub fn divergent_fraction(&self) -> f64 {
        if self.total_points == 0 {
            0.0
        } else {
            self.divergent_points as f64 / self.total_points as f64
        }
    }

    /// Format summary string
    pub fn summary(&self) -> String {
        format!(
            "GPU/CPU Equivalence Report:\n\
             \x20 Max Absolute Error:    {max_abs:.6e} Pa\n\
             \x20 Max Relative Error:    {max_rel:.6e}\n\
             \x20 Divergent Points:      {div} / {tot} ({frac:.4}%)\n\
             \x20 Pass Threshold:        {thresh:.6e}\n\
             \x20 CPU Peak Pressure:     {cpu_peak:.6e} Pa\n\
             \x20 GPU Peak Pressure:     {gpu_peak:.6e} Pa\n\
             \x20 CPU Time:              {cpu_t:.3} ms\n\
             \x20 GPU Time:              {gpu_t:.3} ms\n\
             \x20 Speedup:               {speed:.2}×\n\
             \x20 Status:                {status}",
            max_abs = self.max_absolute_error,
            max_rel = self.max_relative_error,
            div = self.divergent_points,
            tot = self.total_points,
            frac = self.divergent_fraction() * 100.0,
            thresh = self.pass_threshold,
            cpu_peak = self.cpu_peak_pressure,
            gpu_peak = self.gpu_peak_pressure,
            cpu_t = self.cpu_time_ms,
            gpu_t = self.gpu_time_ms,
            speed = self.speedup,
            status = if self.passed() {
                "✓ PASSED"
            } else {
                "✗ FAILED"
            }
        )
    }
}

impl fmt::Display for EquivalenceReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test validation report passed() method
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_equivalence_report_passed() {
        let pass_report = EquivalenceReport {
            max_relative_error: 1e-13,
            divergent_points: 0,
            passed: true,
            pass_threshold: 1e-12,
            total_points: 1000000,
            ..EquivalenceReport::new(1e-12, 1000000)
        };
        assert!(pass_report.passed());

        let fail_report = EquivalenceReport {
            max_relative_error: 1e-11,
            divergent_points: 1,
            passed: false,
            pass_threshold: 1e-12,
            total_points: 1000000,
            ..EquivalenceReport::new(1e-12, 1000000)
        };
        assert!(!fail_report.passed());
    }

    /// Test report display formatting
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_equivalence_report_display() {
        let report = EquivalenceReport {
            max_absolute_error: 1e-15,
            max_relative_error: 1e-14,
            total_points: 1000,
            divergent_points: 0,
            cpu_time_ms: 100.0,
            gpu_time_ms: 10.0,
            speedup: 10.0,
            passed: true,
            ..EquivalenceReport::new(1e-12, 1000)
        };

        let display_str = format!("{}", report);
        assert!(display_str.contains("PASSED"));
        assert!(display_str.contains("10.00"));
    }

    /// Test divergent fraction calculation
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_divergent_fraction() {
        let mut report = EquivalenceReport::new(1e-12, 1000);
        report.divergent_points = 10;

        assert!((report.divergent_fraction() - 0.01).abs() < 1e-10);
        assert_eq!(report.divergent_fraction(), 0.01);
    }

    /// Test empty report divergent fraction
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_divergent_fraction_empty() {
        let report = EquivalenceReport::new(1e-12, 0);
        assert_eq!(report.divergent_fraction(), 0.0);
    }
}
