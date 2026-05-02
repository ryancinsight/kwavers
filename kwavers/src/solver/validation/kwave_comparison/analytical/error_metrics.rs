use ndarray::ArrayView3;

/// Error metrics for comparing numerical and analytical solutions.
///
/// # Mathematical Definitions
///
/// ```text
/// L² error:    ε₂ = √(∫(p_num − p_ana)² dV / ∫p_ana² dV)
/// L∞ error:    ε∞ = max|p_num − p_ana| / max|p_ana|
/// Phase error: Δφ = acos(∫p_num·p_ana dV / √(∫p_num²·∫p_ana²))
/// ```
///
/// # Acceptance Criteria (k-Wave Baseline)
///
/// - L² error < 0.01 (1%)
/// - L∞ error < 0.05 (5%)
/// - Phase error < 0.1 rad (5.7°)
#[derive(Debug, Clone)]
pub struct ErrorMetrics {
    pub l2_error: f64,
    pub linf_error: f64,
    pub phase_error: f64,
    pub correlation: f64,
}

impl ErrorMetrics {
    /// Compute error metrics between numerical and analytical solutions.
    pub fn compute(numerical: ArrayView3<f64>, analytical: ArrayView3<f64>) -> Self {
        assert_eq!(
            numerical.dim(),
            analytical.dim(),
            "Arrays must have same dimensions"
        );

        let mut l2_num = 0.0f64;
        let mut l2_ana = 0.0f64;
        let mut linf_error = 0.0f64;
        let mut correlation_num = 0.0f64;

        let ana_max = analytical
            .iter()
            .map(|x| x.abs())
            .fold(0.0f64, |a, b| a.max(b));

        for (n, a) in numerical.iter().zip(analytical.iter()) {
            let diff = n - a;
            l2_num += diff * diff;
            l2_ana += a * a;
            correlation_num += n * a;
            linf_error = linf_error.max(diff.abs() / ana_max.max(1e-10));
        }

        let l2_error = if l2_ana > 0.0 {
            (l2_num / l2_ana).sqrt()
        } else {
            0.0
        };

        let num_norm = numerical.iter().map(|x| x * x).sum::<f64>().sqrt();
        let ana_norm = analytical.iter().map(|x| x * x).sum::<f64>().sqrt();
        let correlation = if num_norm > 0.0 && ana_norm > 0.0 {
            correlation_num / (num_norm * ana_norm)
        } else {
            0.0
        };

        let phase_error = if correlation.abs() <= 1.0 {
            correlation.acos()
        } else {
            0.0
        };

        Self {
            l2_error,
            linf_error,
            phase_error,
            correlation,
        }
    }

    /// Returns true if L² < 0.01, L∞ < 0.05, phase < 0.1 rad.
    pub fn meets_acceptance_criteria(&self) -> bool {
        self.l2_error < 0.01 && self.linf_error < 0.05 && self.phase_error < 0.1
    }

    /// Generate validation report string.
    pub fn report(&self) -> String {
        format!(
            "Error Metrics:\n\
             - L² error:    {:.4e} ({})\n\
             - L∞ error:    {:.4e} ({})\n\
             - Phase error: {:.4} rad = {:.2}° ({})\n\
             - Correlation: {:.6} ({})\n\
             Overall: {}",
            self.l2_error,
            if self.l2_error < 0.01 { "✓" } else { "✗" },
            self.linf_error,
            if self.linf_error < 0.05 { "✓" } else { "✗" },
            self.phase_error,
            self.phase_error.to_degrees(),
            if self.phase_error < 0.1 { "✓" } else { "✗" },
            self.correlation,
            if self.correlation > 0.99 {
                "✓"
            } else {
                "✗"
            },
            if self.meets_acceptance_criteria() {
                "PASS ✓"
            } else {
                "FAIL ✗"
            }
        )
    }
}
