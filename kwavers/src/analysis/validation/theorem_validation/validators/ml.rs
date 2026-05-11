//! Machine learning convergence theorem validators (PINN).

use super::super::{TheoremValidation, TheoremValidator};

impl TheoremValidator {
    /// Validate PINN convergence theorem: ‖u − u_PINN‖_H1 ≤ C (log N / N)^{1/4} + C W^{-1/2}
    #[must_use]
    pub fn validate_pinn_convergence(
        n_collocation: usize,
        network_width: usize,
        measured_error: f64,
        solution_smoothness: f64,
    ) -> TheoremValidation {
        let n_term = (n_collocation as f64).ln() / (n_collocation as f64);
        let convergence_term = n_term.powf(0.25);
        let width_term = 1.0 / (network_width as f64).sqrt();
        let theoretical_bound = solution_smoothness * (convergence_term + width_term);

        let passed = measured_error <= theoretical_bound * 2.0;
        let relative_error = measured_error / theoretical_bound;

        TheoremValidation {
            theorem: "PINN Convergence Theorem".to_owned(),
            passed,
            error_bound: theoretical_bound,
            measured_error,
            confidence: if passed { 0.8 } else { 0.4 },
            details: format!(
                "N: {}, Width: {}, Theoretical bound: {:.2e}, Measured: {:.2e}, Ratio: {:.2}",
                n_collocation, network_width, theoretical_bound, measured_error, relative_error
            ),
        }
    }
}
