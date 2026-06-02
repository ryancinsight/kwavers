//! Individual theorem validation methods.

mod imaging;
mod ml;
mod physics;
mod spectral;

use super::{TheoremValidation, TheoremValidator};

impl TheoremValidator {
    pub(super) fn calculate_confidence(error_bound: f64, measured_error: f64, passed: bool) -> f64 {
        if !passed {
            return 0.2;
        }
        if error_bound <= 0.0 {
            return 0.5;
        }
        let error_ratio = measured_error / error_bound;
        if error_ratio < 0.1 {
            0.95
        } else if error_ratio < 0.3 {
            0.85
        } else if error_ratio < 0.7 {
            0.7
        } else {
            0.5
        }
    }

    /// Validate Beer-Lambert law for attenuation
    pub fn validate_beer_lambert_law(
        initial_intensity: f64,
        absorption_coeff: f64,
        distances: &[f64],
        measured_intensities: &[f64],
    ) -> TheoremValidation {
        let mut max_error: f64 = 0.0;
        let mut total_error = 0.0;

        for (&distance, &measured) in distances.iter().zip(measured_intensities.iter()) {
            let theoretical = initial_intensity * (-absorption_coeff * distance).exp();
            let error = (measured - theoretical).abs() / theoretical.abs().max(1e-10);
            max_error = max_error.max(error);
            total_error += error;
        }

        let _avg_error = total_error / distances.len() as f64;
        let passed = max_error < 0.05;

        let max_distance = distances.iter().copied().fold(0.0, f64::max);
        let theoretical_bound = absorption_coeff * max_distance * max_distance / 2.0
            * (-absorption_coeff * max_distance).exp();

        TheoremValidation {
            theorem: "Beer-Lambert Law".to_owned(),
            passed,
            error_bound: theoretical_bound,
            measured_error: max_error,
            confidence: Self::calculate_confidence(theoretical_bound, max_error, passed),
            details: format!(
                "Validated over {} points, max error: {:.2e}, theoretical bound: {:.2e}",
                distances.len(),
                max_error,
                theoretical_bound
            ),
        }
    }

    /// Validate Courant-Friedrichs-Lewy (CFL) condition for FDTD
    #[must_use]
    pub fn validate_cfl_condition(
        time_step: f64,
        spatial_step: f64,
        wave_speed: f64,
        dimensions: usize,
    ) -> TheoremValidation {
        let cfl_number = time_step * wave_speed / spatial_step;
        let cfl_limit = 1.0 / (dimensions as f64).sqrt();

        let passed = cfl_number <= cfl_limit * 0.95;
        let stability_margin = cfl_limit - cfl_number;

        TheoremValidation {
            theorem: "CFL Condition".to_owned(),
            passed,
            error_bound: cfl_limit,
            measured_error: cfl_number,
            confidence: if passed { 0.99 } else { 0.1 },
            details: format!(
                "CFL number: {:.3}, limit: {:.3}, stability margin: {:.3}",
                cfl_number, cfl_limit, stability_margin
            ),
        }
    }
}
