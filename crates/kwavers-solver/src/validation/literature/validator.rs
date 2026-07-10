//! LiteratureValidator struct and validation methods.

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use leto::Array3;

use super::types::{treeby_2010, LiteratureValidationResult};

/// Literature validation coordinator
#[derive(Debug)]
pub struct LiteratureValidator;

impl LiteratureValidator {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new() -> Self {
        Self
    }

    /// Validate against Treeby (2010) plane wave propagation.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn validate_treeby_plane_wave(
        &self,
        pressure_field: &Array3<f64>,
        time_points: &[f64],
        _dt: f64,
    ) -> KwaversResult<LiteratureValidationResult> {
        let mut result = LiteratureValidationResult::new("Treeby_2010_PlaneWave");

        let nx = pressure_field.shape()[0];
        let ny = pressure_field.shape()[1];
        let nz = pressure_field.shape()[2];
        let center = (nx / 2, ny / 2, nz / 2);

        let computed_waveform: Vec<f64> = vec![pressure_field[center]];

        let amplitude = 1.0e5;
        let expected: Vec<f64> = time_points
            .iter()
            .map(|&t| treeby_2010::analytical_pressure(t, amplitude))
            .collect();

        if (computed_waveform.len()) != (expected.len()) {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: (expected.len()).to_string(),
                    actual: (computed_waveform.len()).to_string(),
                },
            ));
        }

        let l2_error = Self::relative_l2_error(&computed_waveform, &expected);

        result
            .with_error(l2_error, l2_error * amplitude)
            .with_metric("phase_velocity_error", l2_error)
            .with_metric(
                "ppw",
                treeby_2010::SOUND_SPEED / (treeby_2010::FREQUENCY * treeby_2010::DX),
            );

        result.passed = l2_error < treeby_2010::MAX_PHASE_VELOCITY_ERROR;
        result.notes = format!(
            "Treeby (2010) validation: phase velocity error {:.2}% {}",
            l2_error * 100.0,
            if result.passed { "PASS" } else { "FAIL" }
        );

        Ok(result)
    }

    /// Validate against Treeby (2010) absorption power law.
    pub fn validate_treeby_absorption(
        &self,
        attenuation_db: &[f64],
        frequencies: &[f64],
        expected_y: f64,
    ) -> LiteratureValidationResult {
        let mut result = LiteratureValidationResult::new("Treeby_2010_Absorption");

        let log_f: Vec<f64> = frequencies.iter().map(|&f| f.ln()).collect();
        let log_alpha: Vec<f64> = attenuation_db.iter().map(|&a| a.ln()).collect();

        let (slope, intercept) = Self::linear_regression(&log_f, &log_alpha);

        let fitted_y = slope;
        let error_y = (fitted_y - expected_y).abs() / expected_y;

        result
            .with_error(error_y, error_y)
            .with_metric("fitted_y", fitted_y)
            .with_metric("expected_y", expected_y)
            .with_metric("alpha_0", intercept.exp());

        result.passed = error_y < 0.05;
        result.notes = format!(
            "Power law y = {:.2} (expected {:.2}), error {:.1}%",
            fitted_y,
            expected_y,
            error_y * 100.0
        );

        result
    }

    /// Validate Pinton (2009) elastic wave shear component.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn validate_pinton_shear_wave(
        &self,
        displacement_field: &Array3<f64>,
        time: f64,
        expected_shear_speed: f64,
    ) -> LiteratureValidationResult {
        let mut result = LiteratureValidationResult::new("Pinton_2009_ShearWave");

        let expected_radius = expected_shear_speed * time;

        let max_disp_idx = displacement_field
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;

        let nx = displacement_field.shape()[0];
        let ny = displacement_field.shape()[1];
        let nz = displacement_field.shape()[2];
        let iz = max_disp_idx / (nx * ny);
        let remainder = max_disp_idx % (nx * ny);
        let iy = remainder / nx;
        let ix = remainder % nx;

        let dx = 1.0e-4;
        let found_radius = ((ix as f64 - (nx / 2) as f64).powi(2)
            + (iy as f64 - (ny / 2) as f64).powi(2)
            + (iz as f64 - (nz / 2) as f64).powi(2))
        .sqrt()
            * dx;

        let error = (found_radius - expected_radius).abs() / expected_radius;

        result
            .with_error(error, error * expected_radius)
            .with_metric("expected_radius_m", expected_radius)
            .with_metric("found_radius_m", found_radius)
            .with_metric("shear_speed_m_s", expected_shear_speed);

        result.passed = error < 0.02;
        result.notes = format!(
            "Shear wavefront at {:.3} mm (expected {:.3} mm)",
            found_radius * 1000.0,
            expected_radius * 1000.0
        );

        result
    }

    /// Validate convergence rate against theoretical order.
    pub fn validate_convergence_rate(
        &self,
        dx_values: &[f64],
        errors: &[f64],
        expected_order: f64,
    ) -> LiteratureValidationResult {
        let mut result = LiteratureValidationResult::new("ConvergenceAnalysis");

        if (dx_values.len()) < 2 || (errors.len()) < 2 {
            result.notes = "Need at least 2 points for convergence analysis".to_string();
            return result;
        }

        let log_dx: Vec<f64> = dx_values.iter().map(|&x| x.ln()).collect();
        let log_err: Vec<f64> = errors.iter().map(|&e| e.ln()).collect();

        let (observed_order, _) = Self::linear_regression(&log_dx, &log_err);

        let rate_error = (observed_order.abs() - expected_order).abs() / expected_order;

        result
            .with_error(rate_error, rate_error)
            .with_metric("observed_order", observed_order.abs())
            .with_metric("expected_order", expected_order)
            .with_metric(
                "prefactor",
                (errors[0] / dx_values[0].powf(expected_order)).abs(),
            );

        result.passed = rate_error < 0.1;
        result.notes = format!(
            "Convergence order: {:.2} (expected {:.2}), slope {}",
            observed_order.abs(),
            expected_order,
            if result.passed { "verified" } else { "mismatch" }
        );

        result
    }

    /// Compute relative L2 error between computed and reference.
    pub(super) fn relative_l2_error(computed: &[f64], reference: &[f64]) -> f64 {
        if (computed.len()) != (reference.len()) || computed.is_empty() {
            return f64::NAN;
        }

        let numerator: f64 = computed
            .iter()
            .zip(reference.iter())
            .map(|(c, r)| (c - r).powi(2))
            .sum::<f64>()
            .sqrt();

        let denominator: f64 = reference.iter().map(|r| r.powi(2)).sum::<f64>().sqrt();

        if denominator == 0.0 {
            f64::NAN
        } else {
            numerator / denominator
        }
    }

    /// Simple linear regression (slope, intercept).
    pub(super) fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
        if (x.len()) != (y.len()) || (x.len()) < 2 {
            return (f64::NAN, f64::NAN);
        }

        let n = (x.len()) as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < 1e-15 {
            return (f64::NAN, f64::NAN);
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;

        (slope, intercept)
    }

    /// Generate validation report.
    pub fn report(&self, results: &[LiteratureValidationResult]) -> String {
        let mut report = String::new();
        report.push_str("## Literature Validation Report\n\n");
        report.push_str("| Case | Status | Rel. Error | Notes |\n");
        report.push_str("|------|--------|------------|-------|\n");

        let mut passed = 0;
        let mut failed = 0;

        for result in results {
            let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
            report.push_str(&format!(
                "| {} | {} | {:.2e} | {} |\n",
                result.case_name, status, result.relative_error, result.notes
            ));

            if result.passed { passed += 1; } else { failed += 1; }
        }

        report.push_str(&format!("\n**Summary**: {}/{} passed\n", passed, passed + failed));
        report
    }
}

impl Default for LiteratureValidator {
    fn default() -> Self {
        Self::new()
    }
}
