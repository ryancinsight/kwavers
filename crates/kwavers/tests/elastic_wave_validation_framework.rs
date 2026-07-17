//! Shared validation framework for autodiff elastic-wave solvers.
//!
//! This module provides rigorous mathematical validation tests that can be applied
//! to implementations of `AutodiffElasticWaveEquation`, including:
//! - Physics-Informed Neural Networks (PINN)
//! - Neural operators
//!
//! # Design Principles
//!
//! 1. **Mathematical Rigor**: All validations are derived from theoretical guarantees
//! 2. **Solver Agnostic**: Tests operate on the autodiff trait interface
//! 3. **Analytical Ground Truth**: Compare against closed-form solutions where possible
//! 4. **Current Coverage**: Material properties and wave speeds
//!
//! # Validation Hierarchy
//!
//! Level 1: Material Property Validation (bounds, positivity, consistency)
//! Level 2: Wave Speed Validation (P-wave, S-wave formulae)
//! Level 3: Analytical plane-wave reference identities

use kwavers_physics::foundations::wave_equation::AutodiffElasticWaveEquation;

// ============================================================================
// Validation Result Types
// ============================================================================

/// Result of a validation test with quantitative error metrics
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub passed: bool,
    pub test_name: String,
    pub error_l2: f64,
    pub error_linf: f64,
    pub tolerance: f64,
    pub details: String,
}

impl ValidationResult {
    pub fn success(test_name: impl Into<String>, details: impl Into<String>) -> Self {
        Self {
            passed: true,
            test_name: test_name.into(),
            error_l2: 0.0,
            error_linf: 0.0,
            tolerance: 0.0,
            details: details.into(),
        }
    }

    pub fn failure(
        test_name: impl Into<String>,
        error_l2: f64,
        error_linf: f64,
        tolerance: f64,
        details: impl Into<String>,
    ) -> Self {
        Self {
            passed: false,
            test_name: test_name.into(),
            error_l2,
            error_linf,
            tolerance,
            details: details.into(),
        }
    }

    pub fn with_metrics(
        test_name: impl Into<String>,
        error_l2: f64,
        error_linf: f64,
        tolerance: f64,
    ) -> Self {
        let passed = error_l2 <= tolerance && error_linf <= tolerance;
        Self {
            passed,
            test_name: test_name.into(),
            error_l2,
            error_linf,
            tolerance,
            details: format!(
                "L2 error: {:.3e}, L∞ error: {:.3e}, tolerance: {:.3e}",
                error_l2, error_linf, tolerance
            ),
        }
    }
}

// ============================================================================
// Level 1: Material Property Validation
// ============================================================================

/// Validate material properties for autodiff-based solvers
///
/// This is a mirror of `validate_material_properties` for solvers that implement
/// `AutodiffElasticWaveEquation` instead of `ElasticWaveEquation`.
pub fn validate_material_properties_autodiff<T: AutodiffElasticWaveEquation>(
    solver: &T,
) -> ValidationResult {
    let lambda = solver.lame_lambda();
    let mu = solver.lame_mu();
    let rho = solver.density();

    // Check density positivity
    let rho_min = rho.iter().cloned().fold(f64::INFINITY, f64::min);
    if rho_min <= 0.0 {
        return ValidationResult::failure(
            "material_properties",
            0.0,
            rho_min,
            0.0,
            format!("Density must be positive. Found min(ρ) = {:.3e}", rho_min),
        );
    }

    // Check shear modulus positivity
    let mu_min = mu.iter().cloned().fold(f64::INFINITY, f64::min);
    if mu_min <= 0.0 {
        return ValidationResult::failure(
            "material_properties",
            0.0,
            mu_min,
            0.0,
            format!(
                "Shear modulus must be positive. Found min(μ) = {:.3e}",
                mu_min
            ),
        );
    }

    // Check thermodynamic stability: λ > -2μ/3
    let mut max_violation: f64 = 0.0;
    for (l, m) in lambda.iter().zip(mu.iter()) {
        let bound = -2.0 * m / 3.0;
        if l < &bound {
            max_violation = max_violation.max(bound - l);
        }
    }
    if max_violation > 1e-10 {
        return ValidationResult::failure(
            "material_properties",
            0.0,
            max_violation,
            1e-10,
            format!(
                "Thermodynamic stability violated: λ must be > -2μ/3. Max violation: {:.3e}",
                max_violation
            ),
        );
    }

    // Check bulk modulus positivity: K = λ + 2μ/3 > 0
    let k_min = lambda
        .iter()
        .zip(mu.iter())
        .map(|(l, m)| l + 2.0 * m / 3.0)
        .fold(f64::INFINITY, f64::min);
    if k_min <= 0.0 {
        return ValidationResult::failure(
            "material_properties",
            0.0,
            k_min,
            0.0,
            format!(
                "Bulk modulus K = λ + 2μ/3 must be positive. Found min(K) = {:.3e}",
                k_min
            ),
        );
    }

    // Check Poisson's ratio bounds: -1 < ν < 0.5
    // ν = λ / (2(λ + μ))
    for (l, m) in lambda.iter().zip(mu.iter()) {
        let nu = l / (2.0 * (l + m));
        if nu <= -1.0 || nu >= 0.5 {
            return ValidationResult::failure(
                "material_properties",
                0.0,
                nu,
                0.5,
                format!(
                    "Poisson's ratio ν = {:.3e} outside physical range (-1, 0.5)",
                    nu
                ),
            );
        }
    }

    ValidationResult::success(
        "material_properties",
        format!(
            "All material properties physical: ρ ∈ [{:.3e}, {:.3e}], μ ∈ [{:.3e}, {:.3e}], K > 0",
            rho_min,
            rho.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            mu_min,
            mu.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        ),
    )
}

// ============================================================================
// Level 2: Wave Speed Validation
// ============================================================================

/// Validate wave speeds for autodiff-based solvers
///
/// This is a mirror of `validate_wave_speeds` for solvers that implement
/// `AutodiffElasticWaveEquation` instead of `ElasticWaveEquation`.
pub fn validate_wave_speeds_autodiff<T: AutodiffElasticWaveEquation>(
    solver: &T,
    tolerance: f64,
) -> ValidationResult {
    let lambda = solver.lame_lambda();
    let mu = solver.lame_mu();
    let rho = solver.density();

    let cp_computed = solver.p_wave_speed();
    let cs_computed = solver.s_wave_speed();

    // Compute analytical wave speeds
    let cp_analytical = (&(&lambda + &(&mu * 2.0)) / &rho).mapv(f64::sqrt);
    let cs_analytical = (&mu / &rho).mapv(f64::sqrt);

    // Compute errors
    let cp_error = &cp_computed - &cp_analytical;
    let cs_error = &cs_computed - &cs_analytical;

    let cp_l2 = (cp_error.iter().map(|x| x * x).sum::<f64>() / cp_error.len() as f64).sqrt();
    let cs_l2 = (cs_error.iter().map(|x| x * x).sum::<f64>() / cs_error.len() as f64).sqrt();

    let cp_linf = cp_error
        .iter()
        .cloned()
        .fold(0.0_f64, |a, b| a.max(b.abs()));
    let cs_linf = cs_error
        .iter()
        .cloned()
        .fold(0.0_f64, |a, b| a.max(b.abs()));

    let total_l2 = (cp_l2 * cp_l2 + cs_l2 * cs_l2).sqrt();
    let total_linf = cp_linf.max(cs_linf);

    // Check P-wave > S-wave relationship
    for (cp, cs) in cp_computed.iter().zip(cs_computed.iter()) {
        if cp <= cs {
            return ValidationResult::failure(
                "wave_speeds",
                total_l2,
                total_linf,
                tolerance,
                format!(
                    "P-wave speed must exceed S-wave speed. Found cₚ = {:.3e} ≤ cₛ = {:.3e}",
                    cp, cs
                ),
            );
        }
    }

    ValidationResult::with_metrics("wave_speeds", total_l2, total_linf, tolerance)
}

// ============================================================================
// Level 3: PDE Residual Validation (Plane Wave Solution)
// ============================================================================

/// Analytical plane wave solution for elastic wave equation
///
/// # Theory
///
/// For a homogeneous isotropic medium, plane wave solutions exist:
///
/// **P-wave (longitudinal)**:
/// u(x,t) = A êₖ exp(i(k·x - ωₚt))
/// where ωₚ = cₚ|k|, êₖ = k/|k| (parallel to k)
///
/// **S-wave (transverse)**:
/// u(x,t) = A ê⊥ exp(i(k·x - ωₛt))
/// where ωₛ = cₛ|k|, ê⊥ ⊥ k (perpendicular to k)
pub struct PlaneWaveSolution {
    pub wave_type: WaveType,
    pub wave_vector: [f64; 2], // k = [kₓ, kᵧ]
    pub amplitude: f64,
    pub lambda: f64,
    pub mu: f64,
    pub rho: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveType {
    PWave, // Longitudinal (compression)
    SWave, // Transverse (shear)
}

impl PlaneWaveSolution {
    /// Create P-wave solution
    pub fn p_wave(wave_vector: [f64; 2], amplitude: f64, lambda: f64, mu: f64, rho: f64) -> Self {
        Self {
            wave_type: WaveType::PWave,
            wave_vector,
            amplitude,
            lambda,
            mu,
            rho,
        }
    }

    /// Create S-wave solution
    pub fn s_wave(wave_vector: [f64; 2], amplitude: f64, lambda: f64, mu: f64, rho: f64) -> Self {
        Self {
            wave_type: WaveType::SWave,
            wave_vector,
            amplitude,
            lambda,
            mu,
            rho,
        }
    }

    /// Wave speed
    pub fn speed(&self) -> f64 {
        match self.wave_type {
            WaveType::PWave => ((self.lambda + 2.0 * self.mu) / self.rho).sqrt(),
            WaveType::SWave => (self.mu / self.rho).sqrt(),
        }
    }

    /// Angular frequency ω = c|k|
    pub fn angular_frequency(&self) -> f64 {
        let k_mag = (self.wave_vector[0].powi(2) + self.wave_vector[1].powi(2)).sqrt();
        self.speed() * k_mag
    }

    /// Polarization direction (unit vector parallel to displacement)
    pub fn polarization(&self) -> [f64; 2] {
        let k_mag = (self.wave_vector[0].powi(2) + self.wave_vector[1].powi(2)).sqrt();
        match self.wave_type {
            WaveType::PWave => {
                // Parallel to k
                [self.wave_vector[0] / k_mag, self.wave_vector[1] / k_mag]
            }
            WaveType::SWave => {
                // Perpendicular to k
                [-self.wave_vector[1] / k_mag, self.wave_vector[0] / k_mag]
            }
        }
    }

    /// Evaluate displacement at (x, y, t)
    pub fn displacement(&self, x: f64, y: f64, t: f64) -> [f64; 2] {
        let phase =
            self.wave_vector[0] * x + self.wave_vector[1] * y - self.angular_frequency() * t;
        let polarization = self.polarization();
        let amplitude = self.amplitude * phase.cos();
        [polarization[0] * amplitude, polarization[1] * amplitude]
    }

    /// Evaluate velocity at (x, y, t)
    pub fn velocity(&self, x: f64, y: f64, t: f64) -> [f64; 2] {
        let omega = self.angular_frequency();
        let phase = self.wave_vector[0] * x + self.wave_vector[1] * y - omega * t;
        let polarization = self.polarization();
        let amplitude = self.amplitude * omega * phase.sin();
        [polarization[0] * amplitude, polarization[1] * amplitude]
    }

    /// Evaluate acceleration at (x, y, t) - for PDE residual check
    pub fn acceleration(&self, x: f64, y: f64, t: f64) -> [f64; 2] {
        let omega = self.angular_frequency();
        let phase = self.wave_vector[0] * x + self.wave_vector[1] * y - omega * t;
        let polarization = self.polarization();
        let amplitude = -self.amplitude * omega.powi(2) * phase.cos();
        [polarization[0] * amplitude, polarization[1] * amplitude]
    }

    /// Compute spatial derivatives (for stress calculation)
    pub fn displacement_gradient(&self, x: f64, y: f64, t: f64) -> [[f64; 2]; 2] {
        let phase =
            self.wave_vector[0] * x + self.wave_vector[1] * y - self.angular_frequency() * t;
        let polarization = self.polarization();
        let sin_phase = phase.sin();

        // ∂uᵢ/∂xⱼ
        let scale = -self.amplitude * sin_phase;
        [
            [
                scale * polarization[0] * self.wave_vector[0],
                scale * polarization[0] * self.wave_vector[1],
            ],
            [
                scale * polarization[1] * self.wave_vector[0],
                scale * polarization[1] * self.wave_vector[1],
            ],
        ]
    }
}

/// Comprehensive validation test suite for autodiff-based solvers
///
/// This is a mirror of `run_full_validation_suite` for solvers that implement
/// `AutodiffElasticWaveEquation` instead of `ElasticWaveEquation`.
pub fn run_full_validation_suite_autodiff<T: AutodiffElasticWaveEquation>(
    solver: &T,
    test_name: &str,
) -> Vec<ValidationResult> {
    let results = vec![
        validate_material_properties_autodiff(solver),
        validate_wave_speeds_autodiff(solver, 1e-12),
    ];

    // Report summary
    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();
    println!(
        "\n{} Validation Summary: {}/{} tests passed",
        test_name, passed, total
    );

    for result in &results {
        let status = if result.passed { "✓" } else { "✗" };
        println!("  {} {}: {}", status, result.test_name, result.details);
    }

    results
}

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plane_wave_solution_properties() {
        let lambda = 1e9; // Pa
        let mu = 0.5e9; // Pa
        let rho = 2000.0; // kg/m³

        let solution = PlaneWaveSolution::p_wave([1.0, 0.0], 1e-6, lambda, mu, rho);

        // P-wave speed
        let cp = ((lambda + 2.0 * mu) / rho).sqrt();
        assert!((solution.speed() - cp).abs() < 1e-10);

        // Frequency
        let omega = solution.angular_frequency();
        assert!((omega - cp).abs() < 1e-10); // |k| = 1

        // Polarization parallel to k
        let pol = solution.polarization();
        assert!((pol[0] - 1.0).abs() < 1e-10);
        assert!(pol[1].abs() < 1e-10);
    }

    #[test]
    fn test_s_wave_polarization() {
        let solution = PlaneWaveSolution::s_wave([1.0, 0.0], 1e-6, 1e9, 0.5e9, 2000.0);

        // S-wave polarization perpendicular to k
        let pol = solution.polarization();
        let k_norm = [1.0, 0.0];
        let dot = pol[0] * k_norm[0] + pol[1] * k_norm[1];
        assert!(dot.abs() < 1e-10);
    }
}
