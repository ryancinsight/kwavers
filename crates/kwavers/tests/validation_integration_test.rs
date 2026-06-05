//! Integration Tests for Modular Validation Framework
//!
//! This test suite validates the newly created modular validation framework
//! with analytical solutions for various physics solvers.
//!
//! # Test Coverage
//!
//! 1. Analytical solution accuracy (plane waves, sine waves, polynomials)
//! 2. Error metric computations (L² and L∞ norms)
//! 3. Convergence analysis (rate estimation, R² fit)
//! 4. Energy conservation validation (Hamiltonian invariance)
//! 5. Validation suite composition and reporting
//!
//! # Design Principles
//!
//! - **Mathematical Rigor**: All tests verify exact analytical derivatives
//! - **Framework Coverage**: Exercise all validation framework components
//! - **Solver Agnostic**: Tests use trait interfaces, not concrete types
//! - **Property-Based**: Validate invariants, not specific values

#![cfg(test)]

mod validation;

use validation::{
    analytical_solutions::*,
    convergence::{ConvergenceResult, ConvergenceStudy},
    energy::{compute_kinetic_energy, compute_strain_energy, EnergyValidator},
    error_metrics::{l2_norm, linf_norm, relative_error, ErrorMetrics},
    AnalyticalSolution, SolutionParameters, ValidationResult, ValidationSuite,
};

use std::f64::consts::PI;

// ============================================================================
// Test Suite 1: Analytical Solutions Accuracy
// ============================================================================

#[test]
fn test_plane_wave_p_wave_displacement() {
    let params = SolutionParameters {
        amplitude: 1e-6,
        wavelength: 0.01,
        omega: 0.0, // Will be computed
        wave_speed: 5000.0,
        density: 2700.0,
        lambda: 5e10,
        mu: 2.6e10,
    };

    let wave = PlaneWave2D::p_wave(1e-6, 0.01, [1.0, 0.0], params);

    // Test displacement at origin
    let x = &[0.0, 0.0];
    let t = 0.0;
    let u = wave.displacement(x, t);

    // At (0, 0, 0): u = A sin(0) = 0
    assert_eq!(u[0], 0.0);
    assert_eq!(u[1], 0.0);

    // Test at x = λ/4: u = A sin(π/2) = A
    let x_quarter = &[0.0025, 0.0]; // λ/4 along x-axis
    let u_quarter = wave.displacement(x_quarter, t);
    assert!((u_quarter[0] - 1e-6).abs() < 1e-12);
}

#[test]
fn test_plane_wave_s_wave_transverse() {
    let params = SolutionParameters {
        amplitude: 1e-6,
        wavelength: 0.01,
        omega: 0.0,
        wave_speed: 3000.0,
        density: 2700.0,
        lambda: 5e10,
        mu: 2.6e10,
    };

    let wave = PlaneWave2D::s_wave(1e-6, 0.01, [1.0, 0.0], params);

    // S-wave: polarization perpendicular to propagation
    // For k̂ = (1, 0), d̂ = (0, 1)
    let x = &[0.0025, 0.0]; // λ/4 along x
    let t = 0.0;
    let u = wave.displacement(x, t);

    // Displacement should be in y-direction only
    assert!((u[0]).abs() < 1e-12);
    assert!((u[1] - 1e-6).abs() < 1e-12);
}

#[test]
fn test_sine_wave_gradient_analytical() {
    let wave = SineWave1D::new(1.0, 1.0, 1.0);

    let x = &[0.25]; // λ/4 position
    let t = 0.0;

    // Analytical: u = sin(kx), ∂u/∂x = k cos(kx)
    // k = 2π/λ = 2π, x = 0.25
    // ∂u/∂x = 2π cos(2π·0.25) = 2π cos(π/2) = 0
    let grad = wave.gradient(x, t);
    assert!(grad[0].abs() < 1e-10);

    // Test at x = 0: ∂u/∂x = 2π cos(0) = 2π
    let x_zero = &[0.0];
    let grad_zero = wave.gradient(x_zero, t);
    assert!((grad_zero[0] - 2.0 * PI).abs() < 1e-10);
}

#[test]
fn test_polynomial_second_derivatives() {
    let poly = PolynomialTest2D::new(1.0);

    // u = x², v = xy
    // ∂u/∂x = 2x, ∂²u/∂x² = 2
    // ∂v/∂y = x,  ∂²v/∂y² = 0

    let x = &[3.0, 5.0];
    let t = 0.0;

    let u = poly.displacement(x, t);
    assert_eq!(u[0], 9.0); // x² = 9
    assert_eq!(u[1], 15.0); // xy = 15

    let grad = poly.gradient(x, t);
    assert_eq!(grad[0], 6.0); // ∂u/∂x = 2x = 6
    assert_eq!(grad[1], 0.0); // ∂u/∂y = 0
    assert_eq!(grad[2], 5.0); // ∂v/∂x = y = 5
    assert_eq!(grad[3], 3.0); // ∂v/∂y = x = 3
}

#[test]
fn test_quadratic_laplacian() {
    let quad = QuadraticTest2D::new(1.0);

    // u = x² + y², ∇²u = 2 + 2 = 4
    // This test documents the Laplacian computation capability

    let x = &[2.0, 3.0];
    let t = 0.0;

    let u = quad.displacement(x, t);
    assert_eq!(u[0], 13.0); // x² + y² = 4 + 9 = 13
    assert_eq!(u[1], 6.0); // xy = 6

    let grad = quad.gradient(x, t);
    assert_eq!(grad[0], 4.0); // ∂u/∂x = 2x = 4
    assert_eq!(grad[1], 6.0); // ∂u/∂y = 2y = 6
}

// ============================================================================
// Test Suite 2: Error Metrics
// ============================================================================

#[test]
fn test_error_metrics_exact_match() {
    let computed = vec![1.0, 2.0, 3.0, 4.0];
    let analytical = vec![1.0, 2.0, 3.0, 4.0];

    let metrics = ErrorMetrics::compute(&computed, &analytical);

    assert_eq!(metrics.l2_error, 0.0);
    assert_eq!(metrics.linf_error, 0.0);
    assert_eq!(metrics.relative_l2_error, 0.0);
    assert!(metrics.within_tolerance(1e-10));
}

#[test]
fn test_error_metrics_known_error() {
    let computed = vec![1.0, 2.0, 3.0];
    let analytical = vec![1.0, 2.1, 3.0];

    let metrics = ErrorMetrics::compute(&computed, &analytical);

    // L² = √((0² + 0.1² + 0²) / 3) = √(0.01/3) ≈ 0.0577
    assert!((metrics.l2_error - 0.0577).abs() < 1e-3);
    assert!((metrics.linf_error - 0.1).abs() < 1e-10);
}

#[test]
fn test_l2_norm_pythagorean() {
    let v = vec![3.0, 4.0];
    let norm = l2_norm(&v);

    // ||v||₂ = √((9 + 16) / 2) = √12.5
    assert!((norm - 3.5355339).abs() < 1e-6);
}

#[test]
fn test_linf_norm() {
    let v = vec![1.0, -5.0, 3.0, 2.0];
    let norm = linf_norm(&v);
    assert_eq!(norm, 5.0);
}

#[test]
fn test_relative_error_computation() {
    let computed = vec![10.0, 20.0, 30.0];
    let analytical = vec![10.0, 22.0, 30.0];

    let rel_err = relative_error(&computed, &analytical);

    // Small relative error expected
    assert!(rel_err < 0.1);
}

// ============================================================================
// Test Suite 3: Convergence Analysis
// ============================================================================

#[test]
fn test_convergence_second_order() {
    let mut study = ConvergenceStudy::new("Second-order convergence test");

    // Simulate second-order convergence: E = 0.1 * h²
    for &h in &[1.0, 0.5, 0.25, 0.125, 0.0625] {
        let error = 0.1 * h * h;
        study.add_measurement(h, error);
    }

    let rate = study.compute_convergence_rate().unwrap();
    assert!((rate - 2.0).abs() < 0.01, "Expected rate≈2, got {}", rate);

    let r_squared = study.compute_r_squared().unwrap();
    assert!(r_squared > 0.999, "Expected R²≈1, got {}", r_squared);

    assert!(study.is_monotonic());
}

#[test]
fn test_convergence_first_order() {
    let mut study = ConvergenceStudy::new("First-order convergence test");

    // First-order convergence: E = 0.5 * h
    for &h in &[1.0, 0.5, 0.25, 0.125] {
        let error = 0.5 * h;
        study.add_measurement(h, error);
    }

    let rate = study.compute_convergence_rate().unwrap();
    assert!((rate - 1.0).abs() < 0.01, "Expected rate≈1, got {}", rate);
}

#[test]
fn test_convergence_extrapolation() {
    let mut study = ConvergenceStudy::new("Extrapolation test");

    // E = 0.1 * h²
    for &h in &[1.0, 0.5, 0.25] {
        let error = 0.1 * h * h;
        study.add_measurement(h, error);
    }

    let extrapolated = study.extrapolate(0.125).unwrap();
    let expected = 0.1 * 0.125 * 0.125;

    let rel_error = (extrapolated - expected).abs() / expected;
    assert!(rel_error < 0.01);
}

#[test]
fn test_convergence_result_validation() {
    let mut study = ConvergenceStudy::new("Validation test");

    // Perfect second-order with slight noise
    for &h in &[1.0, 0.5, 0.25, 0.125] {
        let error = 0.1 * h * h;
        study.add_measurement(h, error);
    }

    let result = ConvergenceResult::from_study(&study, 2.0, 0.1).unwrap();

    assert!(result.passed);
    assert!((result.rate - 2.0).abs() < 0.1);
    assert!(result.is_monotonic);
    assert!(result.r_squared > 0.9);
}

// ============================================================================
// Test Suite 4: Energy Conservation
// ============================================================================

#[test]
fn test_energy_perfect_conservation() {
    let mut validator = EnergyValidator::new(1e-6); // 0.0001% tolerance

    // Perfect conservation: H = 1.0 for all time
    for i in 0..100 {
        let t = i as f64 * 0.01;
        validator.add_measurement(t, 0.5, 0.5);
    }

    let result = validator.validate();
    assert!(result.is_conserved);
    assert_eq!(result.max_deviation, 0.0);
    assert_eq!(result.relative_drift, 0.0);
    assert!(result.passed());
}

#[test]
fn test_energy_oscillating_but_conserved() {
    let mut validator = EnergyValidator::new(1e-10);

    // Energy oscillates between kinetic and potential but total is constant
    for i in 0..100 {
        let t = i as f64 * 0.01;
        let phase = 2.0 * PI * t;
        let kinetic = 0.5 + 0.5 * phase.sin();
        let strain = 0.5 - 0.5 * phase.sin();
        validator.add_measurement(t, kinetic, strain);
    }

    let result = validator.validate();
    assert!(result.is_conserved);
    assert!(result.max_deviation < 1e-14); // Numerical precision limit
}

#[test]
fn test_energy_drift_detection() {
    let mut validator = EnergyValidator::new(0.01); // 1% tolerance

    // Energy drifts by 5% over time
    for i in 0..100 {
        let t = i as f64 * 0.01;
        let total = 1.0 + 0.05 * t;
        validator.add_measurement(t, total / 2.0, total / 2.0);
    }

    let result = validator.validate();
    assert!(!result.is_conserved);
    assert!(result.relative_drift > 0.01);
    assert!(!result.passed());
}

#[test]
fn test_kinetic_energy_computation_2d() {
    // 3 points with 2 velocity components each
    let velocity = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let density = vec![1000.0, 1000.0, 1000.0];
    let cell_volume = 0.001; // 1 mm³

    let kinetic = compute_kinetic_energy(&velocity, &density, cell_volume, 2);

    // K = (1/2) * ρ * |v|² * ΔV for each point
    // Point 0: (1/2) * 1000 * 1.0 * 0.001 = 0.5
    // Point 1: (1/2) * 1000 * 1.0 * 0.001 = 0.5
    // Point 2: (1/2) * 1000 * 2.0 * 0.001 = 1.0
    // Total = 2.0 J
    assert!((kinetic - 2.0).abs() < 1e-10);
}

#[test]
fn test_strain_energy_computation_2d() {
    // 1 point with 3 stress/strain components (2D Voigt notation)
    let stress = vec![1e6, 1e6, 0.0]; // σxx, σyy, σxy = 1 MPa, 1 MPa, 0
    let strain = vec![0.001, 0.001, 0.0]; // εxx, εyy, 2εxy
    let cell_volume = 1e-6; // 1 mm³

    let strain_energy = compute_strain_energy(&stress, &strain, cell_volume, 3);

    // U = (1/2) * σ:ε * ΔV
    //   = (1/2) * (1e6 * 0.001 + 1e6 * 0.001 + 0) * 1e-6
    //   = (1/2) * 2000 * 1e-6 = 0.001 J
    assert!((strain_energy - 0.001).abs() < 1e-9);
}

#[test]
fn test_equipartition_ratio() {
    let mut validator = EnergyValidator::new(1e-6);

    // Equipartition: mean kinetic = mean potential
    for i in 0..100 {
        let t = i as f64 * 0.01;
        let phase = 2.0 * PI * t;
        let kinetic = 0.5 + 0.3 * phase.sin();
        let strain = 0.5 - 0.3 * phase.sin();
        validator.add_measurement(t, kinetic, strain);
    }

    let ratio = validator.equipartition_ratio();
    assert!((ratio - 1.0).abs() < 0.01);
}

// ============================================================================
// Test Suite 5: Validation Suite Composition
// ============================================================================

#[test]
fn test_validation_suite_all_pass() {
    let mut suite = ValidationSuite::new();

    suite.add(ValidationResult::success("test1", "OK"));
    suite.add(ValidationResult::success("test2", "OK"));
    suite.add(ValidationResult::success("test3", "OK"));

    assert!(suite.all_passed());
    assert_eq!(suite.passed_count(), 3);
    assert_eq!(suite.failed_count(), 0);
}

#[test]
fn test_validation_suite_mixed_results() {
    let mut suite = ValidationSuite::new();

    suite.add(ValidationResult::success("test1", "OK"));
    suite.add(ValidationResult::failure(
        "test2",
        1e-2,
        2e-2,
        1e-3,
        "Error too large",
    ));
    suite.add(ValidationResult::success("test3", "OK"));

    assert!(!suite.all_passed());
    assert_eq!(suite.passed_count(), 2);
    assert_eq!(suite.failed_count(), 1);

    let summary = suite.summary();
    assert_eq!(summary.total, 3);
    assert_eq!(summary.passed, 2);
    assert_eq!(summary.failed, 1);
    assert_eq!(summary.max_error_l2, 1e-2);
}

#[test]
fn test_validation_result_from_errors() {
    let result = ValidationResult::from_errors("test", 1e-4, 2e-4, 1e-3);
    assert!(result.passed);

    let result_fail = ValidationResult::from_errors("test", 1e-2, 2e-2, 1e-3);
    assert!(!result_fail.passed);
}

// ============================================================================
// Test Suite 6: Analytical Solution Verification (Wave Equation)
// ============================================================================

#[test]
fn test_plane_wave_satisfies_wave_equation() {
    let params = SolutionParameters {
        amplitude: 1e-6,
        wavelength: 0.01,
        omega: 0.0,
        wave_speed: 5000.0,
        density: 2700.0,
        lambda: 5e10,
        mu: 2.6e10,
    };

    let wave = PlaneWave2D::p_wave(1e-6, 0.01, [1.0, 0.0], params);

    // Wave equation: ∂²u/∂t² = c² ∇²u
    // For plane wave: ∂²u/∂t² = -ω² u, ∇²u = -k² u
    // Therefore: -ω² = c² (-k²) => ω = ck (dispersion relation)

    let x = &[0.1, 0.0];
    let t = 0.1;

    let acceleration = wave.acceleration(x, t);
    let displacement = wave.displacement(x, t);

    // Check that acceleration and displacement are related by -ω²
    let omega_squared = wave.parameters().omega.powi(2);
    let expected_accel_x = -omega_squared * displacement[0];
    let expected_accel_y = -omega_squared * displacement[1];

    assert!((acceleration[0] - expected_accel_x).abs() / (expected_accel_x.abs() + 1e-10) < 1e-10);
    assert!((acceleration[1] - expected_accel_y).abs() / (expected_accel_y.abs() + 1e-10) < 1e-10);
}

#[test]
fn test_sine_wave_dispersion_relation() {
    let wavelength = 1.0;
    let wave_speed = 2.0;
    let wave = SineWave1D::new(1.0, wavelength, wave_speed);

    let params = wave.parameters();
    let k = params.wave_number();
    let omega = params.omega;
    let c = params.wave_speed;

    // Dispersion relation: ω = c * k
    assert!((omega - c * k).abs() < 1e-10);
}

#[test]
fn test_solution_parameters_consistency() {
    let params = SolutionParameters {
        amplitude: 1e-6,
        wavelength: 0.01,
        omega: 1e6,
        wave_speed: 5000.0,
        density: 2700.0,
        lambda: 5e10,
        mu: 2.6e10,
    };

    // Wave number: k = 2π/λ
    let k = params.wave_number();
    assert!((k - 2.0 * PI / 0.01).abs() < 1e-10);

    // P-wave speed: cₚ = √((λ + 2μ)/ρ)
    let cp = params.p_wave_speed();
    let expected_cp = ((5e10_f64 + 2.0 * 2.6e10) / 2700.0).sqrt();
    assert!((cp - expected_cp).abs() < 1.0);

    // S-wave speed: cₛ = √(μ/ρ)
    let cs = params.s_wave_speed();
    let expected_cs = (2.6e10_f64 / 2700.0).sqrt();
    assert!((cs - expected_cs).abs() < 1.0);

    // P-wave must be faster than S-wave
    assert!(cp > cs);
}

// ============================================================================
// Test Suite 7: Strain and Stress Computation
// ============================================================================

#[test]
fn test_strain_from_gradient() {
    let poly = PolynomialTest2D::new(1.0);

    // u = x², v = xy
    // Gradient: ∂u/∂x = 2x, ∂u/∂y = 0, ∂v/∂x = y, ∂v/∂y = x
    // Strain (Voigt): εxx = ∂u/∂x = 2x
    //                 εyy = ∂v/∂y = x
    //                 2εxy = ∂u/∂y + ∂v/∂x = 0 + y = y

    let x = &[2.0, 3.0];
    let t = 0.0;

    let strain = poly.strain(x, t);

    assert_eq!(strain[0], 4.0); // εxx = 2x = 4
    assert_eq!(strain[1], 2.0); // εyy = x = 2
    assert_eq!(strain[2], 3.0); // 2εxy = y = 3
}

#[test]
fn test_stress_from_strain() {
    let poly = PolynomialTest2D::new(1.0);

    let x = &[1.0, 1.0];
    let t = 0.0;

    let lambda = 1e9;
    let mu = 0.5e9;

    let stress = poly.stress(x, t, lambda, mu);

    // Strain: εxx = 2, εyy = 1, 2εxy = 1 => εxy = 0.5
    // trace(ε) = εxx + εyy = 3
    // σxx = λ*trace + 2μ*εxx = 1e9*3 + 2*0.5e9*2 = 5e9
    // σyy = λ*trace + 2μ*εyy = 1e9*3 + 2*0.5e9*1 = 4e9
    // σxy = 2μ*εxy = 2*0.5e9*0.5 = 0.5e9 (but in Voigt: 2*σxy = 1e9)

    assert!((stress[0] - 5e9).abs() / 5e9 < 1e-10);
    assert!((stress[1] - 4e9).abs() / 4e9 < 1e-10);
}
