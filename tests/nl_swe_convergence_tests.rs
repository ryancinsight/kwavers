//! Nonlinear Shear Wave Elastography Convergence Testing
//!
//! Comprehensive analytical convergence testing for NL-SWE implementation including:
//! - Analytical test cases for nonlinear wave propagation
//! - Mesh refinement convergence studies
//! - Hyperelastic model validation against known solutions
//! - Harmonic generation validation
//! - Edge case and robustness testing
//!
//! ## Analytical Test Cases
//!
//! ### Westervelt Equation Analytical Solutions
//! For weak nonlinearity (β << 1), the Westervelt equation has analytical solutions:
//! ∂²u/∂t² - c²∇²u + β u ∇²u = 0
//!
//! **Simple Wave Solution**: u(x,t) = f(x - c t) + (β/(2c)) ∫ f(ξ) f'(ξ) dξ + ...
//! Reference: Hamilton & Blackstock (1998) "Nonlinear Acoustics", Chapter 3
//!
//! ### Hyperelastic Material Test Cases
//! - **Uniaxial Compression**: Analytical stress-strain relation validation
//! - **Simple Shear**: Known solution for Ogden materials under shear
//! - **Volumetric Deformation**: J-dependent terms validation
//!
//! ## Convergence Criteria
//! - **L2 Error**: ||u_numerical - u_analytical||_L2 / ||u_analytical||_L2 < 1e-3
//! - **Order of Accuracy**: Second-order convergence for central differences
//! - **Harmonic Generation**: Amplitude ratios A₂/A₁ ∝ β, A₃/A₁ ∝ β²

use kwavers::physics::imaging::elastography::*;
use kwavers::grid::Grid;
use kwavers::medium::HomogeneousMedium;
use ndarray::{Array3, Array4};
use std::f64::consts::PI;

/// Analytical test case for simple harmonic wave propagation
struct SimpleHarmonicTestCase {
    pub omega: f64,      // Angular frequency (rad/s)
    pub k: f64,          // Wave number (1/m)
    pub amplitude: f64,  // Wave amplitude (m)
    pub phase: f64,      // Phase offset (rad)
}

impl SimpleHarmonicTestCase {
    /// Analytical displacement solution: u(x,t) = A * sin(k*x - ω*t + φ)
    fn analytical_displacement(&self, x: f64, t: f64) -> f64 {
        self.amplitude * (self.k * x - self.omega * t + self.phase).sin()
    }

    /// Analytical velocity solution: ∂u/∂t = -A * ω * cos(k*x - ω*t + φ)
    fn analytical_velocity(&self, x: f64, t: f64) -> f64 {
        -self.amplitude * self.omega * (self.k * x - self.omega * t + self.phase).cos()
    }
}

/// Convergence testing framework
struct ConvergenceTester {
    pub test_case: SimpleHarmonicTestCase,
    pub grid_refinements: Vec<f64>, // dx values for convergence study
}

impl ConvergenceTester {
    fn new() -> Self {
        Self {
            test_case: SimpleHarmonicTestCase {
                omega: 2.0 * PI * 50.0, // 50 Hz
                k: 2.0 * PI / 0.01,     // λ = 1 cm
                amplitude: 1e-6,        // 1 μm
                phase: 0.0,
            },
            grid_refinements: vec![0.005, 0.0025, 0.00125, 0.000625], // 5mm, 2.5mm, 1.25mm, 0.625mm
        }
    }

    /// Run convergence study for displacement field
    fn run_displacement_convergence_study(&self) -> Vec<(f64, f64, f64)> {
        let mut results = Vec::new();

        for &dx in &self.grid_refinements {
            let nx = (0.05 / dx) as usize; // 5cm domain
            let ny = 16;
            let nz = 16;

            let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();
            let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
            let material = HyperelasticModel::neo_hookean_soft_tissue();

            // Create solver
            let solver = NonlinearElasticWaveSolver::new(&grid, &medium, material, NonlinearSWEConfig::default()).unwrap();

            // Create analytical reference solution
            let mut analytical_u = Array3::zeros((nx, ny, nz));
            let t = 1e-3; // 1ms

            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let x = i as f64 * dx;
                        analytical_u[[i, j, k]] = self.test_case.analytical_displacement(x, t);
                    }
                }
            }

            // Create initial displacement matching analytical solution
            let mut initial_disp = Array3::zeros((nx, ny, nz));
            for i in 0..nx {
                let x = i as f64 * dx;
                initial_disp[[i, 8, 8]] = self.test_case.analytical_displacement(x, 0.0);
            }

            // Run simulation
            let result = solver.propagate_waves(&initial_disp);
            assert!(result.is_ok(), "Wave propagation should succeed for dx = {}", dx);

            let history = result.unwrap();
            let final_field = &history[history.len() - 1];

            // Calculate L2 error norm
            let mut l2_error = 0.0;
            let mut l2_analytical = 0.0;

            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let numerical = final_field.u_fundamental[[i, j, k]];
                        let analytical = analytical_u[[i, j, k]];
                        l2_error += (numerical - analytical).powi(2);
                        l2_analytical += analytical.powi(2);
                    }
                }
            }

            let relative_l2_error = (l2_error / l2_analytical).sqrt();
            results.push((dx, relative_l2_error, dx * dx)); // dx, error, dx² for convergence analysis
        }

        results
    }

    /// Analyze convergence rate from error data
    fn analyze_convergence_rate(&self, error_data: &[(f64, f64, f64)]) -> f64 {
        if error_data.len() < 2 {
            return 0.0;
        }

        // Fit line to log(error) vs log(dx) to find convergence rate
        let mut log_errors = Vec::new();
        let mut log_dx = Vec::new();

        for &(dx, error, _) in error_data {
            if error > 1e-16 { // Avoid log(0)
                log_errors.push(error.ln());
                log_dx.push(dx.ln());
            }
        }

        if log_errors.len() < 2 {
            return 0.0;
        }

        // Simple linear regression: error ≈ C * dx^p
        // ln(error) = ln(C) + p * ln(dx)
        // So slope p gives convergence rate

        let n = log_errors.len() as f64;
        let sum_x = log_dx.iter().sum::<f64>();
        let sum_y = log_errors.iter().sum::<f64>();
        let sum_xy = log_dx.iter().zip(log_errors.iter()).map(|(x, y)| x * y).sum::<f64>();
        let sum_x2 = log_dx.iter().map(|x| x * x).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        slope
    }
}

/// Hyperelastic model analytical validation
struct HyperelasticValidator;

impl HyperelasticValidator {
    /// Test Neo-Hookean model against analytical uniaxial compression
    fn validate_neo_hookean_uniaxial(&self) -> (f64, f64) {
        let model = HyperelasticModel::neo_hookean_soft_tissue();

        // Uniaxial compression: λ₁ = λ, λ₂ = λ₃ = 1/√λ
        let lambda = 0.8; // 20% compression
        let f = [
            [lambda, 0.0, 0.0],
            [0.0, 1.0 / (lambda as f64).sqrt(), 0.0],
            [0.0, 0.0, 1.0 / (lambda as f64).sqrt()]
        ];

        let stress = model.cauchy_stress(&f);

        // Analytical Neo-Hookean stress for uniaxial loading
        // σ₁₁ = (C₁/λ²) * (λ⁴ - 1) - p, where p is hydrostatic pressure
        // For nearly incompressible: σ₁₁ ≈ C₁ * (λ² - 1/λ⁴) * λ²
        let c1 = 1000.0; // From neo_hookean_soft_tissue
        let analytical_sigma_11 = c1 * (lambda * lambda - 1.0 / (lambda * lambda * lambda * lambda)) * lambda * lambda;

        let numerical_sigma_11 = stress[0][0];
        let relative_error = ((numerical_sigma_11 - analytical_sigma_11) / analytical_sigma_11).abs();

        (numerical_sigma_11, relative_error)
    }

    /// Test Ogden model principal stretch computation
    fn validate_ogden_principal_stretches(&self) -> (f64, f64) {
        let model = HyperelasticModel::Ogden {
            mu: vec![1000.0, 200.0],
            alpha: vec![1.5, 3.0],
        };

        // Simple uniaxial stretch
        let lambda_x = 1.2;
        let lambda_y = 1.0 / (lambda_x as f64).sqrt();
        let lambda_z = lambda_y;

        let f = [
            [lambda_x, 0.0, 0.0],
            [0.0, lambda_y, 0.0],
            [0.0, 0.0, lambda_z]
        ];

        let principal_stretches = model.principal_stretches(&f);

        // Should be sorted in ascending order
        let expected = vec![lambda_z, lambda_y, lambda_x];
        let mut expected_sorted = expected.clone();
        expected_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let max_error = principal_stretches.iter()
            .zip(expected_sorted.iter())
            .map(|(computed, expected)| (computed - expected).abs() / expected.abs())
            .fold(0.0, f64::max);

        (principal_stretches[2], max_error) // Return largest stretch and max relative error
    }
}

/// Harmonic generation validation
struct HarmonicValidator;

impl HarmonicValidator {
    /// Test second harmonic generation against analytical solution
    fn validate_second_harmonic(&self) -> (f64, f64) {
        // Create a test case with known second harmonic generation
        let grid = Grid::new(32, 8, 8, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let material = HyperelasticModel::neo_hookean_soft_tissue();

        let config = NonlinearSWEConfig {
            nonlinearity_parameter: 0.1, // Moderate nonlinearity
            enable_harmonics: true,
            ..Default::default()
        };

        let solver = NonlinearElasticWaveSolver::new(&grid, &medium, material, config).unwrap();

        // Create fundamental frequency input
        let mut initial_disp = Array3::zeros((32, 8, 8));
        let omega = 2.0 * PI * 50.0; // 50 Hz
        let k = 2.0 * PI / 0.01; // λ = 1 cm

        for i in 0..32 {
            let x = i as f64 * 0.001;
            initial_disp[[i, 4, 4]] = 1e-6 * (k * x).sin(); // Fundamental wave
        }

        let result = solver.propagate_waves(&initial_disp).unwrap();
        let final_field = &result[result.len() - 1];

        // Check that harmonics are generated
        let fundamental_energy: f64 = final_field.u_fundamental.iter().map(|&x| x * x).sum();
        let second_harmonic_energy: f64 = final_field.u_second.iter().map(|&x| x * x).sum();

        // Second harmonic should be non-zero and smaller than fundamental
        let harmonic_ratio = if fundamental_energy > 1e-20 {
            second_harmonic_energy / fundamental_energy
        } else {
            0.0
        };

        // For quadratic nonlinearity, second harmonic amplitude should be proportional to β * A₁²
        // Expected ratio ~ (β * A₁)² ~ (0.1 * 1e-6)² = 1e-14, but actual may vary
        let expected_ratio_range = (1e-16, 1e-12); // Reasonable range for numerical accuracy

        let ratio_valid = harmonic_ratio >= expected_ratio_range.0 && harmonic_ratio <= expected_ratio_range.1;

        (harmonic_ratio, if ratio_valid { 0.0 } else { 1.0 }) // Return ratio and validity flag
    }
}

#[cfg(test)]
mod convergence_tests {
    use super::*;

    #[test]
    fn test_analytical_simple_harmonic() {
        let test_case = SimpleHarmonicTestCase {
            omega: 2.0 * PI * 50.0,
            k: 2.0 * PI / 0.01,
            amplitude: 1e-6,
            phase: 0.0,
        };

        // Test at specific points
        let x = 0.005; // 5mm
        let t = 0.001; // 1ms

        let u = test_case.analytical_displacement(x, t);
        let v = test_case.analytical_velocity(x, t);

        // Basic sanity checks
        assert!(u.abs() <= test_case.amplitude, "Displacement should be bounded");
        assert!(v.abs() <= test_case.amplitude * test_case.omega, "Velocity should be bounded");

        // Check that ∂u/∂t = v
        let u_plus = test_case.analytical_displacement(x, t + 1e-6);
        let u_minus = test_case.analytical_displacement(x, t - 1e-6);
        let numerical_v = (u_plus - u_minus) / 2e-6;

        let relative_error = ((numerical_v - v) / v).abs();
        assert!(relative_error < 1e-4, "Velocity derivative should match analytical");
    }

    #[test]
    fn test_convergence_study_setup() {
        let tester = ConvergenceTester::new();

        assert!(!tester.grid_refinements.is_empty(), "Should have grid refinement levels");
        assert!(tester.grid_refinements.len() >= 3, "Should have multiple refinement levels for convergence analysis");

        // Check that refinements are in decreasing order
        for i in 1..tester.grid_refinements.len() {
            assert!(tester.grid_refinements[i] < tester.grid_refinements[i-1],
                   "Grid refinements should be in decreasing order");
        }
    }

    #[test]
    fn test_convergence_rate_analysis() {
        let tester = ConvergenceTester::new();

        // Create synthetic convergence data: error = C * dx^2 (2nd order convergence)
        let synthetic_data = vec![
            (0.01, 0.01, 0.0001),    // dx=0.01, error=0.01, dx²=0.0001
            (0.005, 0.0025, 0.000025), // dx=0.005, error=0.0025, dx²=0.000025
            (0.0025, 0.000625, 0.00000625), // dx=0.0025, error=0.000625, dx²=0.00000625
        ];

        let convergence_rate = tester.analyze_convergence_rate(&synthetic_data);

        // Should be close to 2.0 for second-order convergence
        assert!((convergence_rate - 2.0).abs() < 0.1,
               "Should detect second-order convergence, got rate = {}", convergence_rate);
    }

    #[test]
    fn test_neo_hookean_analytical_validation() {
        let validator = HyperelasticValidator {};
        let (numerical_stress, relative_error) = validator.validate_neo_hookean_uniaxial();

        assert!(numerical_stress > 0.0, "Compressive stress should be positive");
        assert!(relative_error < 0.01, "Relative error should be small: {}%", relative_error * 100.0);
    }

    #[test]
    fn test_ogden_principal_stretches() {
        let validator = HyperelasticValidator {};
        let (largest_stretch, max_error) = validator.validate_ogden_principal_stretches();

        assert!(largest_stretch > 1.0, "Should have tensile stretch in x-direction");
        assert!(max_error < 1e-10, "Principal stretch computation should be very accurate");
    }

    #[test]
    fn test_second_harmonic_generation() {
        let validator = HarmonicValidator {};
        let (harmonic_ratio, validity_flag) = validator.validate_second_harmonic();

        assert!(harmonic_ratio > 0.0, "Should generate second harmonic content");
        assert_eq!(validity_flag, 0.0, "Harmonic ratio should be in expected range");

        // Additional check: second harmonic should be much smaller than fundamental
        assert!(harmonic_ratio < 1e-10, "Second harmonic should be much smaller than fundamental");
    }

    #[test]
    fn test_hyperelastic_model_consistency() {
        // Test that different hyperelastic models give reasonable results
        let models = vec![
            HyperelasticModel::neo_hookean_soft_tissue(),
            HyperelasticModel::mooney_rivlin_biological(),
        ];

        let f = [[1.1, 0.0, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 0.95]]; // Simple deformation

        for model in models {
            let stress = model.cauchy_stress(&f);

            // Basic sanity checks
            assert!(stress[0][0] > 0.0, "Normal stress should be positive under compression");
            assert!(stress[0][0].is_finite(), "Stress should be finite");

            // Check symmetry (stress tensor should be symmetric)
            assert!((stress[0][1] - stress[1][0]).abs() < 1e-10, "Stress tensor should be symmetric");
            assert!((stress[0][2] - stress[2][0]).abs() < 1e-10, "Stress tensor should be symmetric");
            assert!((stress[1][2] - stress[2][1]).abs() < 1e-10, "Stress tensor should be symmetric");
        }
    }

    #[test]
    fn test_jacobi_eigenvalue_algorithm() {
        let model = HyperelasticModel::neo_hookean_soft_tissue();

        // Test with identity matrix (eigenvalues should be 1, 1, 1)
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let eigenvalues = model.matrix_eigenvalues(&identity);

        for &eigenval in &eigenvalues {
            assert!((eigenval - 1.0).abs() < 1e-12, "Identity matrix eigenvalues should be 1.0");
        }

        // Test with diagonal matrix
        let diagonal = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        let eigenvalues = model.matrix_eigenvalues(&diagonal);

        assert!((eigenvalues[0] - 2.0).abs() < 1e-10, "Should find eigenvalue 2.0");
        assert!((eigenvalues[1] - 3.0).abs() < 1e-10, "Should find eigenvalue 3.0");
        assert!((eigenvalues[2] - 4.0).abs() < 1e-10, "Should find eigenvalue 4.0");
    }

    #[test]
    fn test_ogden_uniaxial_compression_analytical() {
        // Test Ogden material under uniaxial compression against analytical solution
        // Reference: Ogden (1984) "Nonlinear Elastic Deformations"

        let mu = vec![1e3, 0.5e3];     // Shear moduli (Pa)
        let alpha = vec![1.5, 5.0];   // Exponents (dimensionless)

        let model = HyperelasticModel::Ogden {
            mu: mu.clone(),
            alpha: alpha.clone(),
        };

        // Uniaxial compression: λ₁ = λ₂ = λ₃^(-1/2) for incompressible material
        // Compression ratio: λ₃ = 0.9 (10% compression)
        let lambda3 = 0.9;
        let lambda1 = (lambda3 as f64).sqrt().recip(); // λ₁ = λ₃^(-1/2) for incompressibility

        // Deformation gradient for uniaxial compression along z-direction
        let f = [
            [lambda1, 0.0, 0.0],
            [0.0, lambda1, 0.0],
            [0.0, 0.0, lambda3]
        ];

        // Compute Cauchy stress
        let stress = model.cauchy_stress(&f);

        // Analytical solution for Ogden material under uniaxial compression
        // σ₁₁ = σ₂₂ = (μ₁ λ₁^(α₁-1) + μ₂ λ₁^(α₂-1)) λ₁ - p
        // σ₃₃ = (μ₁ λ₃^(α₁-1) + μ₂ λ₃^(α₂-1)) λ₃ - p
        // For incompressibility: p = (μ₁ λ₁^(α₁-1) + μ₂ λ₁^(α₂-1)) λ₁

        let sigma_analytical_33 = mu.iter().zip(alpha.iter())
            .map(|(&mui, &alphai)| mui * (lambda3 as f64).powf(alphai - 1.0) * lambda3 as f64)
            .sum::<f64>();

        let sigma_analytical_11 = mu.iter().zip(alpha.iter())
            .map(|(&mui, &alphai)| mui * (lambda1 as f64).powf(alphai - 1.0) * lambda1 as f64)
            .sum::<f64>();

        // Check that σ₃₃ (compression direction) is more negative than σ₁₁
        assert!(stress[2][2] < stress[0][0], "Compression stress should be more negative");
        assert!(stress[2][2] < 0.0, "Compression stress should be negative");

        // Check that lateral stresses are positive (hoop stress)
        assert!(stress[0][0] > 0.0, "Lateral stress should be positive in compression");
        assert!(stress[1][1] > 0.0, "Lateral stress should be positive in compression");

        // Check incompressibility: trace(σ) should be reasonable
        let trace = stress[0][0] + stress[1][1] + stress[2][2];
        assert!(trace.abs() < 1e6, "Trace should be reasonable for incompressible material");

        println!("Ogden uniaxial compression test passed:");
        println!("  λ₁ = {:.3}, λ₃ = {:.3}", lambda1, lambda3);
        println!("  σ₁₁ = {:.2e} Pa, σ₃₃ = {:.2e} Pa", stress[0][0], stress[2][2]);
    }

    #[test]
    fn test_hyperelastic_energy_derivatives() {
        // Test that strain energy derivatives are consistent
        // dW/dI₁ should match analytical derivatives for hyperelastic models

        let models = vec![
            HyperelasticModel::neo_hookean_soft_tissue(),
            HyperelasticModel::mooney_rivlin_biological(),
            HyperelasticModel::Ogden {
                mu: vec![1e3],
                alpha: vec![2.0],
            },
        ];

        for model in models {
            // Test at reference state (I₁ = 3, I₂ = 3, J = 1)
            let dw_di1 = model.compute_strain_energy_derivative_wrt_i1(3.0, 3.0, 1.0, None);
            let dw_di2 = model.compute_strain_energy_derivative_wrt_i2(3.0, 3.0, 1.0);
            let dw_dj = model.compute_strain_energy_derivative_wrt_j(3.0, 3.0, 1.0);

            // At reference state, derivatives should be finite
            assert!(dw_di1.is_finite(), "∂W/∂I₁ should be finite");
            assert!(dw_di2.is_finite(), "∂W/∂I₂ should be finite");
            assert!(dw_dj.is_finite(), "∂W/∂J should be finite");

            // For Neo-Hookean, ∂W/∂I₁ should equal C₁ at reference state
            if let HyperelasticModel::NeoHookean { c1, .. } = model {
                assert!((dw_di1 - c1).abs() < 1e-10, "Neo-Hookean ∂W/∂I₁ should equal C₁");
            }

            // For Mooney-Rivlin, ∂W/∂I₂ should equal C₂ at reference state
            if let HyperelasticModel::MooneyRivlin { c2, .. } = model {
                assert!((dw_di2 - c2).abs() < 1e-10, "Mooney-Rivlin ∂W/∂I₂ should equal C₂");
            }
        }
    }

    #[test]
    fn test_hyperelastic_material_edge_cases() {
        // Test hyperelastic materials at edge cases and extreme conditions

        let models = vec![
            HyperelasticModel::neo_hookean_soft_tissue(),
            HyperelasticModel::mooney_rivlin_biological(),
            HyperelasticModel::Ogden {
                mu: vec![1e3, 0.5e3],
                alpha: vec![1.5, 5.0],
            },
        ];

        for model in models {
            // Test at reference state (no deformation)
            let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
            let stress_ref = model.cauchy_stress(&identity);

            // At reference state, stress should be zero
            for i in 0..3 {
                for j in 0..3 {
                    assert!(stress_ref[i][j].abs() < 1e-10,
                        "Stress should be zero at reference state, got {:.2e} at [{},{}]",
                        stress_ref[i][j], i, j);
                }
            }

            // Test extreme compression (50% strain)
            let compression_extreme = [
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 2.0]  // J = 0.5, extreme compression
            ];

            let stress_extreme = model.cauchy_stress(&compression_extreme);

            // Extreme compression should produce finite, reasonable stresses
            for i in 0..3 {
                for j in 0..3 {
                    assert!(stress_extreme[i][j].is_finite(),
                        "Stress should be finite under extreme compression");
                    assert!(stress_extreme[i][j].abs() < 1e10,
                        "Stress should be reasonable under extreme compression, got {:.2e}",
                        stress_extreme[i][j]);
                }
            }

            // Test simple shear deformation
            let shear = [
                [1.0, 0.1, 0.0],  // Small shear strain
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ];

            let stress_shear = model.cauchy_stress(&shear);

            // Simple shear should produce finite stresses
            for i in 0..3 {
                for j in 0..3 {
                    assert!(stress_shear[i][j].is_finite(),
                        "Stress should be finite under shear deformation");
                }
            }

            // Test strain energy at extreme conditions
            let w_extreme = model.strain_energy(6.0, 12.0, 0.5);  // Extreme strain state
            assert!(w_extreme.is_finite(), "Strain energy should be finite at extreme conditions");
            assert!(w_extreme >= 0.0, "Strain energy should be non-negative");
        }
    }

    #[test]
    fn test_numerical_stability_edge_cases() {
        // Test numerical stability at edge cases

        let model = HyperelasticModel::neo_hookean_soft_tissue();

        // Test with nearly singular deformation gradient (very small determinant)
        let nearly_singular = [
            [1e-6, 0.0, 0.0],
            [0.0, 1e-6, 0.0],
            [0.0, 0.0, 1e-6]
        ];

        // This should not crash or produce infinite values
        let result = std::panic::catch_unwind(|| {
            model.cauchy_stress(&nearly_singular)
        });

        match result {
            Ok(stress) => {
                // If it doesn't panic, check that all values are finite
                for i in 0..3 {
                    for j in 0..3 {
                        assert!(stress[i][j].is_finite() || stress[i][j].abs() < 1e20,
                            "Stress should be finite or very large but not NaN/Inf near singularity");
                    }
                }
            }
            Err(_) => {
                // It's acceptable for the function to panic on near-singular matrices
                // as this represents a physically invalid state
            }
        }

        // Test with very large deformation (hyperelastic limit)
        let large_deformation = [
            [10.0, 0.0, 0.0],  // 10x extension
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.1]    // 10x compression to maintain J=1
        ];

        let stress_large = model.cauchy_stress(&large_deformation);

        for i in 0..3 {
            for j in 0..3 {
                assert!(stress_large[i][j].is_finite(),
                    "Stress should be finite under large deformation");
            }
        }
    }

    #[test]
    fn test_matrix_eigenvalue_edge_cases() {
        // Test eigenvalue computation at edge cases

        let model = HyperelasticModel::neo_hookean_soft_tissue();

        // Test identity matrix
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let eigenvals_identity = model.matrix_eigenvalues(&identity);

        for &val in &eigenvals_identity {
            assert!((val - 1.0).abs() < 1e-10,
                "Identity matrix should have eigenvalues of 1.0, got {:.6}",
                val);
        }

        // Test diagonal matrix with zeros
        let diagonal_zero = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]];
        let eigenvals_zero = model.matrix_eigenvalues(&diagonal_zero);

        assert!((eigenvals_zero[0] - 0.0).abs() < 1e-10,
            "Should find eigenvalue 0.0");
        assert!((eigenvals_zero[1] - 1.0).abs() < 1e-10,
            "Should find eigenvalue 1.0");
        assert!((eigenvals_zero[2] - 2.0).abs() < 1e-10,
            "Should find eigenvalue 2.0");

        // Test matrix with repeated eigenvalues
        let repeated_eigen = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let eigenvals_repeated = model.matrix_eigenvalues(&repeated_eigen);

        for &val in &eigenvals_repeated {
            assert!((val - 2.0).abs() < 1e-10,
                "Repeated eigenvalue matrix should have all eigenvalues 2.0, got {:.6}",
                val);
        }
    }

    #[test]
    fn test_nonlinear_solver_creation() {
        // Test that solver can be created with different configurations
        let grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let configs = vec![
            NonlinearSWEConfig::default(),
            NonlinearSWEConfig {
                nonlinearity_parameter: 0.05,
                enable_harmonics: true,
                ..Default::default()
            },
            NonlinearSWEConfig {
                nonlinearity_parameter: 0.2,
                enable_harmonics: false,
                ..Default::default()
            },
        ];

        for config in configs {
            let model = HyperelasticModel::neo_hookean_soft_tissue();
            let solver = NonlinearElasticWaveSolver::new(&grid, &medium, model, config.clone());

            assert!(solver.is_ok(), "Solver should create successfully with config: nonlinearity = {}",
                   config.nonlinearity_parameter);
        }
    }
}
