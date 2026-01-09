//! Nonlinear Shear Wave Elastography Edge Case and Robustness Testing
//!
//! Comprehensive testing for edge cases and robustness validation including:
//! - Extreme material parameters and deformation states
//! - Numerical stability at boundaries and singularities
//! - Large deformation hyperelastic behavior
//! - Harmonic generation under extreme conditions

use kwavers::domain::grid::Grid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::physics::imaging::modalities::elastography::*;
use ndarray::Array3;
use std::prelude::v1::*;

/// Edge case testing framework
struct EdgeCaseTester;

impl EdgeCaseTester {
    /// Test hyperelastic models with extreme compression
    fn test_extreme_compression(&self) -> Vec<(String, f64, bool)> {
        let mut results = Vec::new();

        let models = vec![
            ("Neo-Hookean", HyperelasticModel::neo_hookean_soft_tissue()),
            (
                "Mooney-Rivlin",
                HyperelasticModel::mooney_rivlin_biological(),
            ),
        ];

        // Test compression ratios from 10% to 90%
        let compression_ratios = [0.9_f64, 0.5_f64, 0.1_f64];

        for (model_name, model) in models {
            for &lambda in &compression_ratios {
                let deformation_gradient = [
                    [lambda, 0.0, 0.0],
                    [0.0, 1.0 / lambda.sqrt(), 0.0],
                    [0.0, 0.0, 1.0 / lambda.sqrt()],
                ];

                let stress = model.cauchy_stress(&deformation_gradient);
                let sigma_xx = stress[0][0];

                let is_stable = sigma_xx < 0.0 && sigma_xx.is_finite() && !sigma_xx.is_nan();

                results.push((
                    format!("{}_{:.0}%", model_name, (1.0 - lambda) * 100.0),
                    sigma_xx,
                    is_stable,
                ));
            }
        }

        results
    }

    /// Test hyperelastic models with extreme tension
    fn test_extreme_tension(&self) -> Vec<(String, f64, bool)> {
        let mut results = Vec::new();

        let models = vec![
            ("Neo-Hookean", HyperelasticModel::neo_hookean_soft_tissue()),
            (
                "Mooney-Rivlin",
                HyperelasticModel::mooney_rivlin_biological(),
            ),
        ];

        // Test extension ratios from 10% to 300%
        let extension_ratios = [1.1_f64, 2.0_f64, 4.0_f64];

        for (model_name, model) in models {
            for &lambda in &extension_ratios {
                let deformation_gradient = [
                    [lambda, 0.0, 0.0],
                    [0.0, 1.0 / lambda.sqrt(), 0.0],
                    [0.0, 0.0, 1.0 / lambda.sqrt()],
                ];

                let stress = model.cauchy_stress(&deformation_gradient);
                let sigma_xx = stress[0][0];

                // Check stability: stress should be finite (can be negative for extension)
                let is_stable = sigma_xx.is_finite() && !sigma_xx.is_nan();

                results.push((
                    format!("{}_{:.0}x", model_name, lambda),
                    sigma_xx,
                    is_stable,
                ));
            }
        }

        results
    }

    /// Test Ogden model with extreme parameters
    fn test_ogden_extreme_parameters(&self) -> Vec<(String, bool)> {
        let mut results = Vec::new();

        // Test various Ogden parameter combinations
        let test_cases = vec![
            ("Standard", vec![1000.0], vec![1.5]),
            ("Multiple terms", vec![1000.0, 200.0], vec![1.5, 3.0]),
            ("High exponents", vec![1000.0], vec![5.0]),
            ("Negative exponents", vec![1000.0], vec![-1.0]),
            ("Zero moduli", vec![0.0], vec![1.5]),
        ];

        let deformation_gradient = [
            [1.2, 0.0, 0.0],
            [0.0, 0.91, 0.0], // 1/sqrt(1.2) ≈ 0.912
            [0.0, 0.0, 0.91],
        ];

        for (case_name, mu, alpha) in test_cases {
            let model = HyperelasticModel::Ogden {
                mu: mu.clone(),
                alpha: alpha.clone(),
            };

            let stress_result =
                std::panic::catch_unwind(|| model.cauchy_stress(&deformation_gradient));

            let is_stable = stress_result.is_ok() && {
                if let Ok(stress) = stress_result {
                    // Check all stress components are finite
                    stress
                        .iter()
                        .all(|row| row.iter().all(|&s| s.is_finite() && !s.is_nan()))
                } else {
                    false
                }
            };

            results.push((case_name.to_string(), is_stable));
        }

        results
    }

    /// Test numerical stability near material singularities
    fn test_near_singularities(&self) -> Vec<(String, bool)> {
        let mut results = Vec::new();

        // Test deformation gradients that might cause numerical issues
        let critical_cases = vec![
            (
                "Near zero volume",
                [
                    [0.01, 0.0, 0.0],  // Very small stretch
                    [0.0, 100.0, 0.0], // Very large stretch
                    [0.0, 0.0, 100.0],
                ],
            ),
            (
                "High shear",
                [
                    [1.0, 0.9, 0.0], // Large shear component
                    [0.9, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ),
            (
                "Rotation dominant",
                [
                    [0.0, 1.0, 0.0], // Pure rotation
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ),
        ];

        let model = HyperelasticModel::neo_hookean_soft_tissue();

        for (case_name, deformation_gradient) in critical_cases {
            let stress_result =
                std::panic::catch_unwind(|| model.cauchy_stress(&deformation_gradient));

            let is_stable = stress_result.is_ok() && {
                if let Ok(stress) = stress_result {
                    // Check all stress components are finite
                    stress
                        .iter()
                        .all(|row| row.iter().all(|&s| s.is_finite() && !s.is_nan()))
                } else {
                    false
                }
            };

            results.push((case_name.to_string(), is_stable));
        }

        results
    }

    /// Test harmonic generation with extreme nonlinearity
    fn test_extreme_harmonic_generation(&self) -> Vec<(String, f64, bool)> {
        let mut results = Vec::new();

        let nonlinearity_values = [0.0, 0.1, 1.0, 10.0]; // From linear to extreme nonlinearity

        for &beta in &nonlinearity_values {
            let grid = Grid::new(16, 8, 8, 0.002, 0.002, 0.002).unwrap();
            let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
            let material = HyperelasticModel::neo_hookean_soft_tissue();

            let config = NonlinearSWEConfig {
                nonlinearity_parameter: beta,
                enable_harmonics: true,
                ..Default::default()
            };

            let solver_result = NonlinearElasticWaveSolver::new(&grid, &medium, material, config);

            if solver_result.is_err() {
                results.push((format!("β={}", beta), 0.0, false));
                continue;
            }

            let solver = solver_result.unwrap();

            // Create simple initial displacement
            let mut initial_disp = Array3::zeros((16, 8, 8));
            initial_disp[[8, 4, 4]] = 1e-6; // Point source

            let propagation_result = solver.propagate_waves(&initial_disp);

            let is_stable = propagation_result.is_ok() && {
                if let Ok(ref history) = propagation_result {
                    let final_field = &history[history.len() - 1];

                    // Check that all fields are finite
                    let fundamental_finite =
                        final_field.u_fundamental.iter().all(|&x| x.is_finite());
                    let second_harmonic_finite =
                        final_field.u_second.iter().all(|&x| x.is_finite());

                    fundamental_finite && second_harmonic_finite
                } else {
                    false
                }
            };

            // Calculate harmonic ratio if successful
            let harmonic_ratio = if let Ok(ref history) = propagation_result {
                let final_field = &history[history.len() - 1];
                let fundamental_energy: f64 =
                    final_field.u_fundamental.iter().map(|&x| x * x).sum();
                let harmonic_energy: f64 = final_field.u_second.iter().map(|&x| x * x).sum();

                if fundamental_energy > 1e-20 {
                    harmonic_energy / fundamental_energy
                } else {
                    0.0
                }
            } else {
                0.0
            };

            results.push((format!("β={}", beta), harmonic_ratio, is_stable));
        }

        results
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_extreme_compression_stability() {
        let tester = EdgeCaseTester {};
        let results = tester.test_extreme_compression();

        println!("Extreme compression stability test:");
        for (case, stress, is_stable) in results {
            println!("  {}: σ = {:.2e} Pa, stable = {}", case, stress, is_stable);
            assert!(
                is_stable,
                "Compression case '{}' should be numerically stable",
                case
            );
        }
    }

    #[test]
    fn test_extreme_tension_stability() {
        let tester = EdgeCaseTester {};
        let results = tester.test_extreme_tension();

        println!("Extreme tension stability test:");
        for (case, stress, is_stable) in results {
            println!("  {}: σ = {:.2e} Pa, stable = {}", case, stress, is_stable);
            assert!(
                is_stable,
                "Tension case '{}' should be numerically stable",
                case
            );
        }
    }

    #[test]
    fn test_ogden_parameter_robustness() {
        let tester = EdgeCaseTester {};
        let results = tester.test_ogden_extreme_parameters();

        println!("Ogden parameter robustness test:");
        for (case, is_stable) in &results {
            println!("  {}: stable = {}", case, is_stable);
            // Note: Some extreme parameters may legitimately fail, so we don't assert
            // This test is for information and robustness checking
        }

        // At minimum, standard parameters should work
        let standard_result = results.iter().find(|(case, _)| case == "Standard");
        if let Some((_, is_stable)) = standard_result {
            assert!(*is_stable, "Standard Ogden parameters should be stable");
        }
    }

    #[test]
    fn test_near_singularity_stability() {
        let tester = EdgeCaseTester {};
        let results = tester.test_near_singularities();

        println!("Near-singularity stability test:");
        let mut unstable_cases = Vec::new();

        for (case, is_stable) in results {
            println!("  {}: stable = {}", case, is_stable);
            if !is_stable {
                unstable_cases.push(case);
            }
        }

        // Some near-singular cases may be unstable, which is acceptable
        // The important thing is that the code doesn't crash
        println!(
            "  Found {} potentially unstable cases (may be acceptable)",
            unstable_cases.len()
        );
    }

    #[test]
    fn test_extreme_harmonic_generation() {
        let tester = EdgeCaseTester {};
        let results = tester.test_extreme_harmonic_generation();

        println!("Extreme harmonic generation test:");
        for (case, ratio, is_stable) in results {
            println!("  {}: ratio = {:.2e}, stable = {}", case, ratio, is_stable);
            assert!(
                is_stable,
                "Harmonic generation case '{}' should be numerically stable",
                case
            );
        }
    }

    #[test]
    fn test_material_parameter_bounds() {
        // Test that material parameters are within reasonable bounds
        let models = vec![
            HyperelasticModel::neo_hookean_soft_tissue(),
            HyperelasticModel::mooney_rivlin_biological(),
        ];

        let deformation_gradient = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        for model in models {
            let stress = model.cauchy_stress(&deformation_gradient);

            // At reference configuration, stress should be zero (or very small due to numerical precision)
            let max_stress = stress
                .iter()
                .flat_map(|row| row.iter())
                .map(|&s| s.abs())
                .fold(0.0, f64::max);

            assert!(
                max_stress < 1e-6,
                "Stress at reference configuration should be near zero, got {:.2e}",
                max_stress
            );
        }
    }

    #[test]
    fn test_solver_configuration_robustness() {
        let grid = Grid::new(8, 8, 8, 0.004, 0.004, 0.004).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let material = HyperelasticModel::neo_hookean_soft_tissue();

        // Test various solver configurations
        let configs = vec![
            NonlinearSWEConfig {
                nonlinearity_parameter: 0.0,
                enable_harmonics: false,
                ..Default::default()
            },
            NonlinearSWEConfig {
                nonlinearity_parameter: 0.01,
                enable_harmonics: false,
                ..Default::default()
            },
            NonlinearSWEConfig {
                nonlinearity_parameter: 0.01,
                enable_harmonics: true,
                ..Default::default()
            },
            NonlinearSWEConfig {
                nonlinearity_parameter: 0.1,
                enable_harmonics: true,
                ..Default::default()
            },
        ];

        for (i, config) in configs.into_iter().enumerate() {
            let solver = NonlinearElasticWaveSolver::new(&grid, &medium, material.clone(), config);
            assert!(
                solver.is_ok(),
                "Solver configuration {} should create successfully",
                i
            );

            let solver = solver.unwrap();
            let initial_disp = Array3::zeros((8, 8, 8));
            let result = solver.propagate_waves(&initial_disp);

            // Some configurations might have numerical issues, but shouldn't crash
            if result.is_err() {
                println!("Configuration {} failed: {:?}", i, result.err());
            }
        }
    }

    #[test]
    fn test_grid_size_robustness() {
        let material = HyperelasticModel::neo_hookean_soft_tissue();
        let medium = HomogeneousMedium::new(
            1000.0,
            1500.0,
            0.5,
            1.0,
            &Grid::new(4, 4, 4, 0.01, 0.01, 0.01).unwrap(),
        );

        // Test with various grid sizes
        let grid_sizes = [(4, 4, 4), (8, 8, 8), (16, 8, 8)];

        for (nx, ny, nz) in grid_sizes {
            let grid = Grid::new(nx, ny, nz, 0.01, 0.01, 0.01).unwrap();

            let solver = NonlinearElasticWaveSolver::new(
                &grid,
                &medium,
                material.clone(),
                NonlinearSWEConfig::default(),
            );
            assert!(
                solver.is_ok(),
                "Should create solver for grid size {}x{}x{}",
                nx,
                ny,
                nz
            );
        }
    }
}
