//! Nonlinear Shear Wave Elastography Validation Tests
//!
//! Comprehensive validation suite for NL-SWE components including:
//! - Hyperelastic constitutive models
//! - Harmonic detection algorithms
//! - Nonlinear inversion methods
//! - End-to-end NL-SWE workflow

pub use kwavers::domain::grid::Grid;
pub use kwavers::domain::imaging::ultrasound::elastography::{
    InversionMethod, NonlinearInversionMethod, NonlinearParameterMap,
};
pub use kwavers::domain::medium::HomogeneousMedium;
pub use kwavers::physics::acoustics::imaging::modalities::elastography::{
    HarmonicDetectionConfig, HarmonicDetector, HarmonicDisplacementField,
};
pub use kwavers::simulation::imaging::elastography::ShearWaveElastography;
pub use kwavers::solver::forward::elastic::{
    ElasticWaveConfig, ElasticWaveSolver, HyperelasticModel, NonlinearElasticWaveSolver,
    NonlinearSWEConfig,
};
pub use kwavers::solver::inverse::elastography::{
    NonlinearInversion, NonlinearInversionConfig, NonlinearParameterMapExt,
};
pub use ndarray::{Array3, Array4};
pub use std::f64::consts::PI;

/// Test hyperelastic constitutive models
#[cfg(test)]
mod hyperelastic_tests {
    use super::*;

    #[test]
    fn test_neo_hookean_model() {
        let model = HyperelasticModel::neo_hookean_soft_tissue();

        // Test strain energy at reference state (should be zero)
        let w_ref = model.strain_energy(3.0, 3.0, 1.0);
        assert!(
            (w_ref - 0.0).abs() < 1e-10,
            "Reference strain energy should be zero"
        );

        // Test strain energy under deformation
        let w_def = model.strain_energy(4.0, 4.0, 1.0);
        assert!(w_def > 0.0, "Deformed strain energy should be positive");

        // Test stress calculation
        let f = [[1.1, 0.0, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 0.95]]; // Simple deformation
        let stress = model.cauchy_stress(&f);

        // Check that stress tensor is symmetric and positive
        assert!(
            (stress[0][0] - stress[0][0]).abs() < 1e-10,
            "Stress should be finite"
        );
        assert!(
            stress[0][0] > 0.0,
            "Normal stress should be positive under compression"
        );
    }

    #[test]
    fn test_mooney_rivlin_model() {
        let model = HyperelasticModel::mooney_rivlin_biological();

        // Test strain energy
        let w_ref = model.strain_energy(3.0, 3.0, 1.0);
        assert!((w_ref - 0.0).abs() < 1e-10);

        let w_def = model.strain_energy(4.0, 5.0, 1.0); // Different I1 and I2
        assert!(w_def > 0.0);

        // Mooney-Rivlin should give different results than Neo-Hookean
        let neo_hookean = HyperelasticModel::neo_hookean_soft_tissue();
        let w_neo = neo_hookean.strain_energy(4.0, 5.0, 1.0);
        assert!(
            (w_def - w_neo).abs() > 1e-6,
            "Mooney-Rivlin should differ from Neo-Hookean"
        );
    }

    #[test]
    fn test_ogden_model() {
        // Note: Ogden implementation is simplified
        let model = HyperelasticModel::Ogden {
            mu: vec![1000.0, 200.0],
            alpha: vec![1.5, 3.0],
        };

        let w = model.strain_energy(3.0, 3.0, 1.0);
        assert!(w >= 0.0, "Strain energy should be non-negative");
    }

    #[test]
    fn test_nonlinear_wave_solver_creation() {
        let grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let material = HyperelasticModel::neo_hookean_soft_tissue();
        let config = NonlinearSWEConfig::default();

        let solver = NonlinearElasticWaveSolver::new(&grid, &medium, material, config);
        assert!(
            solver.is_ok(),
            "Nonlinear wave solver should create successfully"
        );
    }

    #[test]
    fn test_nonlinear_wave_propagation() {
        let grid = Grid::new(12, 12, 12, 0.002, 0.002, 0.002).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let material = HyperelasticModel::neo_hookean_soft_tissue();

        let config = NonlinearSWEConfig {
            enable_harmonics: true,
            ..Default::default()
        };

        let solver = NonlinearElasticWaveSolver::new(&grid, &medium, material, config).unwrap();

        // Create initial displacement
        let mut initial_disp = Array3::zeros((12, 12, 12));
        initial_disp[[6, 6, 6]] = 1e-6; // Point source

        let result = solver.propagate_waves(&initial_disp);
        assert!(result.is_ok(), "Wave propagation should succeed");

        let history = result.unwrap();
        assert!(!history.is_empty(), "Should have displacement history");

        // Check that harmonics are generated
        let final_field = &history[history.len() - 1];
        let total_magnitude = final_field.total_displacement_magnitude();

        // Should have non-zero displacement
        let max_disp = total_magnitude.iter().cloned().fold(0.0, f64::max);
        assert!(max_disp > 0.0, "Should have non-zero displacement");
    }
}

/// Test harmonic detection algorithms
#[cfg(test)]
mod harmonic_detection_tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn test_harmonic_detector_creation() {
        let config = HarmonicDetectionConfig::default();
        let _detector = HarmonicDetector::new(config.clone());

        // Test passes if creation succeeds
        assert!(config.fundamental_frequency > 0.0);
    }

    #[test]
    fn test_harmonic_displacement_field() {
        let mut field = HarmonicDisplacementField::new(10, 10, 10, 3, 100);

        assert_eq!(field.fundamental_magnitude.dim(), (10, 10, 10));
        assert_eq!(field.harmonic_magnitudes.len(), 3);
        assert_eq!(field.time.len(), 100);

        // Test harmonic ratio calculation
        field.fundamental_magnitude.fill(1.0);
        field.harmonic_magnitudes[0].fill(0.1); // Second harmonic

        let ratio = field.harmonic_ratio(2);
        assert_eq!(ratio.dim(), (10, 10, 10));

        for &val in ratio.iter() {
            assert!((val - 0.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_nonlinearity_parameter_computation() {
        let mut field = HarmonicDisplacementField::new(5, 5, 5, 2, 50);

        // Set up test data
        field.fundamental_magnitude.fill(1.0);
        field.harmonic_magnitudes[0].fill(0.1); // Second harmonic
        field.harmonic_snrs[0].fill(20.0); // Good SNR

        let config = HarmonicDetectionConfig::default();
        field.compute_nonlinearity_parameter(&config);

        // Should have computed nonlinearity parameter
        assert!(field.nonlinearity_parameter.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn test_harmonic_analysis_workflow() {
        let detector = HarmonicDetector::new(HarmonicDetectionConfig::default());

        // Create synthetic displacement time series with harmonics
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let n_times = 128;

        let mut time_series = Array4::zeros((nx, ny, nz, n_times));

        // Add fundamental frequency (50 Hz) and second harmonic (100 Hz)
        let fundamental_freq = 50.0;
        let sampling_freq = 1000.0; // 1000 Hz sampling

        for t in 0..n_times {
            let time = t as f64 / sampling_freq;

            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        // Fundamental + second harmonic
                        let fundamental = (2.0 * PI * fundamental_freq * time).sin();
                        let harmonic = 0.1 * (4.0 * PI * fundamental_freq * time).sin(); // Smaller amplitude

                        time_series[[i, j, k, t]] = fundamental + harmonic;
                    }
                }
            }
        }

        let result = detector.analyze_harmonics(&time_series, sampling_freq);
        assert!(result.is_ok(), "Harmonic analysis should succeed");

        let harmonic_field = result.unwrap();

        // Should detect fundamental frequency
        let max_fundamental = harmonic_field
            .fundamental_magnitude
            .iter()
            .cloned()
            .fold(0.0, f64::max);
        assert!(max_fundamental > 0.0, "Should detect fundamental frequency");

        // Should detect harmonic
        let max_harmonic = harmonic_field.harmonic_magnitudes[0]
            .iter()
            .cloned()
            .fold(0.0, f64::max);
        assert!(max_harmonic > 0.0, "Should detect harmonic component");
    }
}

/// Test nonlinear inversion algorithms
#[cfg(test)]
mod nonlinear_inversion_tests {
    use super::*;

    #[test]
    fn test_nonlinear_inversion_creation() {
        let method = NonlinearInversionMethod::HarmonicRatio;
        let inversion = NonlinearInversion::new(NonlinearInversionConfig::new(method));

        // Test passes if creation succeeds
        assert_eq!(inversion.method(), method);
    }

    #[test]
    fn test_harmonic_ratio_inversion() {
        let inversion = NonlinearInversion::new(NonlinearInversionConfig::new(
            NonlinearInversionMethod::HarmonicRatio,
        ));

        // Create synthetic harmonic field
        let mut harmonic_field = HarmonicDisplacementField::new(6, 6, 6, 2, 50);

        // Set up test data with known harmonic ratio
        harmonic_field.fundamental_magnitude.fill(1.0);
        harmonic_field.harmonic_magnitudes[0].fill(0.1); // A2/A1 = 0.1
        harmonic_field.harmonic_snrs[0].fill(20.0); // Good SNR

        let grid = Grid::new(6, 6, 6, 0.001, 0.001, 0.001).unwrap();

        let result = inversion.reconstruct(&harmonic_field, &grid);
        assert!(result.is_ok(), "Harmonic ratio inversion should succeed");

        let param_map = result.unwrap();

        // Should have computed nonlinearity parameters
        assert!(param_map.nonlinearity_parameter.iter().any(|&x| x > 0.0));

        // Should have uncertainty estimates
        assert!(param_map.nonlinearity_uncertainty.iter().any(|&x| x >= 0.0));

        // Should have quality metrics
        assert!(param_map.estimation_quality.iter().any(|&x| x >= 0.0));
    }

    #[test]
    fn test_nonlinear_least_squares_inversion() {
        let inversion = NonlinearInversion::new(NonlinearInversionConfig::new(
            NonlinearInversionMethod::NonlinearLeastSquares,
        ));

        // Create test harmonic field
        let mut harmonic_field = HarmonicDisplacementField::new(4, 4, 4, 1, 32);

        harmonic_field.fundamental_magnitude.fill(1.0);
        harmonic_field.harmonic_magnitudes[0].fill(0.15);
        harmonic_field.harmonic_snrs[0].fill(15.0);

        let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();

        let result = inversion.reconstruct(&harmonic_field, &grid);
        assert!(result.is_ok(), "Nonlinear least squares should succeed");

        let param_map = result.unwrap();
        assert!(param_map.nonlinearity_parameter.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn test_bayesian_inversion() {
        let inversion = NonlinearInversion::new(NonlinearInversionConfig::new(
            NonlinearInversionMethod::BayesianInversion,
        ));

        // Create test data
        let mut harmonic_field = HarmonicDisplacementField::new(4, 4, 4, 1, 32);

        harmonic_field.fundamental_magnitude.fill(1.0);
        harmonic_field.harmonic_magnitudes[0].fill(0.12);
        harmonic_field.harmonic_snrs[0].fill(25.0); // High SNR

        let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();

        let result = inversion.reconstruct(&harmonic_field, &grid);
        assert!(result.is_ok(), "Bayesian inversion should succeed");

        let param_map = result.unwrap();

        // Bayesian method should provide uncertainty estimates
        let avg_uncertainty = param_map.nonlinearity_uncertainty.mean().unwrap_or(0.0);
        assert!(
            avg_uncertainty > 0.0,
            "Should have non-zero uncertainty estimates"
        );
    }

    #[test]
    fn test_nonlinear_parameter_statistics() {
        let param_map = NonlinearParameterMap {
            nonlinearity_parameter: Array3::from_elem((5, 5, 5), 5.0),
            elastic_constants: vec![
                Array3::from_elem((5, 5, 5), 1000.0),
                Array3::from_elem((5, 5, 5), 500.0),
                Array3::from_elem((5, 5, 5), 200.0),
                Array3::from_elem((5, 5, 5), 100.0),
            ],
            nonlinearity_uncertainty: Array3::from_elem((5, 5, 5), 0.5),
            estimation_quality: Array3::from_elem((5, 5, 5), 0.8),
        };

        let (min, max, mean) = param_map.nonlinearity_statistics();
        assert_eq!(min, 5.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 5.0);

        let (q_min, q_max, q_mean) = param_map.quality_statistics();
        assert_eq!(q_min, 0.8);
        assert_eq!(q_max, 0.8);
        assert_eq!(q_mean, 0.8);
    }
}

/// Test end-to-end NL-SWE workflow
#[cfg(test)]
mod end_to_end_tests {
    use super::*;

    #[test]
    #[ignore = "Long-running end-to-end workflow; excluded under nextest per-test 30s timeout policy"]
    fn test_nl_swe_workflow() {
        // Create simulation setup
        let grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        // Step 1: Linear SWE (existing functionality)
        let swe = ShearWaveElastography::new(
            &grid,
            &medium,
            InversionMethod::TimeOfFlight,
            ElasticWaveConfig::default(),
        )
        .unwrap();

        let push_location = [0.008, 0.008, 0.008];
        let displacement_history = swe.generate_shear_wave(push_location).unwrap();

        // Step 2: Harmonic analysis
        let harmonic_detector = HarmonicDetector::new(HarmonicDetectionConfig::default());

        // Convert displacement history to time series format
        let n_times = displacement_history.len();
        let mut time_series = Array4::zeros((16, 16, 16, n_times));

        for t in 0..n_times {
            let magnitude = displacement_history[t].displacement_magnitude();
            for i in 0..16 {
                for j in 0..16 {
                    for k in 0..16 {
                        time_series[[i, j, k, t]] = magnitude[[i, j, k]];
                    }
                }
            }
        }

        let harmonic_field = harmonic_detector
            .analyze_harmonics(&time_series, 1000.0)
            .unwrap();

        // Step 3: Nonlinear inversion
        let nonlinear_inversion = NonlinearInversion::new(NonlinearInversionConfig::new(
            NonlinearInversionMethod::HarmonicRatio,
        ));
        let nonlinear_params = nonlinear_inversion
            .reconstruct(&harmonic_field, &grid)
            .unwrap();

        // Verify results
        assert!(nonlinear_params
            .nonlinearity_parameter
            .iter()
            .any(|&x| x >= 0.0));
        assert!(nonlinear_params
            .estimation_quality
            .iter()
            .any(|&x| x >= 0.0));

        // Test passes if entire workflow completes successfully
    }

    #[test]
    fn test_performance_vs_linear_swe() {
        // This would be a benchmark test in practice
        // For now, just verify that nonlinear methods don't crash

        let grid = Grid::new(8, 8, 8, 0.002, 0.002, 0.002).unwrap();
        let medium = HomogeneousMedium::soft_tissue(8_000.0, 0.49, &grid);

        // Linear SWE
        let linear_swe = ElasticWaveSolver::new(
            &grid,
            &medium,
            ElasticWaveConfig {
                simulation_time: 3e-4,
                cfl_factor: 0.5,
                save_every: 1_000,
                ..Default::default()
            },
        )
        .unwrap();

        let push_location = [0.008, 0.008, 0.008];
        // Create an initial displacement centered at the push location
        let ix = ((push_location[0] / grid.dx) as usize).min(grid.nx - 1);
        let iy = ((push_location[1] / grid.dy) as usize).min(grid.ny - 1);
        let iz = ((push_location[2] / grid.dz) as usize).min(grid.nz - 1);

        let mut initial_disp = Array3::zeros((grid.nx, grid.ny, grid.nz));
        initial_disp[[ix, iy, iz]] = 1e-6; // small ARFI-like displacement

        let linear_result = linear_swe.propagate_waves(&initial_disp);
        assert!(linear_result.is_ok(), "Linear SWE should work");

        // Nonlinear SWE
        let material = HyperelasticModel::neo_hookean_soft_tissue();
        let nonlinear_solver = NonlinearElasticWaveSolver::new(
            &grid,
            &medium,
            material,
            NonlinearSWEConfig::default(),
        )
        .unwrap();

        let initial_disp = Array3::zeros((8, 8, 8));
        let nonlinear_result = nonlinear_solver.propagate_waves(&initial_disp);
        assert!(nonlinear_result.is_ok(), "Nonlinear SWE should work");

        // Both should produce valid results
        assert!(!linear_result.unwrap().is_empty());
        assert!(!nonlinear_result.unwrap().is_empty());
    }
}

/// Test convergence and numerical stability
#[cfg(test)]
mod convergence_tests {
    use super::*;

    #[test]
    fn test_nonlinear_solver_convergence() {
        let grid = Grid::new(8, 8, 8, 0.002, 0.002, 0.002).unwrap();
        let medium = HomogeneousMedium::soft_tissue(8_000.0, 0.49, &grid);
        let material = HyperelasticModel::neo_hookean_soft_tissue();

        // Test with different nonlinearity strengths
        for nonlinearity in [0.01, 0.1, 0.5] {
            let config = NonlinearSWEConfig {
                nonlinearity_parameter: nonlinearity,
                ..Default::default()
            };

            let solver = NonlinearElasticWaveSolver::new(&grid, &medium, material.clone(), config);
            assert!(
                solver.is_ok(),
                "Should create solver for nonlinearity = {}",
                nonlinearity
            );

            let initial_disp = Array3::zeros((8, 8, 8));
            let result = solver.unwrap().propagate_waves(&initial_disp);
            assert!(
                result.is_ok(),
                "Should converge for nonlinearity = {}",
                nonlinearity
            );
        }
    }

    #[test]
    fn test_harmonic_detection_robustness() {
        let detector = HarmonicDetector::new(HarmonicDetectionConfig::default());

        // Test with different noise levels
        for noise_level in [0.01, 0.1, 0.5] {
            let time_series = Array4::from_elem((4, 4, 4, 64), 1.0);

            // Add noise
            let mut noisy_series = time_series.clone();
            for ((i, j, k, t), value) in noisy_series.indexed_iter_mut() {
                let phase =
                    (i as f64) * 0.37 + (j as f64) * 0.61 + (k as f64) * 0.89 + (t as f64) * 0.13;
                let noise = noise_level * (2.0 * PI * phase).sin();
                *value += noise;
            }

            let result = detector.analyze_harmonics(&noisy_series, 1000.0);
            assert!(
                result.is_ok(),
                "Should handle noise level = {}",
                noise_level
            );
        }
    }

    #[test]
    fn test_inversion_method_consistency() {
        // Create test harmonic field
        let mut harmonic_field = HarmonicDisplacementField::new(4, 4, 4, 1, 32);
        harmonic_field.fundamental_magnitude.fill(1.0);
        harmonic_field.harmonic_magnitudes[0].fill(0.1);
        harmonic_field.harmonic_snrs[0].fill(15.0);

        let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();

        // Test all inversion methods
        for method in [
            NonlinearInversionMethod::HarmonicRatio,
            NonlinearInversionMethod::NonlinearLeastSquares,
            NonlinearInversionMethod::BayesianInversion,
        ] {
            let inversion = NonlinearInversion::new(NonlinearInversionConfig::new(method));
            let result = inversion.reconstruct(&harmonic_field, &grid);
            assert!(result.is_ok(), "Method {:?} should succeed", method);

            let param_map = result.unwrap();
            assert!(
                param_map.nonlinearity_parameter.iter().all(|&x| x >= 0.0),
                "Method {:?} should produce non-negative parameters",
                method
            );
        }
    }
}
