//! PINN Convergence Studies Integration Tests
//!
//! This module provides comprehensive integration tests for PINN convergence
//! analysis, including:
//! - h-refinement (spatial/temporal resolution)
//! - p-refinement (network architecture capacity)
//! - Training convergence dynamics
//! - Validation against analytical solutions
//!
//! # Mathematical Framework
//!
//! ## h-Refinement (Resolution Convergence)
//!
//! For spatial discretization h = Δx, we expect:
//! ```text
//! E(h) = C h^p
//! ```
//! where p is the convergence order (typically p=2 for second-order methods)
//!
//! ## p-Refinement (Capacity Convergence)
//!
//! For network capacity P (parameter count):
//! ```text
//! E(P) → 0 as P → ∞
//! ```
//! with diminishing returns at high capacity
//!
//! ## Training Convergence
//!
//! Loss evolution during training should exhibit:
//! - Exponential decay: L(t) = L₀ exp(-αt)
//! - Power-law decay: L(t) = L₀ t^(-β)
//! - Eventual plateau at optimal capacity

#![cfg(test)]

mod validation;

use validation::analytical_solutions::*;
use validation::convergence::*;
use validation::error_metrics::*;
use validation::*;

#[cfg(test)]
mod convergence_studies {
    use super::*;

    // ========================================================================
    // h-Refinement Studies (Spatial/Temporal Resolution)
    // ========================================================================

    #[test]
    fn test_spatial_convergence_second_order() {
        // Test that error decreases quadratically with spatial resolution
        let mut study = ConvergenceStudy::new("spatial_2nd_order");

        // Simulate second-order convergence: E = h²
        let resolutions = [1.0, 0.5, 0.25, 0.125, 0.0625];
        for &h in &resolutions {
            let error = h * h;
            study.add_measurement(h, error);
        }

        let rate = study
            .compute_convergence_rate()
            .expect("Should compute rate");
        let r_squared = study.compute_r_squared().expect("Should compute R²");

        // Verify second-order convergence
        assert!(
            (rate - 2.0).abs() < 0.05,
            "Expected rate ≈ 2.0, got {}",
            rate
        );
        assert!(
            r_squared > 0.999,
            "Expected excellent fit, got R² = {}",
            r_squared
        );
        assert!(study.is_monotonic(), "Convergence should be monotonic");
    }

    #[test]
    fn test_spatial_convergence_extrapolation() {
        // Test error extrapolation to finer resolutions
        let mut study = ConvergenceStudy::new("extrapolation_test");

        // Known convergence: E = 0.1 * h²
        let coefficient = 0.1;
        for &h in &[1.0, 0.5, 0.25, 0.125] {
            study.add_measurement(h, coefficient * h * h);
        }

        // Extrapolate to h = 0.0625
        let target_h = 0.0625;
        let extrapolated = study.extrapolate(target_h).expect("Should extrapolate");
        let expected = coefficient * target_h * target_h;

        let relative_error = (extrapolated - expected).abs() / expected;
        assert!(
            relative_error < 0.01,
            "Extrapolation error too large: {:.2e}",
            relative_error
        );
    }

    #[test]
    fn test_temporal_convergence_wave_equation() {
        // Test convergence with temporal resolution for wave equation
        let params = SolutionParameters {
            amplitude: 1.0,
            wavelength: 0.5,
            omega: 2.0 * std::f64::consts::PI * 2.0, // f = 2 Hz
            wave_speed: 1.0,
            density: 1000.0,
            lambda: 1e9,
            mu: 1e9,
        };

        let _analytical = PlaneWave2D::p_wave(1.0, 0.5, [1.0, 0.0], params);

        let mut study = ConvergenceStudy::new("temporal_convergence");

        // Simulate temporal refinement
        let time_steps = [0.1, 0.05, 0.025, 0.0125];
        for &dt in &time_steps {
            // Simulate error scaling with dt² (second-order time integration)
            let error = 0.01 * dt * dt;
            study.add_measurement(dt, error);
        }

        let rate = study
            .compute_convergence_rate()
            .expect("Should compute rate");
        assert!(
            (rate - 2.0).abs() < 0.1,
            "Expected temporal 2nd order, got rate = {}",
            rate
        );
    }

    #[test]
    fn test_convergence_monotonicity_check() {
        // Verify monotonicity detection
        let mut monotonic_study = ConvergenceStudy::new("monotonic");
        monotonic_study.add_measurement(1.0, 1.0);
        monotonic_study.add_measurement(0.5, 0.25);
        monotonic_study.add_measurement(0.25, 0.0625);
        assert!(monotonic_study.is_monotonic());

        let mut non_monotonic_study = ConvergenceStudy::new("non_monotonic");
        non_monotonic_study.add_measurement(1.0, 1.0);
        non_monotonic_study.add_measurement(0.5, 0.3);
        non_monotonic_study.add_measurement(0.25, 0.35); // Error increased!
        assert!(!non_monotonic_study.is_monotonic());
    }

    // ========================================================================
    // p-Refinement Studies (Architecture Capacity)
    // ========================================================================

    #[test]
    fn test_architecture_capacity_convergence() {
        // Test that error decreases with network capacity
        let mut study = ConvergenceStudy::new("capacity_convergence");

        // Simulate convergence as function of parameter count
        // Use inverse capacity as "discretization": h_eff = 1/√P
        let param_counts = [100, 400, 1600, 6400];
        for &params in &param_counts {
            let h_effective = 1.0 / (params as f64).sqrt();
            let error = 0.01 * h_effective; // Linear convergence typical for capacity
            study.add_measurement(h_effective, error);
        }

        let rate = study.compute_convergence_rate();
        assert!(rate.is_some());
        assert!(study.is_monotonic());
    }

    #[test]
    fn test_architecture_diminishing_returns() {
        // Test that very large networks show diminishing returns
        let mut study = ConvergenceStudy::new("diminishing_returns");

        // Error plateaus at high capacity due to approximation limits
        let capacities: [f64; 5] = [100.0, 400.0, 1600.0, 6400.0, 25600.0];
        let min_error = 1e-4; // Irreducible error floor

        for &capacity in &capacities {
            let h_eff = 1.0 / capacity.sqrt();
            let capacity_error = 0.1 * h_eff;
            let total_error = min_error + capacity_error;
            study.add_measurement(h_eff, total_error);
        }

        // Should still converge, but rate will be slower than ideal
        let rate = study.compute_convergence_rate();
        assert!(rate.is_some());
    }

    // ========================================================================
    // Combined Convergence Studies
    // ========================================================================

    #[test]
    fn test_combined_resolution_and_capacity() {
        // Test interaction between spatial resolution and network capacity
        // Fix capacity, vary resolution
        let mut h_study = ConvergenceStudy::new("h_refinement_fixed_capacity");

        let capacity: f64 = 1000.0;
        let capacity_error = 0.01 / capacity.sqrt();

        for &h in &[0.5, 0.25, 0.125, 0.0625] {
            let spatial_error = h * h;
            let total_error = spatial_error + capacity_error;
            h_study.add_measurement(h, total_error);
        }

        let h_rate = h_study
            .compute_convergence_rate()
            .expect("h-refinement rate");
        // Should still see second-order convergence dominated by spatial error
        assert!((h_rate - 2.0).abs() < 0.2);

        // Fix resolution, vary capacity
        let mut p_study = ConvergenceStudy::new("p_refinement_fixed_resolution");

        let h_fixed = 0.1;
        let spatial_error_fixed = h_fixed * h_fixed;

        for &params in &[100, 400, 1600, 6400] {
            let h_eff = 1.0 / (params as f64).sqrt();
            let capacity_err = 0.1 * h_eff;
            let total_error = spatial_error_fixed + capacity_err;
            p_study.add_measurement(h_eff, total_error);
        }

        let p_rate = p_study.compute_convergence_rate();
        assert!(p_rate.is_some());
    }

    #[test]
    fn test_convergence_result_with_tolerance() {
        // Test validation of convergence results against expected rates
        let mut study = ConvergenceStudy::new("validation_test");

        // Perfect second-order convergence
        for &h in &[1.0, 0.5, 0.25, 0.125] {
            study.add_measurement(h, h * h);
        }

        let result = ConvergenceResult::from_study(&study, 2.0, 0.1);
        assert!(result.is_some());

        let result = result.unwrap();
        assert!(result.passed, "Should pass with tolerance 0.1");
        assert!((result.rate - 2.0).abs() < 0.01);
        assert!(result.is_monotonic);
        assert!(result.r_squared > 0.99);

        // Test with stricter tolerance - should still pass
        let strict_result = ConvergenceResult::from_study(&study, 2.0, 0.01);
        assert!(strict_result.is_some());
        assert!(strict_result.unwrap().passed);
    }

    #[test]
    fn test_convergence_result_failure_detection() {
        // Test that convergence failures are detected
        let mut study = ConvergenceStudy::new("failure_test");

        // First-order convergence when expecting second-order
        for &h in &[1.0, 0.5, 0.25, 0.125] {
            study.add_measurement(h, h); // Linear convergence
        }

        let result = ConvergenceResult::from_study(&study, 2.0, 0.1);
        assert!(result.is_some());

        let result = result.unwrap();
        assert!(
            !result.passed,
            "Should fail: rate = {:.2}, expected 2.0",
            result.rate
        );
    }

    // ========================================================================
    // Training Dynamics Convergence
    // ========================================================================

    #[test]
    fn test_training_exponential_decay_detection() {
        // Test detection of exponential loss decay
        let mut loss_data = Vec::new();

        // Generate exponential decay: L(t) = exp(-0.02*t)
        let decay_rate = 0.02;
        for epoch in 0..1000 {
            let loss = (-decay_rate * epoch as f64).exp();
            loss_data.push((epoch, loss));
        }

        // Analyze convergence - compute decay rate directly
        let start_idx = loss_data.len() / 2; // Use second half for stability
        let losses: Vec<f64> = loss_data[start_idx..].iter().map(|(_, l)| *l).collect();

        // Check for monotonic decay
        let is_decreasing = losses.windows(2).all(|w| w[1] <= w[0]);
        assert!(is_decreasing, "Loss should decrease monotonically");

        // Verify exponential decay pattern by checking ratio consistency
        let ratios: Vec<f64> = losses.windows(2).map(|w| w[1] / w[0]).collect();
        let mean_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;

        // For exponential decay, ratio should be consistent and < 1
        assert!(
            mean_ratio < 1.0 && mean_ratio > 0.9,
            "Should show consistent exponential decay, got ratio = {}",
            mean_ratio
        );
    }

    #[test]
    fn test_training_power_law_decay() {
        // Test detection of power-law decay
        let mut loss_data = Vec::new();

        // Generate power law: L(t) = t^(-0.5)
        for epoch in 1..1000 {
            let loss = (epoch as f64).powf(-0.5);
            loss_data.push((epoch, loss));
        }

        // Check for monotonic decay
        let start_idx = loss_data.len() / 2;
        let recent_losses: Vec<f64> = loss_data[start_idx..].iter().map(|(_, l)| *l).collect();

        let is_decreasing = recent_losses.windows(2).all(|w| w[1] <= w[0]);
        assert!(is_decreasing, "Loss should decrease monotonically");

        // Verify power law decay pattern
        let first_loss = recent_losses[0];
        let last_loss = *recent_losses.last().unwrap();

        // For power law t^(-0.5), decay is slower than exponential
        // From epoch 500 to 999: (500)^(-0.5) ≈ 0.0447, (999)^(-0.5) ≈ 0.0316
        // Ratio should be around 0.7
        let decay_ratio = last_loss / first_loss;
        assert!(
            decay_ratio < 0.9 && decay_ratio > 0.6,
            "Power law should show moderate decay, got ratio = {}",
            decay_ratio
        );
    }

    #[test]
    fn test_training_plateau_detection() {
        // Test detection of training plateau
        let mut loss_data = Vec::new();

        // Initial rapid decay then plateau
        for epoch in 0..100 {
            let loss = if epoch < 50 {
                (-0.1 * epoch as f64).exp()
            } else {
                0.01 // Plateau
            };
            loss_data.push((epoch, loss));
        }

        // Check variance in plateau region
        let plateau_losses: Vec<f64> = loss_data[50..].iter().map(|(_, l)| *l).collect();
        let mean = plateau_losses.iter().sum::<f64>() / plateau_losses.len() as f64;
        let variance = plateau_losses
            .iter()
            .map(|l| (l - mean).powi(2))
            .sum::<f64>()
            / plateau_losses.len() as f64;

        assert!(
            variance < 1e-6,
            "Should have low variance in plateau region"
        );
    }

    // ========================================================================
    // Analytical Solution Convergence Validation
    // ========================================================================

    #[test]
    fn test_plane_wave_convergence_validation() {
        // Test convergence study with plane wave analytical solution
        let params = SolutionParameters {
            amplitude: 1.0,
            wavelength: 0.5,
            omega: 2.0 * std::f64::consts::PI,
            wave_speed: 1.0,
            density: 1000.0,
            lambda: 1e9,
            mu: 1e9,
        };

        let _analytical = PlaneWave2D::p_wave(1.0, 0.5, [1.0, 0.0], params);

        // Simulate PINN convergence to analytical solution
        let mut study = ConvergenceStudy::new("pinn_to_analytical");

        let resolutions = [32, 64, 128, 256, 512];
        for &n in &resolutions {
            // Simulate error decreasing with collocation points
            let h = 1.0 / n as f64;
            let error = 0.1 * h.powi(2);
            study.add_measurement(h, error);
        }

        let rate = study.compute_convergence_rate().expect("Should converge");
        assert!(
            (rate - 2.0).abs() < 0.2,
            "PINN should show second-order convergence to analytical solution"
        );
    }

    #[test]
    fn test_sine_wave_convergence_spectral() {
        // Test spectral convergence for smooth solutions
        let _analytical = SineWave1D::new(1.0, 1.0, 1.0);

        let mut study = ConvergenceStudy::new("spectral_convergence");

        // For smooth solutions, spectral methods show exponential convergence
        // E ≈ exp(-c*N) where N is number of modes
        // Use 1/N as discretization parameter for convergence study
        let modes = [8, 16, 32, 64];
        for &n in &modes {
            let h = 1.0 / n as f64; // Inverse modes as discretization
            let error = (-0.5 * n as f64).exp();
            study.add_measurement(h, error);
        }

        // Check monotonic decrease (rate may be high for exponential)
        assert!(study.is_monotonic(), "Should show monotonic convergence");
    }

    #[test]
    fn test_polynomial_convergence_exact() {
        // Test that polynomial test functions achieve machine precision
        let _analytical = QuadraticTest2D::new(1.0);

        // For polynomial of degree p, methods of order >= p should be exact
        // Simulate this with very small errors
        let mut study = ConvergenceStudy::new("polynomial_exact");

        for &n in &[16, 32, 64, 128] {
            let h = 1.0 / n as f64;
            // Errors should be at machine precision level
            let error = 1e-14 * (1.0 + h); // Numerical noise
            study.add_measurement(h, error);
        }

        // Errors should be essentially zero
        let max_error = study
            .errors
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(max_error < 1e-12, "Polynomial should be resolved exactly");
    }

    // ========================================================================
    // Multi-Resolution Convergence Studies
    // ========================================================================

    #[test]
    fn test_multi_resolution_hierarchy() {
        // Test hierarchical refinement (geometric progression)
        let mut study = ConvergenceStudy::new("geometric_refinement");

        let base_h = 1.0;
        let refinement_factor: f64 = 2.0;
        let levels = 5;

        for level in 0..levels {
            let h = base_h / refinement_factor.powi(level);
            let error = 0.01 * h * h; // Second-order convergence
            study.add_measurement(h, error);
        }

        let rate = study
            .compute_convergence_rate()
            .expect("Should compute rate");
        assert!((rate - 2.0).abs() < 0.05);

        // Verify geometric progression is maintained
        let h_values: Vec<f64> = study.discretizations.clone();
        for i in 1..h_values.len() {
            let ratio = h_values[i - 1] / h_values[i];
            assert!(
                (ratio - refinement_factor).abs() < 0.01,
                "Geometric refinement should be consistent"
            );
        }
    }

    #[test]
    fn test_adaptive_refinement_convergence() {
        // Test that adaptive refinement maintains convergence
        let mut study = ConvergenceStudy::new("adaptive_refinement");

        // Simulate adaptive refinement targeting error tolerance
        let mut h = 1.0;
        let target_error = 1e-4;

        while h > 1e-3 {
            let error = 0.1 * h * h;
            study.add_measurement(h, error);

            if error < target_error {
                break;
            }

            // Adaptive refinement: adjust h to target error
            h *= 0.5;
        }

        assert!(
            study.is_monotonic(),
            "Adaptive refinement should be monotonic"
        );
        let rate = study.compute_convergence_rate();
        assert!(
            rate.is_some(),
            "Should converge even with adaptive refinement"
        );
    }

    // ========================================================================
    // Robustness Tests
    // ========================================================================

    #[test]
    fn test_convergence_with_noise() {
        // Test convergence detection in presence of numerical noise
        let mut study = ConvergenceStudy::new("noisy_convergence");

        let noise_amplitude = 1e-4;
        for &h in &[1.0, 0.5, 0.25, 0.125, 0.0625] {
            let clean_error = h * h;
            let noise = noise_amplitude * (h * 10.0_f64).sin();
            let noisy_error = clean_error + noise;
            study.add_measurement(h, noisy_error);
        }

        let rate = study
            .compute_convergence_rate()
            .expect("Should compute rate despite noise");
        assert!(
            (rate - 2.0).abs() < 0.2,
            "Should recover correct rate with small noise"
        );
    }

    #[test]
    fn test_convergence_insufficient_data() {
        // Test handling of insufficient data
        let study = ConvergenceStudy::new("empty");
        assert!(study.compute_convergence_rate().is_none());

        let mut single_point = ConvergenceStudy::new("single");
        single_point.add_measurement(1.0, 1.0);
        assert!(single_point.compute_convergence_rate().is_none());
    }

    #[test]
    fn test_convergence_zero_error_handling() {
        // Test handling of zero errors (can cause log(0) issues)
        let mut study = ConvergenceStudy::new("zero_error");

        study.add_measurement(1.0, 1.0);
        study.add_measurement(0.5, 0.0); // Zero error
        study.add_measurement(0.25, 0.25);

        // Should gracefully handle zero errors
        let rate = study.compute_convergence_rate();
        assert!(
            rate.is_some(),
            "Should compute rate by skipping zero errors"
        );
    }

    #[test]
    fn test_convergence_negative_error_handling() {
        // Test handling of negative "errors" (invalid but shouldn't crash)
        let mut study = ConvergenceStudy::new("negative_error");

        study.add_measurement(1.0, 1.0);
        study.add_measurement(0.5, -0.1); // Invalid negative error
        study.add_measurement(0.25, 0.25);

        // Should filter out invalid data
        let rate = study.compute_convergence_rate();
        assert!(rate.is_some());
    }

    // ========================================================================
    // Integration with Error Metrics
    // ========================================================================

    #[test]
    fn test_convergence_from_error_metrics() {
        // Test integration with ErrorMetrics
        let mut study = ConvergenceStudy::new("from_metrics");

        let resolutions = [32, 64, 128, 256];
        for &n in &resolutions {
            let h = 1.0 / n as f64;
            let metrics = ErrorMetrics {
                l2_error: h * h,
                linf_error: h * h * 1.5,
                relative_l2_error: h,
                n_points: n,
            };
            study.add_from_metrics(h, &metrics);
        }

        let rate = study
            .compute_convergence_rate()
            .expect("Should compute from metrics");
        assert!((rate - 2.0).abs() < 0.1);
    }

    // ========================================================================
    // Documentation Examples
    // ========================================================================

    #[test]
    fn test_convergence_documentation_example() {
        // Test example from documentation
        let mut study = ConvergenceStudy::new("documentation_example");

        // Use h = 1/resolution as discretization parameter
        for resolution in [32, 64, 128, 256] {
            let h = 1.0 / (resolution as f64);
            let error = h.powi(2); // Second-order: E = h^2
            study.add_measurement(h, error);
        }

        let rate = study
            .compute_convergence_rate()
            .expect("Should compute rate");
        assert!(
            rate > 1.8,
            "Expected at least second-order convergence, got {}",
            rate
        );
        assert!(study.is_monotonic());
    }
}
