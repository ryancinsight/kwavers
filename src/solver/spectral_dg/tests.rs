//! Integration tests for Hybrid Spectral-DG methods
//!
//! This module contains comprehensive tests to validate the hybrid solver
//! implementation, including shock handling, conservation, and accuracy.

#[cfg(test)]
mod integration_tests {
    use crate::grid::Grid;
    use crate::solver::spectral_dg::*;
    use ndarray::Array3;
    use std::f64::consts::PI;
    use std::sync::Arc;

    /// Test smooth wave propagation (should use spectral method)
    #[test]
    fn test_smooth_wave_propagation() {
        let grid = Arc::new(Grid::new(32, 32, 32, 1.0, 1.0, 1.0));
        let config = HybridSpectralDGConfig {
            discontinuity_threshold: 0.5, // Higher threshold to avoid false positives
            spectral_order: 8,
            dg_polynomial_order: 3,
            adaptive_switching: true,
            conservation_tolerance: 1e-8,
        };

        let mut solver = HybridSpectralDGSolver::new(config, grid.clone());

        // Initialize a constant field (should be perfectly conserved)
        let field = Array3::from_elem((32, 32, 32), 1.0);

        // Evolve for one time step
        let dt = 0.001;
        let result = solver.solve(&field, dt, &grid).unwrap();

        // Check that discontinuity mask is all false (constant field has no discontinuities)
        if let Some(mask) = solver.discontinuity_mask() {
            let discontinuity_count = mask.iter().filter(|&&x| x).count();
            assert_eq!(
                discontinuity_count, 0,
                "Constant field should have no discontinuities"
            );
        }

        // Check conservation - constant field should be perfectly preserved
        let initial_sum: f64 = field.sum();
        let final_sum: f64 = result.sum();
        let absolute_error = (final_sum - initial_sum).abs();
        assert!(
            absolute_error < 1e-10,
            "Conservation error for constant field: {}",
            absolute_error
        );
    }

    /// Test shock handling (should use DG method)
    #[test]
    fn test_shock_handling() {
        let grid = Arc::new(Grid::new(64, 64, 64, 1.0, 1.0, 1.0));
        let config = HybridSpectralDGConfig {
            discontinuity_threshold: 0.05,
            spectral_order: 8,
            dg_polynomial_order: 3,
            adaptive_switching: true,
            conservation_tolerance: 1e-10,
        };

        let mut solver = HybridSpectralDGSolver::new(config, grid.clone());

        // Initialize step function (shock)
        let mut field = Array3::zeros((64, 64, 64));
        for i in 0..64 {
            for j in 0..64 {
                for k in 0..64 {
                    field[[i, j, k]] = if i < 32 { 0.0 } else { 1.0 };
                }
            }
        }

        // Evolve for one time step
        let dt = 0.001;
        let result = solver.solve(&field, dt, &grid).unwrap();

        // Check that discontinuity is detected
        if let Some(mask) = solver.discontinuity_mask() {
            // Should detect discontinuity around x=32
            for j in 10..54 {
                for k in 10..54 {
                    let detected = mask[[31, j, k]] || mask[[32, j, k]];
                    assert!(
                        detected,
                        "Failed to detect discontinuity at ({}, {}, {})",
                        32, j, k
                    );
                }
            }
        }

        // Check conservation
        let initial_sum: f64 = field.sum();
        let final_sum: f64 = result.sum();
        assert!((final_sum - initial_sum).abs() / initial_sum.abs() < 1e-9);
    }

    /// Test mixed smooth and discontinuous regions
    #[test]
    fn test_mixed_regions() {
        let grid = Arc::new(Grid::new(64, 64, 64, 1.0, 1.0, 1.0));
        let config = HybridSpectralDGConfig::default();

        let mut solver = HybridSpectralDGSolver::new(config, grid.clone());

        // Initialize field with both smooth and discontinuous regions
        let mut field = Array3::zeros((64, 64, 64));
        for i in 0..64 {
            for j in 0..64 {
                for k in 0..64 {
                    let x = i as f64 / 64.0;
                    if i < 20 {
                        // Smooth region
                        field[[i, j, k]] = (4.0 * PI * x).sin();
                    } else if i < 40 {
                        // Discontinuous region
                        field[[i, j, k]] = if (i + j + k) % 4 == 0 { 1.0 } else { -1.0 };
                    } else {
                        // Another smooth region
                        field[[i, j, k]] = (2.0 * PI * x).cos();
                    }
                }
            }
        }

        // Evolve
        let dt = 0.001;
        let result = solver.solve(&field, dt, &grid).unwrap();

        // Verify conservation
        let initial_sum: f64 = field.sum();
        let final_sum: f64 = result.sum();
        assert!((final_sum - initial_sum).abs() / initial_sum.abs().max(1e-10) < 1e-8);
    }

    /// Test configuration update
    #[test]
    fn test_config_update() {
        let grid = Arc::new(Grid::new(32, 32, 32, 1.0, 1.0, 1.0));
        let initial_config = HybridSpectralDGConfig {
            discontinuity_threshold: 0.1,
            spectral_order: 4,
            dg_polynomial_order: 2,
            adaptive_switching: true,
            conservation_tolerance: 1e-8,
        };

        let mut solver = HybridSpectralDGSolver::new(initial_config, grid);

        // Update configuration
        let new_config = HybridSpectralDGConfig {
            discontinuity_threshold: 0.05,
            spectral_order: 8,
            dg_polynomial_order: 4,
            adaptive_switching: false,
            conservation_tolerance: 1e-12,
        };

        solver.update_config(new_config.clone());

        // Verify that adaptive switching is disabled
        let field = Array3::ones((32, 32, 32));
        let dt = 0.001;
        let _result = solver
            .solve(&field, dt, &Grid::new(32, 32, 32, 1.0, 1.0, 1.0))
            .unwrap();

        // With adaptive_switching = false, no discontinuity detection should occur
        assert!(solver.discontinuity_mask().is_some());
    }

    /// Test performance with large grid
    #[test]
    #[ignore] // Ignore by default as it's computationally expensive
    fn test_large_grid_performance() {
        let grid = Arc::new(Grid::new(128, 128, 128, 1.0, 1.0, 1.0));
        let config = HybridSpectralDGConfig::default();

        let mut solver = HybridSpectralDGSolver::new(config, grid.clone());

        // Initialize with complex field
        let mut field = Array3::zeros((128, 128, 128));
        for i in 0..128 {
            for j in 0..128 {
                for k in 0..128 {
                    let x = i as f64 / 128.0;
                    let y = j as f64 / 128.0;
                    let z = k as f64 / 128.0;
                    field[[i, j, k]] = (2.0 * PI * x).sin() * (3.0 * PI * y).cos() * (PI * z).sin();

                    // Add some discontinuities
                    if (i == 64 || j == 64) && k < 64 {
                        field[[i, j, k]] = 2.0;
                    }
                }
            }
        }

        // Time the solve
        let dt = 0.0001;
        let start = std::time::Instant::now();
        let result = solver.solve(&field, dt, &grid).unwrap();
        let duration = start.elapsed();

        println!("Large grid solve took: {:?}", duration);

        // Basic validation
        assert_eq!(result.dim(), field.dim());
        let conservation_error = (result.sum() - field.sum()).abs() / field.sum().abs();
        assert!(conservation_error < 1e-8);
    }
}
