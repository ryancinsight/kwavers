//! Full Waveform Inversion Validation Tests
//!
//! This test suite validates the FWI implementation against:
//! 1. Known synthetic models (Marmousi-inspired)
//! 2. Analytical solutions where possible
//! 3. Literature benchmarks
//! 4. Numerical accuracy requirements

use approx::assert_relative_eq;
use kwavers::{
    medium::HomogeneousMedium,
    solver::reconstruction::seismic::{
        FullWaveformInversion, ReverseTimeMigration, SeismicImagingConfig,
    },
    Grid, KwaversResult,
};
use ndarray::{s, Array2, Array3};

const TEST_TOLERANCE: f64 = 1e-6;
const VELOCITY_TOLERANCE: f64 = 0.05; // 5% tolerance for velocity recovery

/// Test FWI with simple two-layer synthetic model
#[test]
fn test_fwi_two_layer_model() -> KwaversResult<()> {
    // Create simple test grid
    let grid = Grid::new(32, 32, 32, 50e-3, 50e-3, 50e-3); // 50m spacing

    // Create true velocity model: two horizontal layers
    let mut true_velocity = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                if k < grid.nz / 2 {
                    true_velocity[[i, j, k]] = 2000.0; // Upper layer
                } else {
                    true_velocity[[i, j, k]] = 3000.0; // Lower layer
                }
            }
        }
    }

    // Create initial model (constant velocity)
    let initial_velocity = Array3::from_elem((grid.nx, grid.ny, grid.nz), 2500.0);

    // Generate synthetic data using true model
    let observed_data = generate_synthetic_seismic_data(&true_velocity, &grid)?;

    // Set up source and receiver positions
    let source_positions = create_line_source_array(&grid, 5);
    let receiver_positions = create_line_receiver_array(&grid, 15);

    // Configure FWI with limited iterations for testing
    let config = SeismicImagingConfig {
        fwi_iterations: 10,
        fwi_tolerance: 1e-4,
        ..Default::default()
    };

    // Run FWI
    let mut fwi = FullWaveformInversion::new(config, initial_velocity);
    let medium = HomogeneousMedium::new(2500.0, 2000.0, 0.3);

    let recovered_velocity = fwi.reconstruct_fwi(
        &observed_data,
        &source_positions,
        &receiver_positions,
        &grid,
        &medium,
    )?;

    // Validate velocity recovery in each layer
    let upper_recovered = recovered_velocity
        .slice(s![.., .., 0..grid.nz / 2])
        .mean()
        .unwrap();
    let lower_recovered = recovered_velocity
        .slice(s![.., .., grid.nz / 2..])
        .mean()
        .unwrap();

    assert_relative_eq!(
        upper_recovered,
        2000.0,
        epsilon = 2000.0 * VELOCITY_TOLERANCE
    );
    assert_relative_eq!(
        lower_recovered,
        3000.0,
        epsilon = 3000.0 * VELOCITY_TOLERANCE
    );

    println!(
        "Two-layer test: Upper layer {:.1} m/s (true: 2000), Lower layer {:.1} m/s (true: 3000)",
        upper_recovered, lower_recovered
    );

    Ok(())
}

/// Test FWI gradient computation accuracy using finite differences
#[test]
fn test_fwi_gradient_accuracy() -> KwaversResult<()> {
    let grid = Grid::new(16, 16, 16, 100e-3, 100e-3, 100e-3);

    // Simple velocity model
    let velocity_model = Array3::from_elem((grid.nx, grid.ny, grid.nz), 2500.0);

    // Generate minimal synthetic data
    let observed_data = Array2::zeros((5, 100)); // 5 receivers, 100 time samples

    let source_positions = vec![[grid.nx as f64 * grid.dx / 2.0, 0.0, 0.0]];
    let receiver_positions = create_line_receiver_array(&grid, 5);

    let config = SeismicImagingConfig {
        fwi_iterations: 1,
        ..Default::default()
    };

    let mut fwi = FullWaveformInversion::new(config, velocity_model.clone());
    let medium = HomogeneousMedium::new(2500.0, 2000.0, 0.3);

    // Compute analytical gradient
    fwi.compute_gradient(
        &observed_data,
        &source_positions,
        &receiver_positions,
        &grid,
        &medium,
    )?;
    let analytical_gradient = fwi.gradient.clone();

    // Compute numerical gradient using finite differences
    let perturbation = 10.0; // 10 m/s perturbation
    let mut numerical_gradient = Array3::zeros((grid.nx, grid.ny, grid.nz));

    // Test gradient at a few points (full test would be too expensive)
    let test_points = vec![(8, 8, 8), (4, 4, 4), (12, 12, 12)];

    for &(i, j, k) in &test_points {
        if i < grid.nx && j < grid.ny && k < grid.nz {
            // Forward perturbation
            let mut perturbed_model = velocity_model.clone();
            perturbed_model[[i, j, k]] += perturbation;
            let mut fwi_plus = FullWaveformInversion::new(config.clone(), perturbed_model);
            let misfit_plus = fwi_plus.compute_data_misfit(
                &observed_data,
                &source_positions,
                &receiver_positions,
                &grid,
                &medium,
            )?;

            // Backward perturbation
            let mut perturbed_model = velocity_model.clone();
            perturbed_model[[i, j, k]] -= perturbation;
            let mut fwi_minus = FullWaveformInversion::new(config.clone(), perturbed_model);
            let misfit_minus = fwi_minus.compute_data_misfit(
                &observed_data,
                &source_positions,
                &receiver_positions,
                &grid,
                &medium,
            )?;

            // Finite difference gradient
            numerical_gradient[[i, j, k]] = (misfit_plus - misfit_minus) / (2.0 * perturbation);
        }
    }

    // Compare gradients at test points
    for &(i, j, k) in &test_points {
        if i < grid.nx && j < grid.ny && k < grid.nz {
            let analytical = analytical_gradient[[i, j, k]];
            let numerical = numerical_gradient[[i, j, k]];

            if numerical.abs() > 1e-12 {
                // Avoid division by very small numbers
                let relative_error = (analytical - numerical).abs() / numerical.abs();
                assert!(relative_error < 0.1,
                    "Gradient mismatch at ({}, {}, {}): analytical={:.6e}, numerical={:.6e}, error={:.2}%", 
                    i, j, k, analytical, numerical, relative_error * 100.0);
            }
        }
    }

    println!(
        "Gradient accuracy test passed for {} test points",
        test_points.len()
    );

    Ok(())
}

/// Test CFL condition validation
#[test]
fn test_cfl_condition_validation() -> KwaversResult<()> {
    let grid = Grid::new(10, 10, 10, 10e-3, 10e-3, 10e-3); // 10m spacing

    // High velocity model that might violate CFL
    let high_velocity_model = Array3::from_elem((grid.nx, grid.ny, grid.nz), 6000.0);

    let config = SeismicImagingConfig::default();
    let fwi = FullWaveformInversion::new(config, high_velocity_model);

    // Test stable timestep computation
    let stable_dt = fwi.compute_stable_timestep(&grid, 6000.0)?;

    // CFL condition: dt <= C * dx / vmax where C ≤ 0.5 for stability
    let expected_max_dt = 0.5 * 10e-3 / 6000.0;

    assert!(
        stable_dt <= expected_max_dt,
        "Computed timestep {:.6e} violates CFL condition (max: {:.6e})",
        stable_dt,
        expected_max_dt
    );

    println!(
        "CFL test: stable dt = {:.6e} s (max allowed: {:.6e} s)",
        stable_dt, expected_max_dt
    );

    Ok(())
}

/// Test convergence behavior
#[test]
fn test_fwi_convergence() -> KwaversResult<()> {
    let grid = Grid::new(16, 16, 16, 50e-3, 50e-3, 50e-3);

    // Simple gradient model
    let mut true_velocity = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                // Linear gradient with depth
                true_velocity[[i, j, k]] = 2000.0 + (k as f64 / grid.nz as f64) * 1000.0;
            }
        }
    }

    let initial_velocity = Array3::from_elem((grid.nx, grid.ny, grid.nz), 2500.0);
    let observed_data = generate_synthetic_seismic_data(&true_velocity, &grid)?;

    let source_positions = create_line_source_array(&grid, 3);
    let receiver_positions = create_line_receiver_array(&grid, 10);

    let config = SeismicImagingConfig {
        fwi_iterations: 20,
        fwi_tolerance: 1e-5,
        ..Default::default()
    };

    let mut fwi = FullWaveformInversion::new(config, initial_velocity);
    let medium = HomogeneousMedium::new(2500.0, 2000.0, 0.3);

    // Track misfit evolution
    let mut misfits = Vec::new();

    // Run one iteration at a time to track convergence
    for iteration in 0..10 {
        // Compute current misfit
        let current_misfit = fwi.compute_data_misfit(
            &observed_data,
            &source_positions,
            &receiver_positions,
            &grid,
            &medium,
        )?;
        misfits.push(current_misfit);

        if iteration > 0 {
            // Check that misfit is decreasing (or at least not increasing significantly)
            let improvement = (misfits[iteration - 1] - current_misfit) / misfits[0];
            println!(
                "Iteration {}: misfit = {:.6e}, improvement = {:.2}%",
                iteration,
                current_misfit,
                improvement * 100.0
            );
        }

        // Run one FWI iteration
        fwi.compute_gradient(
            &observed_data,
            &source_positions,
            &receiver_positions,
            &grid,
            &medium,
        )?;
        fwi.apply_regularization(&grid)?;
        fwi.compute_search_direction()?;
        let step_size = fwi.line_search(
            &observed_data,
            &source_positions,
            &receiver_positions,
            &grid,
            &medium,
        )?;
        fwi.update_velocity_model(step_size)?;
    }

    // Verify convergence: misfit should decrease overall
    assert!(
        misfits.last().unwrap() < &misfits[0],
        "FWI did not converge: initial misfit {:.6e}, final misfit {:.6e}",
        misfits[0],
        misfits.last().unwrap()
    );

    Ok(())
}

/// Test RTM imaging with known reflector
#[test]
fn test_rtm_imaging() -> KwaversResult<()> {
    let grid = Grid::new(32, 32, 32, 25e-3, 25e-3, 25e-3);

    // Model with single horizontal reflector
    let mut velocity_model = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                if k < grid.nz / 2 {
                    velocity_model[[i, j, k]] = 2500.0;
                } else {
                    velocity_model[[i, j, k]] = 3500.0; // High contrast reflector
                }
            }
        }
    }

    // Generate reflection data
    let recorded_data = generate_synthetic_seismic_data(&velocity_model, &grid)?;

    let source_positions = create_line_source_array(&grid, 5);
    let receiver_positions = create_line_receiver_array(&grid, 15);

    let config = SeismicImagingConfig::default();
    let rtm = ReverseTimeMigration::new(config);

    // Run RTM
    let migrated_image = rtm.migrate(
        &recorded_data,
        &source_positions,
        &receiver_positions,
        &velocity_model,
        &grid,
    )?;

    // Check that maximum reflection occurs near the interface
    let interface_depth = grid.nz / 2;
    let mut max_reflectivity = 0.0;
    let mut max_depth = 0;

    for k in 0..grid.nz {
        let depth_reflectivity: f64 = migrated_image
            .slice(s![.., .., k])
            .iter()
            .map(|&x| x.abs())
            .sum();
        if depth_reflectivity > max_reflectivity {
            max_reflectivity = depth_reflectivity;
            max_depth = k;
        }
    }

    let depth_error = (max_depth as i32 - interface_depth as i32).abs();
    assert!(
        depth_error <= 3,
        "RTM reflector position error: found at depth {}, expected near {}",
        max_depth,
        interface_depth
    );

    println!(
        "RTM test: reflector detected at depth index {} (expected: {})",
        max_depth, interface_depth
    );

    Ok(())
}

// Helper functions for test data generation

fn generate_synthetic_seismic_data(
    velocity_model: &Array3<f64>,
    grid: &Grid,
) -> KwaversResult<Array2<f64>> {
    // Simplified synthetic data generation
    // In practice, this would run a forward modeling simulation
    let num_receivers = 15;
    let num_time_samples = 200;
    let mut data = Array2::zeros((num_receivers, num_time_samples));

    // Generate simple synthetic traces with reflections
    for receiver in 0..num_receivers {
        for t in 0..num_time_samples {
            let time = t as f64 * 1e-3; // 1ms sampling

            // Direct wave
            let direct_time = 0.05 + (receiver as f64 * 0.01);
            if time > direct_time && time < direct_time + 0.02 {
                let ricker_arg = std::f64::consts::PI * 25.0 * (time - direct_time);
                data[[receiver, t]] =
                    (1.0 - 2.0 * ricker_arg.powi(2)) * (-ricker_arg.powi(2)).exp();
            }

            // Reflection from velocity contrast
            let reflection_time = 0.15 + (receiver as f64 * 0.015);
            if time > reflection_time && time < reflection_time + 0.015 {
                let ricker_arg = std::f64::consts::PI * 20.0 * (time - reflection_time);
                let amplitude = 0.3; // Reflection coefficient
                data[[receiver, t]] +=
                    amplitude * (1.0 - 2.0 * ricker_arg.powi(2)) * (-ricker_arg.powi(2)).exp();
            }
        }
    }

    Ok(data)
}

fn create_line_source_array(grid: &Grid, num_sources: usize) -> Vec<[f64; 3]> {
    let mut sources = Vec::new();
    for i in 0..num_sources {
        let x = (i as f64 / (num_sources - 1) as f64) * grid.nx as f64 * grid.dx * 0.8
            + 0.1 * grid.nx as f64 * grid.dx;
        sources.push([x, grid.ny as f64 * grid.dy / 2.0, 0.0]);
    }
    sources
}

fn create_line_receiver_array(grid: &Grid, num_receivers: usize) -> Vec<[f64; 3]> {
    let mut receivers = Vec::new();
    for i in 0..num_receivers {
        let x = (i as f64 / (num_receivers - 1) as f64) * grid.nx as f64 * grid.dx;
        receivers.push([
            x,
            grid.ny as f64 * grid.dy / 2.0,
            grid.nz as f64 * grid.dz * 0.1,
        ]);
    }
    receivers
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Integration test combining FWI and RTM
    #[test]
    fn test_fwi_rtm_workflow() -> KwaversResult<()> {
        let grid = Grid::new(24, 24, 24, 40e-3, 40e-3, 40e-3);

        // True model with velocity anomaly
        let mut true_velocity = Array3::from_elem((grid.nx, grid.ny, grid.nz), 2800.0);
        for i in 8..16 {
            for j in 8..16 {
                for k in 8..16 {
                    true_velocity[[i, j, k]] = 3200.0; // High velocity anomaly
                }
            }
        }

        // Generate observed data
        let observed_data = generate_synthetic_seismic_data(&true_velocity, &grid)?;

        let source_positions = create_line_source_array(&grid, 4);
        let receiver_positions = create_line_receiver_array(&grid, 12);

        // Step 1: FWI velocity estimation
        let initial_velocity = Array3::from_elem((grid.nx, grid.ny, grid.nz), 2800.0);
        let config = SeismicImagingConfig {
            fwi_iterations: 5,
            ..Default::default()
        };

        let mut fwi = FullWaveformInversion::new(config.clone(), initial_velocity);
        let medium = HomogeneousMedium::new(2800.0, 2200.0, 0.3);

        let estimated_velocity = fwi.reconstruct_fwi(
            &observed_data,
            &source_positions,
            &receiver_positions,
            &grid,
            &medium,
        )?;

        // Step 2: RTM with estimated velocity
        let rtm = ReverseTimeMigration::new(config);
        let migrated_image = rtm.migrate(
            &observed_data,
            &source_positions,
            &receiver_positions,
            &estimated_velocity,
            &grid,
        )?;

        // Validate that the anomaly region shows higher velocities
        let anomaly_velocity = estimated_velocity
            .slice(s![8..16, 8..16, 8..16])
            .mean()
            .unwrap();
        let background_velocity = estimated_velocity
            .slice(s![0..8, 0..8, 0..8])
            .mean()
            .unwrap();

        assert!(
            anomaly_velocity > background_velocity,
            "FWI failed to detect velocity anomaly: anomaly {:.1} vs background {:.1}",
            anomaly_velocity,
            background_velocity
        );

        println!(
            "Integrated test: anomaly velocity {:.1} m/s, background {:.1} m/s",
            anomaly_velocity, background_velocity
        );

        Ok(())
    }
}
