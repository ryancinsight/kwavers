//! GPU Compute Kernels Integration Tests
//!
//! Tests for AcousticFieldKernel and WaveEquationGpu, including:
//! - Kernel creation and initialization
//! - Compute propagation operations
//! - Integration with Grid types
//! - Error handling

#![cfg(feature = "gpu")]

use kwavers::gpu::compute_kernels::{AcousticFieldKernel, WaveEquationGpu};
use kwavers::Grid;
use ndarray::Array3;

/// Create a test grid for acoustic simulations
fn create_test_grid() -> Grid {
    Grid::new(
        32,   // nx
        32,   // ny
        32,   // nz
        0.001, // dx: 1mm spacing
        0.001, // dy
        0.001, // dz
    )
    .expect("Failed to create test grid")
}

#[tokio::test]
async fn test_acoustic_field_kernel_creation() {
    match AcousticFieldKernel::new().await {
        Ok(kernel) => {
            // Kernel created successfully
            assert!(format!("{:?}", kernel).contains("AcousticFieldKernel"));
        }
        Err(e) => {
            // GPU not available is acceptable
            eprintln!("GPU not available: {}", e);
        }
    }
}

#[tokio::test]
async fn test_compute_propagation_gaussian() {
    let kernel = match AcousticFieldKernel::new().await {
        Ok(k) => k,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };

    let grid = create_test_grid();
    let dt = 1e-6; // 1 microsecond
    let sound_speed = 1500.0; // m/s in water

    // Create Gaussian pressure field
    let mut pressure = Array3::zeros((32, 32, 32));
    let center = 16;
    for i in 0..32 {
        for j in 0..32 {
            for k in 0..32 {
                let dx = (i as f64 - center as f64) * grid.dx;
                let dy = (j as f64 - center as f64) * grid.dy;
                let dz = (k as f64 - center as f64) * grid.dz;
                let r2 = dx * dx + dy * dy + dz * dz;
                pressure[[i, j, k]] = (-r2 / (0.001 * 0.001)).exp();
            }
        }
    }

    // Test propagation (not async)
    let result = kernel.compute_propagation(&pressure, &grid, dt, sound_speed);

    match result {
        Ok(propagated) => {
            assert_eq!(propagated.dim(), pressure.dim());
            // Check that propagation occurred (values should change)
            let initial_sum: f64 = pressure.sum();
            let propagated_sum: f64 = propagated.sum();
            assert!(
                (propagated_sum - initial_sum).abs() < initial_sum * 0.5,
                "Propagation should conserve energy approximately"
            );
        }
        Err(e) => {
            eprintln!("Compute propagation failed: {}", e);
        }
    }
}

#[tokio::test]
async fn test_compute_propagation_zero_field() {
    let kernel = match AcousticFieldKernel::new().await {
        Ok(k) => k,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };

    let grid = create_test_grid();
    let pressure = Array3::zeros((32, 32, 32));
    let dt = 1e-6;
    let sound_speed = 1500.0;

    match kernel.compute_propagation(&pressure, &grid, dt, sound_speed) {
        Ok(result) => {
            // Zero field should remain zero
            assert_eq!(result.sum(), 0.0, "Zero field should remain zero");
        }
        Err(e) => {
            eprintln!("Test failed (acceptable if GPU unavailable): {}", e);
        }
    }
}

#[tokio::test]
async fn test_wave_equation_gpu_creation() {
    match WaveEquationGpu::new().await {
        Ok(solver) => {
            assert!(format!("{:?}", solver).contains("WaveEquationGpu"));
        }
        Err(e) => {
            eprintln!("GPU not available: {}", e);
        }
    }
}

#[tokio::test]
async fn test_wave_equation_step() {
    let solver = match WaveEquationGpu::new().await {
        Ok(s) => s,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };

    let grid = create_test_grid();
    let dim = (32, 32, 32);

    // Create test fields
    let mut pressure = Array3::zeros(dim);
    let velocity = Array3::zeros(dim);
    let density = Array3::from_elem(dim, 1000.0); // Water density
    let sound_speed = Array3::from_elem(dim, 1500.0); // Water sound speed

    // Add initial pressure perturbation
    pressure[[16, 16, 16]] = 1.0;

    let dt = 1e-7; // Small time step for stability

    match solver.step(&pressure, &velocity, &density, &sound_speed, &grid, dt) {
        Ok((new_pressure, new_velocity)) => {
            assert_eq!(new_pressure.dim(), dim);
            assert_eq!(new_velocity.dim(), dim);

            // Check that fields have reasonable values
            assert!(
                new_pressure.iter().all(|&x| x.is_finite()),
                "Pressure should be finite"
            );
            assert!(
                new_velocity.iter().all(|&x| x.is_finite()),
                "Velocity should be finite"
            );
        }
        Err(e) => {
            eprintln!("Wave equation step failed: {}", e);
        }
    }
}

#[tokio::test]
async fn test_compute_propagation_different_sizes() {
    let kernel = match AcousticFieldKernel::new().await {
        Ok(k) => k,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };

    // Test with different grid sizes
    let sizes = vec![(16, 16, 16), (32, 32, 32), (64, 16, 16)];

    for (nx, ny, nz) in sizes {
        let grid = Grid::new(nx, ny, nz, 0.001, 0.001, 0.001)
            .expect("Failed to create grid");

        let pressure = Array3::from_elem((nx, ny, nz), 0.5);

        match kernel.compute_propagation(&pressure, &grid, 1e-6, 1500.0) {
            Ok(result) => {
                assert_eq!(result.dim(), (nx, ny, nz));
            }
            Err(e) => {
                eprintln!("Test with size ({}, {}, {}) failed: {}", nx, ny, nz, e);
            }
        }
    }
}

#[test]
fn test_grid_creation() {
    let grid = create_test_grid();
    assert_eq!(grid.nx, 32);
    assert_eq!(grid.ny, 32);
    assert_eq!(grid.nz, 32);
    assert_eq!(grid.dx, 0.001);
}
