//! Ultrasound Physics Performance Benchmarks
//!
//! Comprehensive benchmarking suite comparing different ultrasound simulation
//! methods for accuracy, speed, and clinical relevance.
//!
//! ## Benchmark Categories
//!
//! - **FDTD vs Analytical**: Wave equation accuracy validation
//! - **SWE Performance**: Elastography reconstruction timing
//! - **Memory Efficiency**: Grid size scaling analysis
//! - **Clinical Workflows**: End-to-end simulation timing
//!
//! ## Performance Metrics
//!
//! - **Accuracy**: RMS error vs analytical solutions
//! - **Speed**: GFLOPS, time-to-solution
//! - **Memory**: Peak usage, bandwidth requirements
//! - **Scalability**: Weak/strong scaling with grid size

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers::error::KwaversResult;
use kwavers::grid::Grid;
use kwavers::medium::homogeneous::HomogeneousMedium;
use kwavers::physics::imaging::elastography::{InversionMethod, ShearWaveElastography};
use kwavers::solver::fdtd::finite_difference::compute_derivative;
use ndarray::{Array1, Array2, Array3};
use std::f64::consts::PI;

/// Benchmark 1D wave equation accuracy and performance
fn bench_1d_wave_equation(c: &mut Criterion) {
    let mut group = c.benchmark_group("1d_wave_equation");

    for &grid_size in &[100, 500, 1000, 5000] {
        group.bench_function(format!("fdtd_{}", grid_size), |b| {
            b.iter(|| {
                // Setup parameters
                let frequency = 1000.0; // Hz
                let wave_speed = 343.0; // m/s
                let amplitude = 1.0;
                let wavelength = wave_speed / frequency;
                let dx = wavelength / 20.0;
                let dt = dx / wave_speed * 0.9; // Slightly below CFL

                // Create spatial grid
                let x: Array1<f64> = Array1::linspace(0.0, (grid_size - 1) as f64 * dx, grid_size);

                // Initial conditions
                let mut u_current: Array1<f64> = (&x * 2.0 * PI / wavelength).mapv(f64::sin) * amplitude;
                let mut u_previous: Array1<f64> = u_current.clone();

                // Time stepping (100 steps)
                for _ in 0..100 {
                    let mut u_next = Array1::<f64>::zeros(grid_size);

                    // Interior points
                    for i in 1..grid_size - 1 {
                        let u_xx = (u_current[i + 1] - 2.0 * u_current[i] + u_current[i - 1]) / (dx * dx);
                        u_next[i] = 2.0 * u_current[i] - u_previous[i] + (wave_speed * wave_speed * dt * dt) * u_xx;
                    }

                    // Boundary conditions (absorbing)
                    u_next[0] = u_current[1];
                    u_next[grid_size - 1] = u_current[grid_size - 2];

                    u_previous = u_current;
                    u_current = u_next;
                }

                black_box(u_current);
            })
        });
    }

    group.finish();
}

/// Benchmark SWE reconstruction performance
fn bench_swe_reconstruction(c: &mut Criterion) {
    let mut group = c.benchmark_group("swe_reconstruction");

    for &grid_size in &[32, 64, 128] {
        let grid = Grid::new(grid_size, grid_size, grid_size, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        group.bench_function(format!("swe_{}", grid_size), |b| {
            b.iter(|| {
                let swe = ShearWaveElastography::new(&grid, &medium, InversionMethod::TimeOfFlight).unwrap();
                let push_location = [grid.dx * 10.0, grid.dy * 10.0, grid.dz * 10.0];
                let displacement = swe.generate_shear_wave(push_location).unwrap();
                let elasticity = swe.reconstruct_elasticity(&displacement).unwrap();

                black_box(elasticity);
            })
        });
    }

    group.finish();
}

/// Benchmark memory usage scaling
fn bench_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling");

    group.bench_function("array_allocation", |b| {
        b.iter(|| {
            // Simulate typical ultrasound grid allocations
            let grid_sizes = [64, 128, 256];

            for &size in &grid_sizes {
                let pressure_field: Array3<f64> = Array3::zeros((size, size, size));
                let velocity_field: Array3<f64> = Array3::zeros((size, size, size));
                let displacement_field: Array3<f64> = Array3::zeros((size, size, size));

                // Simulate computation
                let mut result = Array3::<f64>::zeros((size, size, size));
                for i in 0..size {
                    for j in 0..size {
                        for k in 0..size {
                            result[[i, j, k]] = pressure_field[[i, j, k]] +
                                               velocity_field[[i, j, k]] +
                                               displacement_field[[i, j, k]];
                        }
                    }
                }

                black_box(result);
            }
        })
    });

    group.finish();
}

/// Benchmark derivative computation performance
fn bench_derivative_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("derivative_computation");

    for &grid_size in &[100, 500, 1000, 5000] {
        group.bench_function(format!("finite_diff_{}", grid_size), |b| {
            b.iter(|| {
                // Create test field with known derivative
                let x: Array1<f64> = Array1::linspace(0.0, 10.0, grid_size);
                let field: Array1<f64> = (&x * &x).mapv(f64::sin); // sin(x²)

                // Analytical derivative: 2x * cos(x²)
                let analytical_derivative: Array1<f64> = (&x * 2.0) * (&x * &x).mapv(f64::cos);

                // Numerical derivative
                let mut numerical_derivative = Array1::<f64>::zeros(grid_size);
                let dx = x[1] - x[0];

                compute_derivative(&field, dx, &mut numerical_derivative);

                // Compute error
                let error: Array1<f64> = &numerical_derivative - &analytical_derivative;
                let rms_error = (error.iter().map(|x| x * x).sum::<f64>() / grid_size as f64).sqrt();

                black_box(rms_error);
            })
        });
    }

    group.finish();
}

/// Benchmark clinical workflow performance
fn bench_clinical_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("clinical_workflow");

    group.bench_function("liver_fibrosis_assessment", |b| {
        b.iter(|| {
            // Simulate complete clinical workflow
            let grid = Grid::new(100, 100, 80, 0.001, 0.001, 0.001).unwrap();
            let medium = HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid); // Liver properties

            // SWE workflow
            let swe = ShearWaveElastography::new(&grid, &medium, InversionMethod::TimeOfFlight).unwrap();
            let push_location = [0.025, 0.025, 0.015]; // 25mm lateral, 15mm depth
            let displacement = swe.generate_shear_wave(push_location).unwrap();

            // Simulate tracking (simplified)
            let displacement_magnitude = displacement.mapv(f64::abs);

            // Reconstruct elasticity
            let elasticity_map = swe.reconstruct_elasticity(&displacement_magnitude).unwrap();

            // Clinical analysis (simplified)
            let mean_stiffness = elasticity_map.youngs_modulus.mean().unwrap();
            let std_stiffness = {
                let variance = elasticity_map.youngs_modulus.iter()
                    .map(|&x| (x - mean_stiffness).powi(2))
                    .sum::<f64>() / elasticity_map.youngs_modulus.len() as f64;
                variance.sqrt()
            };

            // Fibrosis staging
            let fibrosis_stage = if mean_stiffness < 5_000.0 {
                "F0-F1"
            } else if mean_stiffness < 8_000.0 {
                "F2"
            } else if mean_stiffness < 12_000.0 {
                "F3"
            } else {
                "F4"
            };

            black_box((mean_stiffness, std_stiffness, fibrosis_stage));
        })
    });

    group.finish();
}

/// Benchmark ultrasound physics validation
fn bench_physics_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("physics_validation");

    group.bench_function("wave_equation_convergence", |b| {
        b.iter(|| {
            // Test convergence of wave equation solution
            let grid_sizes = [50, 100, 200];
            let mut errors = Vec::new();

            for &size in &grid_sizes {
                let frequency = 1000.0;
                let wave_speed = 343.0;
                let wavelength = wave_speed / frequency;
                let dx = wavelength / 20.0;
                let dt = dx / wave_speed * 0.9;

                // Analytical solution at final time
                let final_time = 0.001; // 1 ms
                let analytical = |x: f64| (2.0 * PI * frequency * final_time).sin() *
                                        (2.0 * PI / wavelength * x).sin();

                // Numerical solution
                let x: Array1<f64> = Array1::linspace(0.0, wavelength, size);
                let mut u_current: Array1<f64> = (&x * 2.0 * PI / wavelength).mapv(f64::sin);
                let mut u_previous: Array1<f64> = u_current.clone();

                let steps = (final_time / dt) as usize;
                for _ in 0..steps {
                    let mut u_next = Array1::<f64>::zeros(size);

                    for i in 1..size - 1 {
                        let u_xx = (u_current[i + 1] - 2.0 * u_current[i] + u_current[i - 1]) / (dx * dx);
                        u_next[i] = 2.0 * u_current[i] - u_previous[i] + (wave_speed * wave_speed * dt * dt) * u_xx;
                    }

                    u_previous = u_current;
                    u_current = u_next;
                }

                // Compute RMS error
                let mut error_sum = 0.0;
                for (i, &x_val) in x.iter().enumerate() {
                    let analytical_val = analytical(x_val);
                    let error = u_current[i] - analytical_val;
                    error_sum += error * error;
                }
                let rms_error = (error_sum / size as f64).sqrt();
                errors.push(rms_error);
            }

            // Check convergence rate (should be ~1/dx² for second-order method)
            let convergence_rate = if errors.len() >= 2 {
                (errors[errors.len() - 2] / errors[errors.len() - 1]).log2()
            } else {
                0.0
            };

            black_box((errors, convergence_rate));
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_1d_wave_equation,
    bench_swe_reconstruction,
    bench_memory_scaling,
    bench_derivative_computation,
    bench_clinical_workflow,
    bench_physics_validation
);
criterion_main!(benches);
