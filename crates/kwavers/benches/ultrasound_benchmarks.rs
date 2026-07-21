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
use kwavers_analysis::signal_processing::beamforming::utils::{
    SteeringVector, SteeringVectorMethod,
};
use kwavers_analysis::signal_processing::beamforming::MinimumVariance;
use kwavers_grid::Grid;
use kwavers_imaging::ultrasound::elastography::InversionMethod;
use kwavers_math::fft::Complex64;
use kwavers_medium::homogeneous::HomogeneousMedium;
use kwavers_simulation::imaging::elastography::ShearWaveElastography;
use kwavers_solver::forward::elastic::{ElasticWaveConfig, ElasticWaveField};
// Simple finite difference derivative for benchmarking
fn compute_derivative(field: &Array1<f64>, dx: f64, derivative: &mut Array1<f64>) {
    for i in 1..field.len() - 1 {
        derivative[i] = (field[i + 1] - field[i - 1]) / (2.0 * dx);
    }
    // Boundary conditions
    derivative[0] = (field[1] - field[0]) / dx;
    derivative[field.len() - 1] = (field[field.len() - 1] - field[field.len() - 2]) / dx;
}
use leto::{Array1, Array3};
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
                let x: Array1<f64> = Array1::from_shape_fn(grid_size, |[i]| {
                    0.0 + i as f64 * ((grid_size - 1) as f64 * dx - 0.0) / (grid_size - 1) as f64
                });

                // Initial conditions
                let mut u_current: Array1<f64> =
                    &(&(&(&x * 2.0) * PI) / wavelength).mapv(f64::sin) * amplitude;
                let mut u_previous: Array1<f64> = u_current.clone();

                // Time stepping (100 steps)
                for _ in 0..100 {
                    let mut u_next = Array1::<f64>::zeros(grid_size);

                    // Interior points
                    for i in 1..grid_size - 1 {
                        let u_xx =
                            (u_current[i + 1] - 2.0 * u_current[i] + u_current[i - 1]) / (dx * dx);
                        u_next[i] = 2.0 * u_current[i] - u_previous[i]
                            + (wave_speed * wave_speed * dt * dt) * u_xx;
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

/// Build a nonzero plane-wave snapshot for reconstruction benchmarks.
fn manufactured_shear_wave(grid_size: usize, spacing: f64) -> ElasticWaveField {
    let mut field = ElasticWaveField::new(grid_size, grid_size, grid_size);
    let cells_per_axis =
        u32::try_from(grid_size - 1).expect("benchmark grids fit within u32 dimensions");
    let domain_length = f64::from(cells_per_axis) * spacing;
    let wave_number = 2.0 * PI / domain_length;
    const DISPLACEMENT_AMPLITUDE: f64 = 1e-6;

    for i in 0..grid_size {
        for j in 0..grid_size {
            for k in 0..grid_size {
                let diagonal_index =
                    u32::try_from(i + j + k).expect("benchmark grid index sum fits within u32");
                let distance = f64::from(diagonal_index) * spacing;
                field.uz[[i, j, k]] = DISPLACEMENT_AMPLITUDE * (wave_number * distance).sin();
            }
        }
    }

    field
}

/// Benchmark SWE reconstruction performance.
///
/// Forward-wave generation has its canonical instrument in
/// `nl_swe_performance`; this instrument isolates inversion scaling.
fn bench_swe_reconstruction(c: &mut Criterion) {
    let mut group = c.benchmark_group("swe_reconstruction");

    for &grid_size in &[32, 64, 128] {
        const SPACING: f64 = 0.001;
        let grid = Grid::new(grid_size, grid_size, grid_size, SPACING, SPACING, SPACING).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let swe = ShearWaveElastography::new(
            &grid,
            &medium,
            InversionMethod::TimeOfFlight,
            ElasticWaveConfig::default(),
        )
        .unwrap();
        let displacement = [manufactured_shear_wave(grid_size, SPACING)];

        let reference = swe
            .reconstruct_elasticity(&displacement)
            .expect("manufactured shear wave must reconstruct");
        assert_eq!(reference.shear_wave_speed.shape(), [grid_size; 3]);
        assert!(
            reference
                .shear_wave_speed
                .iter()
                .all(|speed| speed.is_finite() && *speed > 0.0),
            "reconstructed shear-wave speed must be finite and positive"
        );

        group.bench_function(format!("swe_{}", grid_size), |b| {
            b.iter(|| {
                let elasticity = swe
                    .reconstruct_elasticity(black_box(&displacement))
                    .expect("manufactured shear wave must reconstruct");

                black_box(elasticity);
            })
        });
    }

    group.finish();
}

/// Benchmark memory usage scaling
fn bench_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling");

    group.bench_function("field_allocation_and_sum", |b| {
        b.iter(|| {
            // Measure representative field allocations across grid scales.
            let grid_sizes = [64, 128, 256];

            for &size in &grid_sizes {
                let pressure_field: Array3<f64> = Array3::zeros((size, size, size));
                let velocity_field: Array3<f64> = Array3::zeros((size, size, size));
                let displacement_field: Array3<f64> = Array3::zeros((size, size, size));

                // Measure a complete element-wise three-field sum.
                let mut result = Array3::<f64>::zeros((size, size, size));
                for i in 0..size {
                    for j in 0..size {
                        for k in 0..size {
                            result[[i, j, k]] = pressure_field[[i, j, k]]
                                + velocity_field[[i, j, k]]
                                + displacement_field[[i, j, k]];
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
                let x: Array1<f64> = Array1::from_shape_fn(grid_size, |[i]| {
                    0.0 + i as f64 * (10.0 - 0.0) / (grid_size - 1) as f64
                });
                let field: Array1<f64> = (&x * &x).mapv(f64::sin); // sin(x²)

                // Analytical derivative: 2x * cos(x²)
                let analytical_derivative: Array1<f64> = &(&x * 2.0) * &(&x * &x).mapv(f64::cos);

                // Numerical derivative
                let mut numerical_derivative = Array1::<f64>::zeros(grid_size);
                let dx = x[1] - x[0];

                compute_derivative(&field, dx, &mut numerical_derivative);

                // Compute error
                let error: Array1<f64> = &numerical_derivative - &analytical_derivative;
                let rms_error =
                    (error.iter().map(|x| x * x).sum::<f64>() / grid_size as f64).sqrt();

                black_box(rms_error);
            })
        });
    }

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
                let analytical = |x: f64| {
                    (2.0 * PI * frequency * final_time).sin() * (2.0 * PI / wavelength * x).sin()
                };

                // Numerical solution
                let x: Array1<f64> = Array1::from_shape_fn(size, |[i]| {
                    0.0 + i as f64 * (wavelength - 0.0) / (size - 1) as f64
                });
                let mut u_current: Array1<f64> = (&(&(&x * 2.0) * PI) / wavelength).mapv(f64::sin);
                let mut u_previous: Array1<f64> = u_current.clone();

                let steps = (final_time / dt) as usize;
                for _ in 0..steps {
                    let mut u_next = Array1::<f64>::zeros(size);

                    for i in 1..size - 1 {
                        let u_xx =
                            (u_current[i + 1] - 2.0 * u_current[i] + u_current[i - 1]) / (dx * dx);
                        u_next[i] = 2.0 * u_current[i] - u_previous[i]
                            + (wave_speed * wave_speed * dt * dt) * u_xx;
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

/// Benchmark the 8-element MVDR solve without coupling a wall-clock bound to
/// correctness tests. Criterion owns statistical timing and reports the
/// distribution under the selected execution environment.
fn bench_mvdr_weights(c: &mut Criterion) {
    let num_sensors = 8;
    let sensor_positions: Vec<[f64; 3]> = (0..num_sensors)
        .map(|i| [i as f64 * 0.001, 0.0, 0.0])
        .collect();
    let mut covariance = leto::Array2::<Complex64>::zeros((num_sensors, num_sensors));
    for i in 0..num_sensors {
        covariance[[i, i]] = Complex64::new(1.0, 0.0);
    }
    let steering_vector = SteeringVector::compute(
        &SteeringVectorMethod::PlaneWave,
        [0.0, 0.0, 1.0],
        1e6,
        &sensor_positions,
        1500.0,
    )
    .expect("MVDR steering vector");
    let steering_vector = steering_vector.mapv(|c| Complex64::new(c.re, c.im));
    let mvdr = MinimumVariance::with_diagonal_loading(0.01);

    c.bench_function("mvdr_weights_8", |b| {
        b.iter(|| {
            let weights = mvdr
                .compute_weights(black_box(&covariance), black_box(&steering_vector))
                .expect("MVDR weights");
            black_box(weights);
        });
    });
}

criterion_group!(
    benches,
    bench_1d_wave_equation,
    bench_swe_reconstruction,
    bench_memory_scaling,
    bench_derivative_computation,
    bench_physics_validation,
    bench_mvdr_weights
);
criterion_main!(benches);
