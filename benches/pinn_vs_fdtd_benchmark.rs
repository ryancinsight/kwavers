//! Performance benchmarks comparing PINN vs FDTD methods for 2D wave equations
//!
//! This benchmark suite compares the performance and accuracy of:
//! - Physics-Informed Neural Networks (PINN) using Burn framework
//! - Finite-Difference Time Domain (FDTD) method
//!
//! The benchmarks test different problem sizes and complexities to understand
//! the trade-offs between the two approaches.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kwavers::grid::Grid;
use kwavers::medium::homogeneous::HomogeneousMedium;
use kwavers::medium::ArrayAccess;
use kwavers::solver::fdtd::{FdtdConfig, FdtdSolver};
use ndarray::{Array1, Array2};

#[cfg(feature = "pinn")]
use burn::backend::NdArray;
#[cfg(feature = "pinn")]
use kwavers::ml::pinn::burn_wave_equation_2d::{
    BurnPINN2DConfig, BurnPINN2DWave, Geometry2D, BurnLossWeights2D,
};

/// Benchmark configuration for different problem sizes
struct BenchmarkConfig {
    /// Grid size for FDTD (nx, ny, nz)
    grid_size: (usize, usize, usize),
    /// Spatial step size (m)
    dx: f64,
    /// Time step size (s)
    dt: f64,
    /// Total simulation time (s)
    total_time: f64,
    /// Number of collocation points for PINN
    num_collocation: usize,
    /// PINN training epochs
    pinn_epochs: usize,
}

impl BenchmarkConfig {
    /// Small problem for quick benchmarking
    fn small() -> Self {
        Self {
            grid_size: (32, 32, 1), // 2D slice
            dx: 1e-3,
            dt: 1e-6,
            total_time: 1e-4,
            num_collocation: 1000,
            pinn_epochs: 50,
        }
    }

    /// Medium problem
    fn medium() -> Self {
        Self {
            grid_size: (64, 64, 1),
            dx: 1e-3,
            dt: 1e-6,
            total_time: 1e-4,
            num_collocation: 5000,
            pinn_epochs: 100,
        }
    }

    /// Large problem
    fn large() -> Self {
        Self {
            grid_size: (128, 128, 1),
            dx: 1e-3,
            dt: 1e-6,
            total_time: 1e-4,
            num_collocation: 10000,
            pinn_epochs: 200,
        }
    }
}

/// Generate analytical solution for 2D wave equation
/// u(x,y,t) = sin(πx) * sin(πy) * cos(π√2 * c * t)
fn analytical_solution_2d(x: f64, y: f64, t: f64, wave_speed: f64) -> f64 {
    let k = std::f64::consts::PI * 2.0_f64.sqrt();
    (x * std::f64::consts::PI).sin() * (y * std::f64::consts::PI).sin() * (k * wave_speed * t).cos()
}

/// Generate initial condition data for PINN training
fn generate_training_data(
    config: &BenchmarkConfig,
    wave_speed: f64,
    n_data_points: usize,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array2<f64>) {
    let mut x_data = Vec::with_capacity(n_data_points);
    let mut y_data = Vec::with_capacity(n_data_points);
    let mut t_data = Vec::with_capacity(n_data_points);
    let mut u_data = Vec::with_capacity(n_data_points);

    let (nx, ny, _) = config.grid_size;
    let lx = (nx - 1) as f64 * config.dx;
    let ly = (ny - 1) as f64 * config.dx;

    for _i in 0..n_data_points {
        // Sample points within domain
        let x = rand::random::<f64>() * lx;
        let y = rand::random::<f64>() * ly;
        let t = rand::random::<f64>() * config.total_time;

        let u = analytical_solution_2d(x, y, t, wave_speed);

        x_data.push(x);
        y_data.push(y);
        t_data.push(t);
        u_data.push(u);
    }

    (
        Array1::from_vec(x_data),
        Array1::from_vec(y_data),
        Array1::from_vec(t_data),
        Array2::from_shape_vec((n_data_points, 1), u_data).unwrap(),
    )
}

/// FDTD benchmark for 2D wave equation
fn fdtd_2d_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("fdtd_2d_wave");

    let wave_speed = 343.0; // m/s (speed of sound in air)
    let configs = vec![
        ("small", BenchmarkConfig::small()),
        ("medium", BenchmarkConfig::medium()),
        ("large", BenchmarkConfig::large()),
    ];

    for (name, config) in configs {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_setup", name)),
            &config,
            |b, config| {
                b.iter(|| {
                    // Create grid and medium
                    let grid = Grid::new(
                        config.grid_size.0,
                        config.grid_size.1,
                        config.grid_size.2,
                        config.dx,
                        config.dx,
                        config.dx,
                    ).expect("Grid creation failed");

                    let medium = HomogeneousMedium::new(1.225, wave_speed, 0.0, 1.0, &grid);

                    // Create FDTD solver
                    let fdtd_config = FdtdConfig {
                        spatial_order: 4,
                        cfl_factor: 0.95,
                        ..Default::default()
                    };
                    let solver = FdtdSolver::new(fdtd_config, &grid).expect("FDTD solver creation failed");

                    black_box((grid, medium, solver))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_simulation", name)),
            &config,
            |b, config| {
                b.iter(|| {
                    // Setup
                    let grid = Grid::new(
                        config.grid_size.0,
                        config.grid_size.1,
                        config.grid_size.2,
                        config.dx,
                        config.dx,
                        config.dx,
                    ).expect("Grid creation failed");

                    let medium = HomogeneousMedium::new(1.225, wave_speed, 0.0, 1.0, &grid);

                    let fdtd_config = FdtdConfig {
                        spatial_order: 4,
                        cfl_factor: 0.95,
                        ..Default::default()
                    };
                    let mut solver = FdtdSolver::new(fdtd_config, &grid).expect("FDTD solver creation failed");

                    // Initialize fields
                    let mut pressure = grid.create_field();
                    let mut vx = grid.create_field();
                    let mut vy = grid.create_field();
                    let mut vz = grid.create_field();

                    // Add initial perturbation (Gaussian pulse)
                    let center_x = (grid.nx / 2) as f64 * grid.dx;
                    let center_y = (grid.ny / 2) as f64 * grid.dy;
                    let sigma = 0.01; // 1 cm width

                    for i in 0..grid.nx {
                        for j in 0..grid.ny {
                            for k in 0..grid.nz {
                                let x = i as f64 * grid.dx;
                                let y = j as f64 * grid.dy;
                                let r2 = (x - center_x).powi(2) + (y - center_y).powi(2);
                                pressure[[i, j, k]] = (-r2 / (2.0 * sigma * sigma)).exp();
                            }
                        }
                    }

                    // Time stepping
                    let n_steps = (config.total_time / config.dt) as usize;
                    let dt = config.dt;

                    for _step in 0..n_steps.min(1000) { // Limit steps for benchmarking
                        // Update velocity from pressure
                        solver.update_velocity(
                            &mut vx,
                            &mut vy,
                            &mut vz,
                            &pressure,
                            medium.density_array().view(),
                            dt,
                        ).expect("Velocity update failed");

                        // Update pressure from velocity
                        solver.update_pressure(
                            &mut pressure,
                            &vx,
                            &vy,
                            &vz,
                            medium.density_array().view(),
                            medium.sound_speed_array().view(),
                            dt,
                        ).expect("Pressure update failed");
                    }

                    black_box(pressure)
                });
            },
        );
    }

    group.finish();
}

/// PINN benchmark for 2D wave equation
#[cfg(feature = "pinn")]
fn pinn_2d_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pinn_2d_wave");

    let wave_speed = 343.0;
    let configs = vec![
        ("small", BenchmarkConfig::small()),
        ("medium", BenchmarkConfig::medium()),
        ("large", BenchmarkConfig::large()),
    ];

    for (name, config) in configs {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_setup", name)),
            &config,
            |b, config| {
                b.iter(|| {
                    type Backend = burn::backend::Autodiff<NdArray<f32>>;
                    let device = Default::default();

                    let pinn_config = BurnPINN2DConfig {
                        hidden_layers: vec![50, 50, 50],
                        num_collocation_points: config.num_collocation,
                        learning_rate: 1e-3,
                        loss_weights: BurnLossWeights2D {
                            data: 1.0,
                            pde: 1.0,
                            boundary: 10.0,
                            initial: 10.0,
                        },
                        ..Default::default()
                    };

                    let geometry = Geometry2D::rectangular(
                        0.0,
                        (config.grid_size.0 - 1) as f64 * config.dx,
                        0.0,
                        (config.grid_size.1 - 1) as f64 * config.dx,
                    );

                    let trainer = BurnPINN2DWave::<Backend>::new_trainer(pinn_config, geometry, &device)
                        .expect("PINN trainer creation failed");

                    black_box(trainer)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_training", name)),
            &config,
            |b, config| {
                b.iter(|| {
                    type Backend = burn::backend::Autodiff<NdArray<f32>>;
                    let device = Default::default();

                    let pinn_config = BurnPINN2DConfig {
                        hidden_layers: vec![50, 50, 50],
                        num_collocation_points: config.num_collocation,
                        learning_rate: 1e-3,
                        loss_weights: BurnLossWeights2D {
                            data: 1.0,
                            pde: 1.0,
                            boundary: 10.0,
                            initial: 10.0,
                        },
                        ..Default::default()
                    };

                    let geometry = Geometry2D::rectangular(
                        0.0,
                        (config.grid_size.0 - 1) as f64 * config.dx,
                        0.0,
                        (config.grid_size.1 - 1) as f64 * config.dx,
                    );

                    let mut trainer = BurnPINN2DWave::<Backend>::new_trainer(pinn_config, geometry, &device)
                        .expect("PINN trainer creation failed");

                    // Generate training data
                    let (x_data, y_data, t_data, u_data) =
                        generate_training_data(config, wave_speed, 100);

                    // Train PINN
                    let _metrics = trainer.train(
                        &x_data,
                        &y_data,
                        &t_data,
                        &u_data,
                        wave_speed,
                        &device,
                        config.pinn_epochs,
                    ).expect("PINN training failed");

                    black_box(trainer)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_prediction", name)),
            &config,
            |b, config| {
                b.iter(|| {
                    type Backend = burn::backend::Autodiff<NdArray<f32>>;
                    let device = Default::default();

                    let pinn_config = BurnPINN2DConfig {
                        hidden_layers: vec![50, 50, 50],
                        num_collocation_points: config.num_collocation,
                        ..Default::default()
                    };

                    let geometry = Geometry2D::rectangular(
                        0.0,
                        (config.grid_size.0 - 1) as f64 * config.dx,
                        0.0,
                        (config.grid_size.1 - 1) as f64 * config.dx,
                    );

                    let trainer = BurnPINN2DWave::<Backend>::new_trainer(pinn_config, geometry, &device)
                        .expect("PINN trainer creation failed");

                    // Generate test points
                    let n_test = 1000;
                    let mut x_test = Vec::with_capacity(n_test);
                    let mut y_test = Vec::with_capacity(n_test);
                    let mut t_test = Vec::with_capacity(n_test);

                    let lx = (config.grid_size.0 - 1) as f64 * config.dx;
                    let ly = (config.grid_size.1 - 1) as f64 * config.dx;

                    for _ in 0..n_test {
                        x_test.push(rand::random::<f64>() * lx);
                        y_test.push(rand::random::<f64>() * ly);
                        t_test.push(rand::random::<f64>() * config.total_time);
                    }

                    let x_test = Array1::from_vec(x_test);
                    let y_test = Array1::from_vec(y_test);
                    let t_test = Array1::from_vec(t_test);

                    // Make predictions
                    let predictions = trainer.pinn().predict(
                        &x_test,
                        &y_test,
                        &t_test,
                        &device,
                    ).expect("PINN prediction failed");

                    black_box(predictions)
                });
            },
        );
    }

    group.finish();
}

/// Memory usage comparison benchmark
fn memory_usage_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_comparison");

    let _wave_speed = 343.0;
    let config = BenchmarkConfig::medium();

    // FDTD memory usage
    group.bench_function("fdtd_memory", |b| {
        b.iter(|| {
            let grid = Grid::new(
                config.grid_size.0,
                config.grid_size.1,
                config.grid_size.2,
                config.dx,
                config.dx,
                config.dx,
            ).expect("Grid creation failed");

            // FDTD requires storing multiple field arrays
            let pressure = grid.create_field();
            let vx = grid.create_field();
            let vy = grid.create_field();
            let vz = grid.create_field();

            black_box((pressure, vx, vy, vz))
        });
    });

    // PINN memory usage
    #[cfg(feature = "pinn")]
    group.bench_function("pinn_memory", |b| {
        b.iter(|| {
            type Backend = burn::backend::Autodiff<NdArray<f32>>;
            let device = Default::default();

            let pinn_config = BurnPINN2DConfig {
                hidden_layers: vec![100, 100, 100],
                num_collocation_points: config.num_collocation,
                ..Default::default()
            };

            let geometry = Geometry2D::rectangular(0.0, 1.0, 0.0, 1.0);
            let trainer = BurnPINN2DWave::<Backend>::new_trainer(pinn_config, geometry, &device)
                .expect("PINN trainer creation failed");

            black_box(trainer)
        });
    });

    group.finish();
}

/// Accuracy comparison benchmark
fn accuracy_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_comparison");

    let wave_speed = 343.0;
    let config = BenchmarkConfig::small();

    // FDTD accuracy
    group.bench_function("fdtd_accuracy", |b| {
        b.iter(|| {
            let grid = Grid::new(
                config.grid_size.0,
                config.grid_size.1,
                config.grid_size.2,
                config.dx,
                config.dx,
                config.dx,
            ).expect("Grid creation failed");

            let medium = HomogeneousMedium::new(1.225, wave_speed, 0.0, 1.0, &grid);

            let fdtd_config = FdtdConfig {
                spatial_order: 4,
                cfl_factor: 0.95,
                ..Default::default()
            };
            let mut solver = FdtdSolver::new(fdtd_config, &grid).expect("FDTD solver creation failed");

            // Initialize with analytical solution
            let mut pressure = grid.create_field();
            let mut vx = grid.create_field();
            let mut vy = grid.create_field();
            let mut vz = grid.create_field();

            let t = config.total_time * 0.5; // Mid-simulation time
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        pressure[[i, j, k]] = analytical_solution_2d(x, y, t, wave_speed);
                    }
                }
            }

            // Run simulation for a few steps
            let n_steps = 10;
            let dt = config.dt;

            for _step in 0..n_steps {
                solver.update_velocity(
                    &mut vx,
                    &mut vy,
                    &mut vz,
                    &pressure,
                    medium.density_array().view(),
                    dt,
                ).expect("Velocity update failed");

                solver.update_pressure(
                    &mut pressure,
                    &vx,
                    &vy,
                    &vz,
                    medium.density_array().view(),
                    medium.sound_speed_array().view(),
                    dt,
                ).expect("Pressure update failed");
            }

            // Compute error against analytical solution
            let mut max_error: f64 = 0.0;
            let t_final = t + n_steps as f64 * dt;
            for i in 1..grid.nx-1 { // Skip boundaries
                for j in 1..grid.ny-1 {
                    for k in 0..grid.nz {
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let analytical = analytical_solution_2d(x, y, t_final, wave_speed);
                        let error = (pressure[[i, j, k]] - analytical).abs();
                        max_error = max_error.max(error);
                    }
                }
            }

            black_box(max_error)
        });
    });

    // PINN accuracy
    #[cfg(feature = "pinn")]
    group.bench_function("pinn_accuracy", |b| {
        b.iter(|| {
            type Backend = burn::backend::Autodiff<NdArray<f32>>;
            let device = Default::default();

            let pinn_config = BurnPINN2DConfig {
                hidden_layers: vec![50, 50, 50],
                num_collocation_points: config.num_collocation,
                learning_rate: 1e-3,
                loss_weights: BurnLossWeights2D {
                    data: 1.0,
                    pde: 1.0,
                    boundary: 10.0,
                    initial: 10.0,
                },
                ..Default::default()
            };

            let geometry = Geometry2D::rectangular(
                0.0,
                (config.grid_size.0 - 1) as f64 * config.dx,
                0.0,
                (config.grid_size.1 - 1) as f64 * config.dx,
            );

            let mut trainer = BurnPINN2DWave::<Backend>::new_trainer(pinn_config, geometry, &device)
                .expect("PINN trainer creation failed");

            // Generate training data
            let (x_data, y_data, t_data, u_data) =
                generate_training_data(&config, wave_speed, 200);

            // Train PINN
            let _metrics = trainer.train(
                &x_data,
                &y_data,
                &t_data,
                &u_data,
                wave_speed,
                &device,
                config.pinn_epochs,
            ).expect("PINN training failed");

            // Generate test points
            let n_test = 100;
            let mut x_test = Vec::new();
            let mut y_test = Vec::new();
            let mut t_test = Vec::new();
            let mut u_analytical = Vec::new();

            let lx = (config.grid_size.0 - 1) as f64 * config.dx;
            let ly = (config.grid_size.1 - 1) as f64 * config.dx;

            for _ in 0..n_test {
                let x = rand::random::<f64>() * lx;
                let y = rand::random::<f64>() * ly;
                let t = rand::random::<f64>() * config.total_time;

                x_test.push(x);
                y_test.push(y);
                t_test.push(t);
                u_analytical.push(analytical_solution_2d(x, y, t, wave_speed));
            }

            let x_test = Array1::from_vec(x_test);
            let y_test = Array1::from_vec(y_test);
            let t_test = Array1::from_vec(t_test);

            // Make predictions
            let predictions = trainer.pinn().predict(
                &x_test,
                &y_test,
                &t_test,
                &device,
            ).expect("PINN prediction failed");

            // Compute error
            let mut max_error: f64 = 0.0;
            for i in 0..n_test {
                let predicted = predictions[[i, 0]] as f64;
                let analytical = u_analytical[i];
                let error = (predicted - analytical).abs();
                max_error = max_error.max(error);
            }

            black_box(max_error)
        });
    });

    group.finish();
}

#[cfg(feature = "pinn")]
criterion_group!(
    benches,
    fdtd_2d_benchmark,
    pinn_2d_benchmark,
    memory_usage_benchmark,
    accuracy_benchmark
);

#[cfg(not(feature = "pinn"))]
criterion_group!(
    benches,
    fdtd_2d_benchmark,
    memory_usage_benchmark,
    accuracy_benchmark
);

criterion_main!(benches);
