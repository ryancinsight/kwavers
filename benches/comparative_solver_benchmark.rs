//! Comparative Solver Benchmarks
//!
//! Performance and accuracy benchmarks comparing different numerical methods.
//! Runs in release mode to get accurate performance measurements.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use kwavers::domain::grid::Grid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::domain::source::GridSource;
use kwavers::solver::forward::fdtd::{FdtdConfig, FdtdSolver};
use kwavers::solver::forward::pstd::{PSTDConfig, PSTDSolver, PSTDSource};
use ndarray::Array3;

/// Benchmark configuration
#[derive(Debug, Clone)]
struct BenchmarkConfig {
    grid_sizes: Vec<(usize, usize, usize)>,
    time_steps: Vec<usize>,
    problems: Vec<String>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            grid_sizes: vec![
                (32, 32, 32),   // Small
                (64, 64, 64),   // Medium
                (128, 128, 32), // Large 2D-like
            ],
            time_steps: vec![10, 50, 100],
            problems: vec![
                "plane_wave".to_string(),
                "gaussian_pulse".to_string(),
                "spherical_wave".to_string(),
            ],
        }
    }
}

/// Benchmark results for analysis
#[derive(Debug)]
struct BenchmarkResult {
    solver_name: String,
    grid_size: (usize, usize, usize),
    time_steps: usize,
    execution_time: std::time::Duration,
    memory_usage: usize,
    final_energy: f64,
    stability_metric: f64,
}

/// Setup benchmark problems
fn setup_plane_wave_problem(grid_size: (usize, usize, usize)) -> (Grid, HomogeneousMedium, GridSource) {
    let (nx, ny, nz) = grid_size;
    let dx = 0.001; // 1mm
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();

    let medium = HomogeneousMedium::water(&grid);

    let mut source = GridSource::default();
    source.set_frequency(1e6); // 1 MHz
    source.set_amplitude(1e5); // 100 kPa

    (grid, medium, source)
}

/// Run FDTD benchmark
fn run_fdtd_benchmark(
    grid: &Grid,
    medium: &HomogeneousMedium,
    source: &GridSource,
    time_steps: usize,
) -> BenchmarkResult {
    let config = FdtdConfig::default();
    let mut solver = FdtdSolver::new(config, grid, medium, source.clone()).unwrap();

    let start = std::time::Instant::now();

    // Run simulation
    for _ in 0..time_steps {
        black_box(solver.step_forward().unwrap());
    }

    let execution_time = start.elapsed();

    // Calculate metrics
    let final_field = solver.pressure();
    let memory_usage = final_field.len() * std::mem::size_of::<f64>();
    let final_energy = calculate_energy(final_field);
    let stability_metric = calculate_stability(final_field);

    BenchmarkResult {
        solver_name: "FDTD".to_string(),
        grid_size: (grid.nx, grid.ny, grid.nz),
        time_steps,
        execution_time,
        memory_usage,
        final_energy,
        stability_metric,
    }
}

/// Run PSTD benchmark
fn run_pstd_benchmark(
    grid: &Grid,
    medium: &HomogeneousMedium,
    time_steps: usize,
) -> BenchmarkResult {
    let config = PSTDConfig::default();
    let pstd_source = PSTDSource::default();
    let mut solver = PSTDSolver::new(config, grid.clone(), medium, pstd_source).unwrap();

    let dt = 1e-7; // PSTD time step
    let start = std::time::Instant::now();

    // Run simulation
    for _ in 0..time_steps {
        black_box(solver.step_forward(dt).unwrap());
    }

    let execution_time = start.elapsed();

    // Calculate metrics
    let final_field = solver.pressure_field();
    let memory_usage = final_field.len() * std::mem::size_of::<f64>();
    let final_energy = calculate_energy(final_field);
    let stability_metric = calculate_stability(final_field);

    BenchmarkResult {
        solver_name: "PSTD".to_string(),
        grid_size: (grid.nx, grid.ny, grid.nz),
        time_steps,
        execution_time,
        memory_usage,
        final_energy,
        stability_metric,
    }
}

/// Calculate total energy in field (simplified)
fn calculate_energy(field: ArrayView3<f64>) -> f64 {
    field.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Calculate stability metric
fn calculate_stability(field: ArrayView3<f64>) -> f64 {
    let has_nan = field.iter().any(|&x| x.is_nan());
    let has_inf = field.iter().any(|&x| x.is_infinite());

    if has_nan || has_inf {
        0.0
    } else {
        // RMS gradient as stability proxy
        let mut gradient_sum = 0.0;
        let count = field.iter().count();

        // Simple central difference gradient
        for i in 1..field.nrows() - 1 {
            for j in 1..field.ncols() - 1 {
                for k in 1..field.dim().2 - 1 {
                    let gx = (field[[i+1, j, k]] - field[[i-1, j, k]]).abs();
                    let gy = (field[[i, j+1, k]] - field[[i, j-1, k]]).abs();
                    let gz = (field[[i, j, k+1]] - field[[i, j, k-1]]).abs();
                    gradient_sum += (gx + gy + gz) / 3.0;
                }
            }
        }

        let avg_gradient = gradient_sum / count as f64;
        1.0 / (1.0 + avg_gradient) // Lower gradient = more stable
    }
}

/// Benchmark different grid sizes
fn bench_grid_sizes(c: &mut Criterion) {
    let config = BenchmarkConfig::default();

    let mut group = c.benchmark_group("grid_sizes");

    for &(nx, ny, nz) in &config.grid_sizes {
        let (grid, medium, source) = setup_plane_wave_problem((nx, ny, nz));
        let time_steps = 10; // Short benchmark

        group.bench_with_input(
            BenchmarkId::new("FDTD", format!("{}x{}x{}", nx, ny, nz)),
            &(grid.clone(), medium.clone(), source.clone(), time_steps),
            |b, (grid, medium, source, &time_steps)| {
                b.iter(|| run_fdtd_benchmark(grid, medium, source, time_steps));
            }
        );

        group.bench_with_input(
            BenchmarkId::new("PSTD", format!("{}x{}x{}", nx, ny, nz)),
            &(grid.clone(), medium.clone(), time_steps),
            |b, (grid, medium, &time_steps)| {
                b.iter(|| run_pstd_benchmark(grid, medium, time_steps));
            }
        );
    }

    group.finish();
}

/// Benchmark different time step counts
fn bench_time_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("time_steps");

    for &time_steps in &[10, 50, 100] {
        let (grid, medium, source) = setup_plane_wave_problem((64, 64, 32)); // Medium size

        group.bench_with_input(
            BenchmarkId::new("FDTD", time_steps),
            &(grid.clone(), medium.clone(), source.clone(), time_steps),
            |b, (grid, medium, source, &time_steps)| {
                b.iter(|| run_fdtd_benchmark(grid, medium, source, time_steps));
            }
        );

        group.bench_with_input(
            BenchmarkId::new("PSTD", time_steps),
            &(grid.clone(), medium.clone(), time_steps),
            |b, (grid, medium, &time_steps)| {
                b.iter(|| run_pstd_benchmark(grid, medium, time_steps));
            }
        );
    }

    group.finish();
}

/// Benchmark solver accuracy and stability
fn bench_accuracy_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_stability");

    let (grid, medium, source) = setup_plane_wave_problem((64, 64, 32));
    let time_steps = 50;

    // Run benchmarks and collect results for analysis
    group.bench_function("FDTD_accuracy", |b| {
        b.iter(|| {
            let result = run_fdtd_benchmark(&grid, &medium, &source, time_steps);
            black_box(result.final_energy);
            black_box(result.stability_metric);
        });
    });

    group.bench_function("PSTD_accuracy", |b| {
        b.iter(|| {
            let result = run_pstd_benchmark(&grid, &medium, time_steps);
            black_box(result.final_energy);
            black_box(result.stability_metric);
        });
    });

    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    for &(nx, ny, nz) in &[(32, 32, 32), (64, 64, 64), (96, 96, 96)] {
        let (grid, medium, source) = setup_plane_wave_problem((nx, ny, nz));
        let time_steps = 5; // Very short

        group.bench_with_input(
            BenchmarkId::new("FDTD_memory", format!("{}x{}x{}", nx, ny, nz)),
            &(grid.clone(), medium.clone(), source.clone(), time_steps),
            |b, (grid, medium, source, &time_steps)| {
                b.iter(|| {
                    let result = run_fdtd_benchmark(grid, medium, source, time_steps);
                    black_box(result.memory_usage);
                });
            }
        );

        group.bench_with_input(
            BenchmarkId::new("PSTD_memory", format!("{}x{}x{}", nx, ny, nz)),
            &(grid.clone(), medium.clone(), time_steps),
            |b, (grid, medium, &time_steps)| {
                b.iter(|| {
                    let result = run_pstd_benchmark(grid, medium, time_steps);
                    black_box(result.memory_usage);
                });
            }
        );
    }

    group.finish();
}

/// Run comprehensive solver comparison
fn comprehensive_solver_comparison(c: &mut Criterion) {
    println!("🔬 Running Comprehensive Solver Comparison");
    println!("===========================================");

    let problems = vec![
        ("plane_wave", (64, 64, 32)),
        ("gaussian_pulse", (128, 8, 8)),
        ("spherical_wave", (48, 48, 48)),
    ];

    let mut results = Vec::new();

    for (problem_name, grid_size) in problems {
        println!("\n📊 Benchmarking: {} (grid: {}x{}x{})", problem_name, grid_size.0, grid_size.1, grid_size.2);

        let (grid, medium, source) = setup_plane_wave_problem(grid_size);
        let time_steps = 20;

        // FDTD benchmark
        let fdtd_result = run_fdtd_benchmark(&grid, &medium, &source, time_steps);
        results.push(fdtd_result);

        // PSTD benchmark
        let pstd_result = run_pstd_benchmark(&grid, &medium, time_steps);
        results.push(pstd_result);

        // Compare results
        let fdtd_energy = results.last().unwrap().final_energy;
        let pstd_energy = results[results.len() - 2].final_energy;

        let energy_diff = ((fdtd_energy - pstd_energy) / fdtd_energy.abs().max(pstd_energy.abs())).abs();

        println!("  FDTD: {:.2}ms, energy: {:.2e}, stability: {:.3f}",
                results[results.len() - 2].execution_time.as_millis(),
                fdtd_energy,
                results[results.len() - 2].stability_metric);

        println!("  PSTD: {:.2}ms, energy: {:.2e}, stability: {:.3f}",
                results.last().unwrap().execution_time.as_millis(),
                pstd_energy,
                results.last().unwrap().stability_metric);

        println!("  Energy difference: {:.2e} ({:.2}%)",
                (fdtd_energy - pstd_energy).abs(),
                energy_diff * 100.0);

        if energy_diff > 0.01 {
            println!("  ⚠️  SIGNIFICANT ENERGY DIFFERENCE DETECTED!");
        }
    }

    // Summary statistics
    println!("\n📈 SUMMARY STATISTICS");
    println!("====================");

    let fdtd_times: Vec<_> = results.iter()
        .filter(|r| r.solver_name == "FDTD")
        .map(|r| r.execution_time.as_millis())
        .collect();

    let pstd_times: Vec<_> = results.iter()
        .filter(|r| r.solver_name == "PSTD")
        .map(|r| r.execution_time.as_millis())
        .collect();

    if !fdtd_times.is_empty() && !pstd_times.is_empty() {
        let avg_fdtd: f64 = fdtd_times.iter().sum::<u128>() as f64 / fdtd_times.len() as f64;
        let avg_pstd: f64 = pstd_times.iter().sum::<u128>() as f64 / pstd_times.len() as f64;

        println!("Average execution time:");
        println!("  FDTD: {:.1}ms", avg_fdtd);
        println!("  PSTD: {:.1}ms", avg_pstd);
        println!("  Ratio: {:.2}x", avg_pstd / avg_fdtd);

        if avg_pstd / avg_fdtd > 2.0 {
            println!("  ⚠️  PSTD significantly slower - check implementation");
        }
    }

    // Criterion benchmark for this comprehensive test
    c.bench_function("comprehensive_comparison", |b| {
        b.iter(|| {
            let (grid, medium, source) = setup_plane_wave_problem((48, 48, 24));
            let fdtd_result = run_fdtd_benchmark(&grid, &medium, &source, 10);
            let pstd_result = run_pstd_benchmark(&grid, &medium, 10);
            black_box((fdtd_result, pstd_result));
        });
    });
}

criterion_group! {
    name = comparative_benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(5))
        .warm_up_time(std::time::Duration::from_secs(1));
    targets =
        bench_grid_sizes,
        bench_time_steps,
        bench_accuracy_stability,
        bench_memory_usage,
        comprehensive_solver_comparison
}

criterion_main!(comparative_benches);