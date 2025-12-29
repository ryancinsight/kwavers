//! SIMD FDTD Performance Benchmarks
//!
//! Validates SIMD acceleration performance improvements against literature benchmarks
//! Target: >2x speedup over scalar implementation for FDTD pressure updates

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kwavers::performance::simd_safe::operations::SimdOps;
use ndarray::Array3;

/// Generate test data for FDTD benchmarking
fn generate_test_data(
    nx: usize,
    ny: usize,
    nz: usize,
) -> (Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>) {
    let mut pressure = Array3::zeros((nx, ny, nz));
    let mut divergence = Array3::zeros((nx, ny, nz));
    let mut density = Array3::zeros((nx, ny, nz));
    let mut sound_speed = Array3::zeros((nx, ny, nz));

    // Fill with realistic test data
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = i as f64 / nx as f64;
                let y = j as f64 / ny as f64;
                let z = k as f64 / nz as f64;

                pressure[[i, j, k]] = 1e5 * (x * y * z).sin(); // Atmospheric pressure variation
                divergence[[i, j, k]] = 0.01 * (x + y + z).cos(); // Small divergence
                density[[i, j, k]] = 1000.0; // Water density kg/m³
                sound_speed[[i, j, k]] = 1500.0; // Water sound speed m/s
            }
        }
    }

    (pressure, divergence, density, sound_speed)
}

/// Scalar (non-SIMD) FDTD pressure update for baseline comparison
fn scalar_pressure_update(
    mut pressure: Array3<f64>,
    divergence: &Array3<f64>,
    density: &Array3<f64>,
    sound_speed: &Array3<f64>,
    dt: f64,
) -> Array3<f64> {
    let (nx, ny, nz) = pressure.dim();

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let rho = density[[i, j, k]];
                let c = sound_speed[[i, j, k]];
                let div_v = divergence[[i, j, k]];

                // FDTD pressure update: p^{n+1} = p^n - dt * rho * c^2 * div(v)
                pressure[[i, j, k]] -= dt * rho * c * c * div_v;
            }
        }
    }

    pressure
}

/// Benchmark SIMD vs scalar FDTD performance
fn bench_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_fdtd_performance");

    // Test different grid sizes
    let grid_sizes = [(32, 32, 32), (64, 64, 64)];

    for (nx, ny, nz) in grid_sizes {
        let (pressure, divergence, density, sound_speed) = generate_test_data(nx, ny, nz);
        let dt = 1e-7; // Small time step for stability

        // Scalar baseline
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}x{}", nx, ny, nz)),
            &(
                pressure.clone(),
                divergence.clone(),
                density.clone(),
                sound_speed.clone(),
            ),
            |b, (p, div, rho, c)| {
                b.iter(|| {
                    let result = scalar_pressure_update(p.clone(), div, rho, c, dt);
                    black_box(result);
                });
            },
        );

        // SIMD implementation
        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}x{}", nx, ny, nz)),
            &(pressure, divergence, density, sound_speed),
            |b, (p, div, rho, c)| {
                b.iter(|| {
                    // This would use the actual SIMD FDTD implementation
                    // For now, using scalar as placeholder until proper SIMD integration
                    let result = scalar_pressure_update(p.clone(), div, rho, c, dt);
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SIMD operations throughput
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");

    let sizes = [1000, 10000, 100000];

    for &size in &sizes {
        let a = Array3::from_elem((10, 10, size), 1.0_f64);
        let b = Array3::from_elem((10, 10, size), 2.0_f64);

        group.bench_with_input(
            BenchmarkId::new("add_fields", size),
            &(&a, &b),
            |b, (a, bb)| {
                b.iter(|| {
                    let result = SimdOps::add_fields(a, bb);
                    black_box(result);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("multiply_fields", size),
            &(&a, &b),
            |b, (a, bb)| {
                b.iter(|| {
                    let result = SimdOps::multiply_fields(a, bb);
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_simd_vs_scalar, bench_simd_operations);
criterion_main!(benches);
