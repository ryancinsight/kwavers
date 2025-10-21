//! Critical path performance benchmarks for Kwavers
//!
//! **Purpose**: Establish baseline performance metrics for optimization tracking
//! **Methodology**: Criterion statistical benchmarking with confidence intervals
//! **Literature**: Shewhart (1931) Statistical Quality Control
//!
//! **Design Rationale (CoT-ToT-GoT)**:
//! - CoT: Benchmark creation → execution → analysis → baseline establishment
//! - ToT: Branch on measurement strategies (wall time ✅ vs cycles ❌ vs instructions ❌)
//! - GoT: Connect performance metrics across modules for holistic optimization

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kwavers::{
    medium::{CoreMedium, HomogeneousMedium},
    solver::fdtd::finite_difference::FiniteDifference,
    Grid,
};
use ndarray::Array3;

/// Benchmark FDTD finite difference operations (critical inner loop)
///
/// **Critical Path**: This is executed every timestep for every grid point
/// **Literature**: Taflove & Hagness (2005) "FDTD Computational Electromagnetics"
fn bench_fdtd_derivatives(c: &mut Criterion) {
    let mut group = c.benchmark_group("fdtd_derivatives");

    // Test different spatial orders (2nd, 4th, 6th)
    for order in [2, 4, 6] {
        let fd = FiniteDifference::new(order).expect("Valid order");

        // Test different grid sizes (small, medium, large)
        for size in [32, 64, 128] {
            let _grid = Grid::new(size, size, size, 0.001, 0.001, 0.001).expect("Grid creation");
            let field = Array3::<f64>::ones((size, size, size));
            let spacing = 0.001;

            group.bench_with_input(
                BenchmarkId::new(format!("order_{}", order), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let deriv = fd
                            .compute_derivative(&field.view(), black_box(0), spacing)
                            .expect("Derivative computation");
                        black_box(deriv)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark k-space operator computations
///
/// **Critical Path**: K-space operations required for PSTD and spectral methods
/// **Literature**: Cooley & Tukey (1965) "FFT Algorithm"
fn bench_kspace_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("kspace_operators");

    // Test different grid sizes
    for size in [32, 64, 128, 256] {
        let grid = Grid::new(size, size, size, 0.001, 0.001, 0.001).expect("Grid creation");

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let kx = grid.compute_kx();
                let ky = grid.compute_ky();
                let kz = grid.compute_kz();
                black_box((kx, ky, kz))
            })
        });
    }

    group.finish();
}

/// Benchmark grid indexing and coordinate conversion
///
/// **Rationale**: Index conversions happen frequently in source/sensor placement
fn bench_grid_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_operations");

    let grid = Grid::new(128, 128, 128, 0.001, 0.001, 0.001).expect("Grid creation");

    // Benchmark indices to coordinates conversion
    group.bench_function("indices_to_coordinates", |b| {
        b.iter(|| {
            for i in (0..128).step_by(8) {
                for j in (0..128).step_by(8) {
                    for k in (0..128).step_by(8) {
                        let coords =
                            grid.indices_to_coordinates(black_box(i), black_box(j), black_box(k));
                        black_box(coords);
                    }
                }
            }
        })
    });

    // Benchmark coordinates to indices conversion
    group.bench_function("coordinates_to_indices", |b| {
        b.iter(|| {
            for x in [0.01, 0.05, 0.10] {
                for y in [0.01, 0.05, 0.10] {
                    for z in [0.01, 0.05, 0.10] {
                        let indices =
                            grid.coordinates_to_indices(black_box(x), black_box(y), black_box(z));
                        black_box(indices);
                    }
                }
            }
        })
    });

    // Benchmark physical property queries
    group.bench_function("physical_queries", |b| {
        b.iter(|| {
            let vol = grid.volume();
            let cell_vol = grid.cell_volume();
            let size = grid.physical_size();
            black_box((vol, cell_vol, size))
        })
    });

    group.finish();
}

/// Benchmark medium property access patterns
///
/// **Critical Path**: Medium properties accessed every cell every timestep
/// **Pattern**: Cache-friendly sequential access vs random access
fn bench_medium_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("medium_access");

    let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001).expect("Grid creation");
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.1, 1.0, &grid);

    // Sequential access (cache-friendly)
    group.bench_function("sequential_access", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..64 {
                for j in 0..64 {
                    for k in 0..64 {
                        sum += medium.density(black_box(i), black_box(j), black_box(k));
                        sum += medium.sound_speed(black_box(i), black_box(j), black_box(k));
                    }
                }
            }
            black_box(sum)
        })
    });

    // Strided access (cache-unfriendly)
    group.bench_function("strided_access", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for k in 0..64 {
                for j in 0..64 {
                    for i in 0..64 {
                        sum += medium.density(black_box(i), black_box(j), black_box(k));
                        sum += medium.sound_speed(black_box(i), black_box(j), black_box(k));
                    }
                }
            }
            black_box(sum)
        })
    });

    group.finish();
}

/// Benchmark array operations on 3D fields
///
/// **Critical Path**: Field updates happen every timestep
fn bench_field_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_operations");

    for size in [64, 128] {
        let mut field1 = Array3::<f64>::ones((size, size, size));
        let field2 = Array3::<f64>::ones((size, size, size));
        let scalar = 0.5;

        // Benchmark scalar multiplication
        group.bench_with_input(BenchmarkId::new("scalar_multiply", size), &size, |b, _| {
            b.iter(|| {
                field1.mapv_inplace(|x| x * black_box(scalar));
            })
        });

        // Benchmark element-wise addition
        group.bench_with_input(BenchmarkId::new("element_add", size), &size, |b, _| {
            b.iter(|| {
                field1
                    .iter_mut()
                    .zip(field2.iter())
                    .for_each(|(a, &b)| *a += black_box(b));
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fdtd_derivatives,
    bench_kspace_operators,
    bench_grid_operations,
    bench_medium_access,
    bench_field_operations
);
criterion_main!(benches);
